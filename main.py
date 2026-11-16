"""
Multi-tenant restaurant voice agent (Cartesia Line SDK).

One deployed agent serves every restaurant on the platform platform. On each
call, the agent POSTs the dialed number (`CallRequest.to`) to the platform's
resolve endpoint, which returns the businessId + per-tenant system prompt
+ introduction + transfer phone. All tool URLs are then built with that
businessId, and the caller's phone is forwarded on every relevant tool
call for caller-recognition / phone-based verification.

Nothing in this file is hard-coded to any single restaurant. Adding a new
tenant is a number import on the platform side; no redeploy needed here.

Environment variables (set in the Cartesia dashboard → Environment):
  ANTHROPIC_API_KEY        — Claude API key (required)
  PLATFORM_BASE_URL          — platform API host (one for all tenants)
  PLATFORM_WEBHOOK_SECRET  — shared secret; must match the platform's PLATFORM_WEBHOOK_SECRET
"""
import os
import json
from typing import Annotated, Optional

import httpx
from loguru import logger

import asyncio
import re
import time

from line import AgentEndCall, AgentSendText, AgentUpdateCall, CallEnded, CallStarted
from line.events import AgentToolReturned, UserDtmfSent, UserTurnEnded, UserTurnStarted

# If the caller's first turn lands within this many seconds of call start, the
# proactive greeting cannot have finished playing (it was barged over) — the
# wrapper re-emits it once, uninterruptible. Greetings are ~2s of audio.
GREETING_GRACE_S = 3.5

# Mid-call STT/TTS mutation on the caller's turn truncated replies, so this
# was off and a language change had to wait for the LLM to call
# switch_voice_to_language — a whole model round-trip late. On call
# CA58c1d02… (2026-08-14) that cost a French caller 36 of his 62 seconds in a
# German voice, and he hung up. Back on, but only on a CONFIDENT change
# (_detect_caller_language now needs a clear margin), at most
# MAX_LANGUAGE_SWITCHES times and never twice inside the cooldown — so it
# cannot oscillate on mixed/noisy ASR the way the old any-nonzero-score rule
# could. AUTO_LANGUAGE_SWITCH=0 restores the tool-only path.
AUTO_LANGUAGE_SWITCH = os.getenv("AUTO_LANGUAGE_SWITCH", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _env_num(name: str, default: float, cast=float):
    """Tunable from the Cartesia dashboard, but a typo there must never take
    the phone line down at boot — bad values fall back to the default."""
    try:
        return cast(str(os.getenv(name, "")).strip() or default)
    except (TypeError, ValueError):
        logger.warning(f"[cartesia] ignoring unparseable {name}={os.getenv(name)!r}; using {default}")
        return cast(default)


MAX_LANGUAGE_SWITCHES = _env_num("MAX_LANGUAGE_SWITCHES", 3, int)
LANGUAGE_SWITCH_COOLDOWN_S = _env_num("LANGUAGE_SWITCH_COOLDOWN_S", 8.0)

# Caller silent after we finish talking → re-prompt, then hang up.
# 7s re-prompted people who were merely thinking: three "Sind Sie noch da?"
# fit inside one 62s call, one of them landing while the caller was already
# answering. Each further prompt now waits IDLE_BACKOFF longer (12s, 18s,
# 24s), so a hesitant caller gets ~54s before we hang up instead of ~21s.
# Lost-EOT: UserTurnStarted fired (barge-in cancelled us) but UserTurnEnded
# never arrived — that is the Agil 27s dead-air. The UserTurnStarted run
# filter keeps a sleep alive after barge-in so we can still speak.
IDLE_S = _env_num("IDLE_S", 12.0)
IDLE_BACKOFF = _env_num("IDLE_BACKOFF", 0.5)
IDLE_LOST_EOT_S = _env_num("IDLE_LOST_EOT_S", 10.0)
IDLE_AFTER_FAREWELL_S = _env_num("IDLE_AFTER_FAREWELL_S", 4.0)
MAX_SILENCE_PROMPTS = _env_num("MAX_SILENCE_PROMPTS", 3, int)

# Silence is not always absence. Two cases where the other side is THINKING
# about a question we just asked, and the guest-leg impatience above is wrong:
#
#   supervisor leg — we called the chef with a Rückfrage ("Was sag ich dem
#     Gast?"). They check the book, ask the kitchen, count covers. At 12s/18s
#     we asked "Sind Sie noch da?" twice and hung up at ~30s, which drops the
#     consult and sends the guest a non-answer.
#   guest leg on hold — the Rückfrage is open and the guest is waiting on US.
#     Asking whether THEY are still there is backwards; they know they are.
#
# Both get a longer timer, a line that fits the situation, and more patience
# before the line is dropped. Still bounded — an open silent call costs money.
SUPERVISOR_IDLE_S = _env_num("SUPERVISOR_IDLE_S", 30.0)
CONSULT_HOLD_IDLE_S = _env_num("CONSULT_HOLD_IDLE_S", 20.0)
MAX_WAIT_PROMPTS = _env_num("MAX_WAIT_PROMPTS", 4, int)

# Consult tools on the GUEST leg. Their result tells us whether the Rückfrage
# is still open, which is what puts the guest on hold.
_CONSULT_TOOL_NAMES = {"ask_supervisor", "check_supervisor", "relay_to_supervisor"}
# supervisor-consult.service.ts `check()` statuses that mean "still open".
_CONSULT_OPEN_STATUSES = {"pending", "relay", "ringing", "relaying", "asking", "confirming"}

_FAREWELL_RE = re.compile(
    r"\b("
    r"tschüss|tschüs|tschüß|tschau|ciao|ade|"
    r"auf\s+wiederhören|auf\s+wiedersehen|"
    r"bye|goodbye|good\s*bye|"
    r"au\s+revoir|à\s+bientôt|a\s+bientot|"
    r"arrivederci|addio|"
    r"bis\s+dann|bis\s+gleich|mach['’]?s\s+gut"
    r")\b",
    re.IGNORECASE,
)
_BARE_ACK_RE = re.compile(
    r"^\s*("
    r"ja|jo|ju|yes|yeah|yep|oui|si|sí|"
    r"ok|okay|okey|oki|"
    r"mhm|mmh|aha|ach\s+so|"
    r"genau|richtig|klar|sure|"
    r"danke|dankeschön|merci|thanks|thank\s+you|"
    r"bitte|gut|super|perfekt"
    r")[\s.!?…,]*$",
    re.IGNORECASE,
)

# end_call tool: default description treats any "signals they're finished"
# as a hangup, which fired on a bare "Ja".
END_CALL_DESCRIPTION = (
    "Ends the conversation and hangs up. "
    "Use ONLY after the caller said an explicit goodbye "
    "(tschüss, auf wiederhören, bye, goodbye, au revoir, ciao). "
    "Do NOT use on ja / yes / oui / okay / danke / merci / thanks / mhm / genau "
    "or any other short confirmation. "
    "Do NOT use during a transfer or call-back. "
    "Say one warm farewell in the same turn, then call this. Irreversible."
)

_IDLE_REPROMPT_BY_LANG = {
    "de": "Sind Sie noch da?",
    "en": "Are you still there?",
    "fr": "Vous êtes toujours là ?",
    "it": "C'è ancora qualcuno?",
    "es": "¿Sigue ahí?",
    "nl": "Bent u er nog?",
    "pl": "Czy nadal pani/pan tam jest?",
    "tr": "Hâlâ orada mısınız?",
    "pt": "Ainda está aí?",
    "hu": "Még ott van?",
    "ru": "Вы ещё на линии?",
    "ar": "هل ما زلت على الخط؟",
}
_IDLE_GOODBYE_BY_LANG = {
    "de": "Auf Wiederhören!",
    "en": "Goodbye!",
    "fr": "Au revoir !",
    "it": "Arrivederci!",
    "es": "¡Adiós!",
    "nl": "Tot ziens!",
    "pl": "Do widzenia!",
    "tr": "Hoşça kalın!",
    "pt": "Adeus!",
    "hu": "Viszontlátásra!",
    "ru": "До свидания!",
    "ar": "مع السلامة!",
}


# Spoken to the CHEF on the Rückfrage leg when they go quiet. "Sind Sie noch
# da?" implies they wandered off; they are usually mid-thought or counting
# covers. This says "take your time" without asking a question back.
_SUPERVISOR_NUDGE_BY_LANG = {
    "de": "Lassen Sie sich Zeit — ich warte.",
    "en": "Take your time — I'm waiting.",
    "fr": "Prenez votre temps — j'attends.",
    "it": "Con calma — resto in linea.",
}
# Spoken to the GUEST while the Rückfrage is still open. They are holding for
# us, so the line has to say WE are still working, not ask after them.
_CONSULT_HOLD_BY_LANG = {
    "de": "Ich bin noch dran — ich frage gerade nach.",
    "en": "Still with you — I'm just checking now.",
    "fr": "Je suis toujours là — je me renseigne.",
    "it": "Sono ancora qui — sto verificando.",
    "es": "Sigo aquí — lo estoy consultando.",
    "nl": "Ik ben er nog — ik vraag het even na.",
    "pl": "Jestem — właśnie dopytuję.",
    "tr": "Buradayım — hemen soruyorum.",
    "pt": "Continuo aqui — estou a confirmar.",
    "hu": "Itt vagyok — épp utánanézek.",
    "ru": "Я на линии — уточняю.",
    "ar": "ما زلت معك — أتحقق الآن.",
}


def _lang_line(table: dict, language: str, fallback: str = "en") -> str:
    code = (language or "").strip().lower()[:2]
    return table.get(code) or table.get(fallback) or next(iter(table.values()))


def _is_farewell(text: str) -> bool:
    return bool(_FAREWELL_RE.search(text or ""))


def _is_bare_ack(text: str) -> bool:
    return bool(_BARE_ACK_RE.match((text or "").strip()))


# Tino 2026-08-19: barge-in "Yep" while the closed-Friday line was still
# playing was taken as agreement to Wednesday. Replay unheard closed facts
# instead of continuing. Not a prompt — wrapper state + the tool payload.
_CLOSED_TOOL_NAMES = {"check_availability"}
_REPLAY_CLOSED = {
    "de": "Wir haben am {date} geschlossen{reason}.{next}",
    "en": "We're closed on {date}{reason}.{next}",
    "fr": "Nous sommes fermés le {date}{reason}.{next}",
}
_REPLAY_REASON = {
    "de": {
        "HOLIDAY": " — Feiertag",
        "SICK": " wegen Krankheit im Team",
        "MISSING_STAFF": " — Personalmangel",
        "PRIVATE_EVENT": " — privates Event",
        "MAINTENANCE": " wegen Wartung",
    },
    "en": {
        "HOLIDAY": " — it's a holiday",
        "SICK": " — the team is out sick",
        "MISSING_STAFF": " — we're short-staffed",
        "PRIVATE_EVENT": " — private event",
        "MAINTENANCE": " — maintenance",
    },
    "fr": {
        "HOLIDAY": " — jour férié",
        "SICK": " — l'équipe est malade",
        "MISSING_STAFF": " — manque de personnel",
        "PRIVATE_EVENT": " — événement privé",
        "MAINTENANCE": " — maintenance",
    },
}
_REPLAY_NEXT = {
    "de": " Nächster offener Tag ist {next}.",
    "en": " Next open day is {next}.",
    "fr": " Prochain jour d'ouverture : {next}.",
}


def _closed_facts_from_payload(tool_name: str, result, tool_args=None) -> Optional[dict]:
    """Pull closed-day facts out of a check_availability tool result."""
    if (tool_name or "") not in _CLOSED_TOOL_NAMES:
        return None
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            return None
    if not isinstance(result, dict):
        return None
    if result.get("available") is True or result.get("open") is True:
        return None
    if result.get("available") is not False and result.get("open") is not False:
        if not result.get("blockedAllDay"):
            return None
    args = tool_args if isinstance(tool_args, dict) else {}
    message = str(result.get("message") or "").strip()
    if not message and not result.get("blockedAllDay"):
        return None
    return {
        "message": message,
        "date": str(result.get("date") or args.get("date") or ""),
        "reason": str(result.get("closureReason") or "").upper(),
        "next": str(result.get("nextOpenDate") or ""),
    }


def _closed_facts_from_tool(out) -> Optional[dict]:
    if not isinstance(out, AgentToolReturned):
        return None
    return _closed_facts_from_payload(
        getattr(out, "tool_name", "") or "",
        getattr(out, "result", None),
        getattr(out, "tool_args", None),
    )


def _replay_closed_line(facts: dict, language: str) -> str:
    """One uninterruptible sentence for the facts the caller did not hear."""
    if not facts:
        return ""
    lang = (language or "en").strip().lower()[:2]
    date = (facts.get("date") or "").strip()
    message = (facts.get("message") or "").strip()
    if not date:
        return message
    tmpl = _REPLAY_CLOSED.get(lang) or _REPLAY_CLOSED["en"]
    reason_bit = (_REPLAY_REASON.get(lang) or _REPLAY_REASON["en"]).get(
        (facts.get("reason") or "").upper(), ""
    )
    nxt = (facts.get("next") or "").strip()
    next_bit = ""
    if nxt:
        next_bit = (_REPLAY_NEXT.get(lang) or _REPLAY_NEXT["en"]).format(next=nxt)
    return tmpl.format(date=date, reason=reason_bit, next=next_bit).strip()


def _should_block_end_call(caller_text: str, agent_text: str) -> bool:
    """True when end_call is the model hanging up on a confirmation, not a goodbye."""
    if _is_farewell(caller_text):
        return False
    if _is_bare_ack(caller_text) and not _is_farewell(agent_text):
        return True
    return False


def _idle_reprompt(language: str) -> str:
    return _lang_line(_IDLE_REPROMPT_BY_LANG, language)


def _idle_goodbye(language: str) -> str:
    return _lang_line(_IDLE_GOODBYE_BY_LANG, language)


def _supervisor_nudge(language: str) -> str:
    return _lang_line(_SUPERVISOR_NUDGE_BY_LANG, language, "de")


def _consult_hold_line(language: str) -> str:
    return _lang_line(_CONSULT_HOLD_BY_LANG, language)


def _consult_still_open(result) -> bool:
    """True while a Rückfrage is unresolved, from a consult tool's payload."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (TypeError, ValueError):
            # ask_supervisor answers with a bare id / free text: it only ever
            # STARTS a consult, so an unparseable reply means still open.
            return True
    if not isinstance(result, dict):
        return True
    status = str(result.get("status") or "").strip().lower()
    if not status:
        return True
    return status in _CONSULT_OPEN_STATUSES


# A FIRST utterance that is nothing but a (possibly repeated) bare salutation
# means the caller heard silence and is probing the line — the intro audio
# never played, whatever the timing says. Only AMBIGUOUS fillers live here:
# "allô"/"hallo"/"hi"/"hello" are one sound across languages, so they must
# not flip the voice. Unique openers (bonjour, bonsoir, guten Tag) are
# handled in _UNIQUE_OPENERS and DO switch.
_BARE_SALUTATION_RE = re.compile(
    r"^\s*(?:ja[,!.\s]*|und[,!.\s]*)?"
    r"(?:hallo|hall o|all[oô]|allo|hi|hey|hello|servus|moin|pronto)"
    r"[\s?!.,…]*"
    r"(?:hallo[\s?!.,…]*)*$",
    re.IGNORECASE,
)

# First-word openers that exist in only one of our languages. "Bonjour" is
# never "Hallo"; leaving it in the bare-salutation ban is what kept a
# French caller on the German clone for the whole greeting.
_UNIQUE_OPENERS: dict[str, re.Pattern[str]] = {
    "de": re.compile(
        r"^\s*(?:guten\s+(?:tag|abend|morgen)|grüß\s*gott|gruess\s*gott)\b",
        re.IGNORECASE,
    ),
    "fr": re.compile(r"^\s*(?:bonjour|bonsoir)\b", re.IGNORECASE),
    "en": re.compile(r"^\s*(?:good\s+(?:morning|afternoon|evening))\b", re.IGNORECASE),
    "it": re.compile(r"^\s*(?:buongiorno|buonasera)\b", re.IGNORECASE),
    "es": re.compile(r"^\s*(?:buenos\s+d[ií]as|buenas\s+(?:tardes|noches))\b", re.IGNORECASE),
    "nl": re.compile(r"^\s*(?:goedendag|goedenavond|goedemorgen)\b", re.IGNORECASE),
    "pl": re.compile(r"^\s*(?:dzie[nń]\s+dobry|dobry\s+wiecz[oó]r)\b", re.IGNORECASE),
    "tr": re.compile(r"^\s*(?:merhaba|g[uü]nayd[iı]n|iyi\s+ak[sş]amlar)\b", re.IGNORECASE),
    "pt": re.compile(r"^\s*(?:bom\s+dia|boa\s+(?:tarde|noite))\b", re.IGNORECASE),
    "hu": re.compile(r"^\s*(?:j[oó]\s+napot|j[oó]\s+est[eé]t)\b", re.IGNORECASE),
    "ru": re.compile(r"^\s*(?:здравствуйте|добрый\s+(?:день|вечер))\b", re.IGNORECASE),
    "ar": re.compile(r"^\s*(?:السلام\s+عليكم|مرحبا|أهلا)\b"),
}

# Country prefix → likely spoken language. Mixed countries (BE/CH/NL) omitted
# so we never guess wrong; those fall through to the greeting language.
_CALLER_LANG_BY_CC: dict[str, str] = {
    "1": "en",
    "33": "fr",
    "39": "it",
    "43": "de",
    "44": "en",
    "49": "de",
    "61": "en",
    "64": "en",
    "353": "en",
}


def _language_from_phone(number: str) -> Optional[str]:
    digits = re.sub(r"\D", "", number or "")
    if not digits:
        return None
    for cc in sorted(_CALLER_LANG_BY_CC, key=len, reverse=True):
        if digits.startswith(cc):
            return _CALLER_LANG_BY_CC[cc]
    return None


def _is_greeting_only(text: str) -> bool:
    """True when the caller only greeted back (the intro already did that)."""
    t = (text or "").strip()
    if not t:
        return True
    if _BARE_SALUTATION_RE.match(t):
        return True
    for pat in _UNIQUE_OPENERS.values():
        if pat.search(t):
            rest = pat.sub("", t).strip(" \t.!,?…;:-")
            if not rest:
                return True
    return False


# Last-call 2026-08-19: German +49 said "Allô" (STT of Hallo under a French
# greeting) and the model filled with "Mhm, bonjour! Qu'est-ce que je peux
# faire pour vous?" - same bot line as "how can I help", plus the German
# clone speaking French. Strip these before they reach TTS.
_HELP_OFFER_RE = re.compile(
    r"[^.!?\n]*\b("
    r"qu['’]est[- ]ce que je (peux|puis) (faire pour vous|vous aider)"
    r"|que puis[- ]je (faire|vous aider)"
    r"|comment (puis[- ]je|je peux) vous aider"
    r"|je peux vous aider"
    r"|how can i help( you)?"
    r"|what can i do for you"
    r"|wie kann ich (ihnen |dir )?helfen"
    r"|was kann ich für sie tun"
    r"|womit kann ich (ihnen )?dienen"
    r"|kann ich (ihnen |dir )?(sonst noch )?helfen"
    r")\b[^.!?\n]*[.!?…]?",
    re.IGNORECASE,
)
_REGREET_PREFIX_RE = re.compile(
    r"^\s*(?:mhm|mmh|okay|ok|ja)[\s,!.…-]*"
    r"(?:bonjour|bonsoir|hallo|guten\s+(?:tag|abend|morgen)|hello|hi|hey|"
    r"bienvenue(?:\s+chez[\w\s]*)?)[\s!.?,…]*",
    re.IGNORECASE,
)
_REGREET_ONLY_RE = re.compile(
    r"^\s*(?:mhm|mmh|okay|ok|ja)?[\s,!.…-]*"
    r"(?:bonjour|bonsoir|hallo|guten\s+(?:tag|abend|morgen)|hello|hi|hey|"
    r"bienvenue(?:\s+chez[\w\s]*)?)"
    r"[\s!.?,…]*$",
    re.IGNORECASE,
)


# LLM stage directions ("[switching to German…]") are not TTS markup.
# Keep [pause] — Cartesia consumes that silently.
_STAGE_DIR_RE = re.compile(r"\[(?!pause\b)[^\]]*\]", re.IGNORECASE)

# Cartesia TTS has no fade: interruptible audio is a hard cut, mid-phoneme.
# Wind-down = finish the sentence already on the wire (interruptible=False)
# and never start the next one if the caller has taken the floor.
_SENTENCE_RE = re.compile(
    r"(.+?[.!?…](?:[\"'»”’])?)(?=\s+|[A-ZÄÖÜÁÀÂÉÈÊÍÓÚÄÅÇÑ«\"']|$)",
    re.DOTALL,
)


def _split_off_sentences(buf: str) -> tuple[list[str], str]:
    """Pull complete spoken sentences off a streaming buffer. Remainder stays."""
    if not buf:
        return [], ""
    units: list[str] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(buf):
        if m.start() != pos:
            break
        piece = (m.group(1) or "").strip()
        if piece:
            units.append(piece)
        pos = m.end()
        while pos < len(buf) and buf[pos] in " \t\r\n":
            pos += 1
    return units, buf[pos:]


def _speak(text: str) -> AgentSendText:
    """One uninterruptible phrase — finishes instead of clicking off."""
    return AgentSendText(text=text, interruptible=False)


def _scrub_bot_talk(text: str) -> str:
    """Drop re-greet + how-can-I-help. Empty string means do not speak."""
    if not text:
        return ""
    # Keep TTS markup, scrub the audible words.
    t = text
    t = _STAGE_DIR_RE.sub("", t)
    t = _HELP_OFFER_RE.sub("", t)
    t = _REGREET_PREFIX_RE.sub("", t)
    # A leftover that is only a greeting / mhm / pause / emotion is silence.
    audible = _MARKUP_RE.sub("", t)
    if _REGREET_ONLY_RE.match(audible) or not _has_speech(audible):
        return ""
    return t.strip()
from line.llm_agent import (
    LlmAgent,
    LlmConfig,
    ToolEnv,
    end_call,
    knowledge_base,
    loopback_tool,
    passthrough_tool,
    send_dtmf,
    transfer_call,
    voicemail,
)
from line.voice_agent_app import AgentEnv, CallRequest, PreCallResult, VoiceAgentApp


# Heuristic language markers for restaurant phone calls. Used to auto-switch
# voice_id + STT/TTS language BEFORE the LLM turn — no tool round-trip.
# Restaurant-phone vocabulary for every language the app offers. Only langs
# in THIS tenant's supported/voices set are scored (candidates).
_LANG_MARKERS: dict[str, re.Pattern[str]] = {
    "de": re.compile(
        r"\b(ich|wir|mein|meine|bitte|danke|dankeschön|reservier\w*|tisch|uhr|"
        r"personen|guten\s+tag|guten\s+abend|möchte|möchten|heute|morgen|"
        r"abend|mittag|verfügbar|buchung|anrufen|sprechen)\b",
        re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(je|nous|vous|tu|suis|sommes|bonjour|bonsoir|merci|réserv\w*|reservation|"
        r"personnes?|voudrais|voulez|s['’]il|aujourd|demain|soir|midi|"
        r"disponible|réserver|parler|[eê]tes|est-ce|ouvert|ferm[ée]s?)\b",
        re.IGNORECASE,
    ),
    "en": re.compile(
        r"\b(i|we|my|please|thanks|thank\s+you|reserv\w*|people|persons?|"
        r"would|today|tomorrow|tonight|book|booking|available|speak)\b",
        re.IGNORECASE,
    ),
    "it": re.compile(
        r"\b(io|noi|buongiorno|buonasera|grazie|prenot\w*|tavolo|persone|"
        r"vorrei|oggi|domani|sera|disponib\w*|parlare)\b",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"\b(yo|nosotros|buenos\s+d[ií]as|buenas\s+tardes|gracias|reserv\w*|"
        r"mesa|personas?|quisiera|hoy|ma[nñ]ana|noche|disponible|hablar)\b",
        re.IGNORECASE,
    ),
    "nl": re.compile(
        r"\b(ik|wij|alsjeblieft|dankjewel|bedankt|reserver\w*|tafel|personen|"
        r"vandaag|morgen|avond|beschikbaar|spreken|graag)\b",
        re.IGNORECASE,
    ),
    "pl": re.compile(
        r"\b(ja|my|prosz[eę]|dzi[eę]kuj[eę]|rezerw\w*|stolik|os[oó]b\w*|"
        r"dzisiaj|jutro|wiecz[oó]r|dost[eę]pn\w*|m[oó]wi[cć])\b",
        re.IGNORECASE,
    ),
    "tr": re.compile(
        r"\b(ben|biz|l[uü]tfen|te[sş]ekk[uü]r|rezerv\w*|masa|ki[sş]i|"
        r"bug[uü]n|yar[iı]n|ak[sş]am|m[uü]sait|konu[sş])\b",
        re.IGNORECASE,
    ),
    "pt": re.compile(
        r"\b(eu|n[oó]s|por\s+favor|obrigad[oa]|reserv\w*|mesa|pessoas?|"
        r"hoje|amanh[aã]|noite|dispon[ií]vel|falar)\b",
        re.IGNORECASE,
    ),
    "hu": re.compile(
        r"\b([eé]n|mi|k[eé]rem|k[oö]sz[oö]n[oö]m|foglal\w*|asztal|szem[eé]ly|"
        r"ma|holnap|este|el[eé]rhet[oő]|besz[eé]l)\b",
        re.IGNORECASE,
    ),
    "ru": re.compile(
        r"\b(я|мы|пожалуйста|спасибо|брон\w*|стол|человек|"
        r"сегодня|завтра|вечер|свободн\w*|говорить)\b",
        re.IGNORECASE,
    ),
    "ar": re.compile(
        r"(أنا|نحن|من\s+فضلك|شكرا|حجز|طاولة|أشخاص|اليوم|غدا|مساء|متاح)",
    ),
}

# Short decisive words. "table" is shared EN/FR so it is not a strong marker.
_STRONG_MARKERS: dict[str, re.Pattern[str]] = {
    "de": re.compile(
        r"\b(nein|doch|danke|bitte|tisch|uhr|abend|reservierung|guten\s+(?:tag|abend|morgen))\b",
        re.IGNORECASE,
    ),
    "fr": re.compile(
        r"\b(oui|non|merci|voilà|voila|personne|quelqu\w*|réserver|reserver|bonjour|bonsoir)\b",
        re.IGNORECASE,
    ),
    "en": re.compile(r"\b(yes|no|thanks|please|booking|tonight)\b", re.IGNORECASE),
    "it": re.compile(r"\b(sì|si|grazie|prego|tavolo|prenotare|buongiorno|buonasera)\b", re.IGNORECASE),
    "es": re.compile(r"\b(s[ií]|gracias|mesa|reservar|buenos\s+d[ií]as)\b", re.IGNORECASE),
    "nl": re.compile(r"\b(ja|nee|alsjeblieft|tafel|reserveren|goedendag)\b", re.IGNORECASE),
    "pl": re.compile(r"\b(tak|nie|prosz[eę]|stolik|rezerwacj\w*|dzie[nń]\s+dobry)\b", re.IGNORECASE),
    "tr": re.compile(r"\b(evet|hay[iı]r|l[uü]tfen|masa|rezervasyon|merhaba)\b", re.IGNORECASE),
    "pt": re.compile(r"\b(sim|n[aã]o|obrigad[oa]|mesa|reservar|bom\s+dia)\b", re.IGNORECASE),
    "hu": re.compile(r"\b(igen|nem|k[eé]rem|asztal|foglal[aá]s)\b", re.IGNORECASE),
    "ru": re.compile(r"\b(да|нет|спасибо|стол|бронь|здравствуйте)\b", re.IGNORECASE),
    "ar": re.compile(r"(نعم|لا|شكرا|طاولة|حجز|مرحبا)"),
}

# Script / diacritic boosts. Weak overlaps (ü, á) only count if markers already hit.
_ORTHOGRAPHY: list[tuple[str, str, int]] = [
    ("de", r"[äöüß]", 2),
    ("fr", r"[àâçéèêëïîôùûœæ]", 2),
    ("es", r"[ñ¿¡]", 2),
    ("tr", r"[ıİşşğĞ]", 2),
    ("pl", r"[ąćęłńóśźż]", 2),
    ("pt", r"[ãõ]", 2),
    ("hu", r"[őű]", 2),
    ("cs", r"[řěůďťň]", 2),
    ("sv", r"[å]", 2),
    ("ru", r"[а-яё]", 3),
    ("uk", r"[іїєґ]", 3),
    ("el", r"[α-ωάέήίόύώ]", 3),
    ("ar", r"[\u0600-\u06FF]", 3),
]


def _detect_caller_language(
    text: str, candidates: set[str], current: str, hint: str = ""
) -> Optional[str]:
    """Return a new language code if the caller clearly switched; else None.

    Conservative: short/ambiguous turns (bare 'Hallo') keep the current voice
    so we never thrash on greetings shared across languages. A turn under 6
    characters needs a strong marker, and the winner must beat the runner-up
    by 2 — mixed turns like "Sind-vous une personne?" only move us when one
    language is genuinely dominant.

    Unique openers (bonjour / bonsoir / guten Tag) switch immediately ONLY
    when they are the whole greeting. "Je suis François Müller, guten Tag."
    is French with a German hello tacked on — that must NOT steal the French
    clone (2026-08-22: German clone then spoke French = German accent).

    Diacritics (ü in Müller) only boost a language that already has word
    markers. A name must never flip the voice on its own.
    """
    t = (text or "").strip()
    if not t or not candidates:
        return None
    cur = (current or "")[:2]

    # Unique opener at the START, and nothing else real after it.
    unique_hits = [
        lang
        for lang, pat in _UNIQUE_OPENERS.items()
        if lang in candidates and pat.search(t)
    ]
    if len(unique_hits) == 1:
        opener = unique_hits[0]
        rest = _UNIQUE_OPENERS[opener].sub("", t).strip(" \t.!,?…;:-")
        if (not rest or _BARE_SALUTATION_RE.match(rest)) and opener != cur:
            return opener
        # Mixed ("Bonjour, ich möchte…") — fall through to scoring.

    # A turn that is nothing but an ambiguous salutation decides nothing:
    # "allô" and "hallo" are one sound. Country code is NOT a hint - a tenant
    # gets French speakers on +49 numbers.
    if _BARE_SALUTATION_RE.match(t):
        return None
    if len(t) < 4:
        return None
    scores: dict[str, int] = {c: 0 for c in candidates}
    for lang, pat in _LANG_MARKERS.items():
        if lang not in scores:
            continue
        scores[lang] = len(pat.findall(t))
    for lang, pat in _STRONG_MARKERS.items():
        if lang in scores and pat.search(t):
            scores[lang] += 2
    for lang, pat, boost in _ORTHOGRAPHY:
        if lang not in scores:
            continue
        # No IGNORECASE: Turkish ı/İ would otherwise match ASCII "i" in every
        # French/English sentence and tie the score.
        if not re.search(pat, t):
            continue
        # Weak diacritics only count once the language already has a word hit.
        if scores[lang] <= 0:
            continue
        scores[lang] += boost

    best_lang = max(scores, key=lambda k: scores[k])
    best_score = scores[best_lang]
    if best_score <= 0:
        return None
    if best_lang == cur:
        return None
    # A very short turn is only ever decisive through a strong marker (+2).
    if len(t) < 6 and best_score < 2:
        return None
    # Mixed utterance: current language is clearly present (≥2, so a lone
    # shared cognate like "personen" does not freeze us). Otherwise a
    # trailing "guten Tag" after "Je suis…" steals the German clone and the
    # model keeps answering in French (wrong accent).
    if cur in scores and scores[cur] >= 2:
        return None
    # Require a clear winner to avoid flipping on mixed / noisy ASR.
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) > 1 and best_score - ordered[1] < 2:
        return None
    return best_lang


def _guess_spoken_language(text: str, candidates: set[str]) -> Optional[str]:
    """Language of text we are ABOUT to speak. Used to pick the matching clone
    so French words never go out through the German voice."""
    t = (text or "").strip()
    if not t or not candidates:
        return None
    scores: dict[str, int] = {c: 0 for c in candidates}
    for lang, pat in _LANG_MARKERS.items():
        if lang in scores:
            scores[lang] = len(pat.findall(t))
    for lang, pat in _STRONG_MARKERS.items():
        if lang in scores and pat.search(t):
            scores[lang] += 2
    for lang, pat in _UNIQUE_OPENERS.items():
        if lang in scores and pat.search(t):
            scores[lang] += 3
    best_lang = max(scores, key=lambda k: scores[k])
    if scores[best_lang] <= 0:
        return None
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) > 1 and ordered[1] > 0 and scores[best_lang] - ordered[1] < 2:
        return None
    return best_lang


def _jev_hint_from(resp) -> str:
    """jev_hint from a live-transcript response, or "" (fail open, never raises).
    The API only sends it when JEV_ENABLED and JEV_HINTS are on."""
    try:
        if resp is None or getattr(resp, "status_code", 0) != 200:
            return ""
        h = (resp.json() or {}).get("jev_hint")
        return h.strip()[:400] if isinstance(h, str) else ""
    except Exception:
        return ""


class _CallLanguage:
    """The language + voice this call is CURRENTLY speaking, shared by the
    wrapper and the switch_voice_to_language tool.

    The tool is built in get_agent, before the wrapper exists, and used to
    yield an AgentUpdateCall and nothing else — so the wrapper never learned
    the caller had moved to French and went on picking German from every
    _BY_LANG table it owns. On CA58c1d02… that is why a French call was asked
    "Sind Sie noch da?" three times; the same table would have said goodbye in
    German too. One object, both directions.
    """

    __slots__ = ("code", "voice_id")

    def __init__(self, code: str = "", voice_id: str = ""):
        self.code = (code or "").strip().lower()[:2]
        self.voice_id = (voice_id or "").strip()


class _TenantVoiceAgent:
    """Per-tenant call wrapper: cloned-voice switch + greeting re-arm.

    VOICE — switch TTS to the tenant's cloned voice on the CALLER'S FIRST
    UTTERANCE. Findings from the Jul 2 2026 dead-greeting hunt (dual-channel +
    platform recordings, all confirmed by ear): a per-call voice override
    applied at call start — via pre_call config.tts, via AgentUpdateCall on
    CallStarted, or combined with a delayed AgentSendText greeting — silences
    EVERYTHING the agent emits before the caller's first utterance. The text
    is logged, no audio plays. The only proactive greeting that has ever
    produced audio is the platform's native introduction with NO voice
    override in flight (the pre-multi-tenant June behavior).

    So the greeting plays through the native intro path in the agent's default
    dashboard voice, and we swap to the tenant's clone the moment the caller
    first speaks — the pipeline is warm by then (same AgentUpdateCall the
    mid-call language switcher uses). For the primary tenant the default voice
    IS their clone, so the swap is a no-op there.

    MULTI-LANGUAGE — `voices` is {lang: voice_id}. On every caller turn we
    heuristically detect language and emit AgentUpdateCall *before* the LLM
    runs, so the reply already uses the correct voice with zero tool latency.

    GREETING RE-ARM — the SDK marks `_introduction_sent = True` BEFORE
    yielding the intro AgentSendText on CallStarted (llm_agent.py:254-258).
    If the caller makes noise the instant the call connects ("Hallo?" while
    picking up), the harness cancels the opening turn, the intro is dropped,
    and the SDK never re-emits it → silent greeting (the exact pre-June
    `_IntroDelayedAgent` bug pattern, now inside the SDK). We track whether
    the intro actually flowed through this wrapper during the CallStarted
    turn; if not, we speak the tenant greeting once at the start of the first
    real turn. The system prompt's GREETING rule ("never greet yourself")
    keeps the LLM from doubling it.
    """

    def __init__(
        self,
        inner: LlmAgent,
        voice_id: str,
        language: str,
        greeting: str = "",
        business_id: str = "",
        call_id: str = "",
        voices: Optional[dict] = None,
        lang_state: Optional[_CallLanguage] = None,
        lang_hint: str = "",
        languages: Optional[list] = None,
        leg: str = "guest",
    ):
        self.inner = inner
        # "guest" (inbound caller) or "supervisor" (outbound Rückfrage to the
        # chef). The supervisor leg exists to wait for a considered answer, so
        # it must never be as impatient as the guest leg.
        self.leg = (leg or "guest").strip().lower()
        # Must come first: `language` / `voice_id` are properties over it, and
        # __getattr__ would otherwise chase a missing _lang into the inner agent.
        self._lang = lang_state or _CallLanguage(language, voice_id)
        self.voice_id = (voice_id or "").strip()
        self.language = (language or "").strip().lower()[:2]
        self.greeting = (greeting or "").strip()
        self.business_id = (business_id or "").strip()
        self.call_id = (call_id or "").strip()
        raw_voices = voices if isinstance(voices, dict) else {}
        self.voices: dict[str, str] = {
            str(k).lower()[:2]: str(v).strip()
            for k, v in raw_voices.items()
            if k and v and str(v).strip()
        }
        # Prefer map entry for the primary language when present.
        if self.language and self.voices.get(self.language):
            self.voice_id = self.voices[self.language]
        elif self.voice_id and self.language and self.language not in self.voices:
            self.voices[self.language] = self.voice_id
        extra = languages if isinstance(languages, (list, tuple, set)) else []
        self.candidates: set[str] = set(self.voices.keys()) | {
            str(x).lower()[:2] for x in extra if x and str(x).strip()
        }
        if self.language:
            self.candidates.add(self.language)
        self._switched = False
        self._greeting_emitted = False
        self._regreeted = False
        self._greeting_pending = False
        self._call_started_at = 0.0
        self._last_caller_text = ""
        # Jev classifier hint from the API (previous caller turn). Advisory only.
        self._jev_hint = ""
        self._last_agent_text = ""
        self._silence_count = 0
        self._switch_count = 0
        self._last_switch_at = 0.0
        self._ended = False
        self._must_repeat: Optional[dict] = None
        self._was_cut_off = False
        # Guest leg only: a Rückfrage is open, so the guest is on hold.
        self._consulting = False
        # The language TTS is currently configured for. Tracks what we last
        # told Cartesia, which is not always the conversation language.
        self._spoken_lang = self.language
        self.lang_hint = (lang_hint or "").strip().lower()[:2]

    # The call's language and voice live in the shared _CallLanguage so a
    # switch made by the LLM's tool is visible here (and vice versa).
    @property
    def language(self) -> str:
        return self._lang.code

    @language.setter
    def language(self, value: str):
        self._lang.code = (value or "").strip().lower()[:2]

    @property
    def voice_id(self) -> str:
        return self._lang.voice_id

    @voice_id.setter
    def voice_id(self, value: str):
        self._lang.voice_id = (value or "").strip()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def _post_transcript(self, role: str, text: str):
        """LIVE TRANSCRIPT — fire-and-forget push of one conversation line to
        the platform, so the owner app can show the call as it happens. Never blocks
        or fails the call: a lost line just means a gap in the live view (the
        full transcript still arrives via the normal post-call path)."""
        text = (text or "").strip()
        if not text or not self.business_id or not WEBHOOK_SECRET:
            return

        async def _send():
            try:
                r = await _http().post(
                    platform_api(self.business_id, "tools", "live-transcript"),
                    headers={"X-Cartesia-Secret": WEBHOOK_SECRET},
                    json={"call_id": self.call_id, "role": role, "text": text, "surface": "phone"},
                    timeout=3.0,
                )
                if role == "caller":
                    self._jev_hint = _jev_hint_from(r)
            except Exception:
                pass  # best-effort by design

        try:
            asyncio.get_running_loop().create_task(_send())
        except RuntimeError:
            pass

    def _language_switch_allowed(self) -> bool:
        """Rate-limit the auto path. A caller who mixes languages ("Sind-vous
        une personne?") must never make us flip voice every turn: each flip
        re-configures STT/TTS mid-conversation, which is what truncated
        replies when this path was first tried."""
        if self._switch_count >= MAX_LANGUAGE_SWITCHES:
            logger.info(
                f"[cartesia] language switch suppressed call={self.call_id} "
                f"— already switched {self._switch_count}x"
            )
            return False
        if self._last_switch_at and (time.monotonic() - self._last_switch_at) < LANGUAGE_SWITCH_COOLDOWN_S:
            logger.info(
                f"[cartesia] language switch suppressed call={self.call_id} "
                f"— within {LANGUAGE_SWITCH_COOLDOWN_S:.0f}s cooldown"
            )
            return False
        return True

    def _emit_voice_update(self, voice_id: str, language: str):
        """Yield-able AgentUpdateCall for voice + language (STT/TTS).

        Language-only is allowed: a tenant may support ES/TR/PL without a
        dedicated clone. Switching STT/TTS language still fixes the accent.
        """
        vid = (voice_id or "").strip()
        lang = (language or "").strip().lower()[:2]
        # Single choke point for "what language is TTS set to right now", so
        # _pin_clone can tell a real no-op from a needed language-only switch.
        if vid and lang:
            self._spoken_lang = lang
            return AgentUpdateCall(voice_id=vid, language=lang)
        if vid:
            return AgentUpdateCall(voice_id=vid)
        if lang:
            self._spoken_lang = lang
            return AgentUpdateCall(language=lang)
        return None

    def _pin_clone(self, language: str):
        """Make TTS actually speak `language` — right clone AND right language.

        Two ways this goes wrong if it is skipped:

        Lines we generate ourselves (idle re-prompt, goodbye, filler, closed-day
        replay) have a language that is not in doubt, so it must never be
        guessed. On call CA5706468… a French sentence moved the clone to FR and
        the German "Sind Sie noch da?" went out French-accented.

        And a tenant may offer a language it has no CLONE for — a tenant
        answers ES/TR/PL on a DE/FR/EN voice map. Gating on the clone meant
        those got no update at all, so Spanish went out through whichever clone
        was live, in that clone's language. There is no Spanish clone to switch
        to, but switching the STT/TTS language alone still fixes the accent, so
        a missing clone must never mean "leave it where it was".
        """
        lang = (language or "").strip().lower()[:2]
        if not lang:
            return None
        vid = self.voices.get(lang) or ""
        already = lang == self._spoken_lang and (not vid or vid == self.voice_id)
        if already:
            return None
        logger.info(
            f"[cartesia] tts → {lang} ({vid or 'language-only, no clone'}) "
            f"call={self.call_id}"
        )
        # vid may be "": _emit_voice_update then sends a language-only update.
        upd = self._emit_voice_update(vid, lang)
        if upd is not None and vid:
            self.voice_id = vid
        return upd

    def _waiting_on_an_answer(self) -> bool:
        """True when this silence is someone THINKING about a question we
        asked, not someone who has drifted off.

        The whole supervisor leg is one Rückfrage, so it is always waiting.
        The guest leg only waits while a consult is open.
        """
        if self.leg == "supervisor":
            return True
        return self._consulting

    def _idle_line(self) -> str:
        """What to say into the silence, for THIS leg and THIS situation."""
        if self.leg == "supervisor":
            return _supervisor_nudge(self.language)
        if self._consulting:
            return _consult_hold_line(self.language)
        return _idle_reprompt(self.language)

    async def _idle_watch(self):
        """Stay inside the current process() task until the caller speaks
        (UserTurnStarted/Ended cancels us) or we re-prompt / hang up."""
        while not self._ended:
            waiting = self._waiting_on_an_answer()
            if _is_farewell(self._last_agent_text):
                delay = IDLE_AFTER_FAREWELL_S
            elif waiting:
                base = SUPERVISOR_IDLE_S if self.leg == "supervisor" else CONSULT_HOLD_IDLE_S
                delay = base * (1.0 + IDLE_BACKOFF * self._silence_count)
            else:
                # Back off after each unanswered prompt: someone who did not
                # react to "Sind Sie noch da?" is thinking, reading a calendar
                # or talking to someone in the room — asking again on the same
                # short timer is what made the agent feel impatient.
                delay = IDLE_S * (1.0 + IDLE_BACKOFF * self._silence_count)
            await asyncio.sleep(delay)
            if self._ended:
                return
            self._silence_count += 1
            farewell = _is_farewell(self._last_agent_text)
            # A farewell always ends the call. Otherwise the budget depends on
            # whether we are waiting on a thought: never drop the chef mid-
            # Rückfrage (that loses the consult), never drop a guest who is
            # holding for us. Both stay bounded so the line cannot hang open.
            limit = MAX_WAIT_PROMPTS if waiting else MAX_SILENCE_PROMPTS
            if farewell or self._silence_count >= limit:
                if not farewell:
                    bye = _idle_goodbye(self.language)
                    pin = self._pin_clone(self.language)
                    if pin is not None:
                        yield pin
                    yield _speak(bye)
                    self._post_transcript("agent", bye)
                    self._last_agent_text = bye
                logger.info(
                    f"[cartesia] idle hangup call={self.call_id} leg={self.leg} "
                    f"silence={self._silence_count} waiting={waiting} farewell={farewell}"
                )
                yield AgentEndCall(reason="agent_ended", interruptible=False)
                self._ended = True
                return
            prompt = self._idle_line()
            logger.warning(
                f"[cartesia] idle re-prompt call={self.call_id} leg={self.leg} "
                f"waiting={waiting} #{self._silence_count} after {delay:.0f}s"
            )
            pin = self._pin_clone(self.language)
            if pin is not None:
                yield pin
            yield _speak(prompt)
            self._post_transcript("agent", prompt)
            self._last_agent_text = prompt

    def _gate_end_call(self, out):
        """Swallow end_call on a bare Ja/okay; otherwise force uninterruptible hangup."""
        if not isinstance(out, AgentEndCall):
            return out
        if _should_block_end_call(self._last_caller_text, self._last_agent_text):
            logger.warning(
                f"[cartesia] blocked end_call call={self.call_id} "
                f"caller={self._last_caller_text!r} agent={self._last_agent_text[:60]!r}"
            )
            return None
        self._ended = True
        if out.interruptible:
            return out.model_copy(update={"interruptible": False})
        return out

    async def process(self, env, event):
        if isinstance(event, CallEnded) or self._ended:
            self._ended = True
            if isinstance(event, CallEnded):
                async for out in self.inner.process(env, event):
                    yield out
            return

        # Barge-in cancelled the previous turn. If UserTurnEnded never arrives
        # (the Agil case), this sleep is the only thing that will speak again.
        if isinstance(event, UserTurnStarted):
            if self._must_repeat:
                self._was_cut_off = True
            self._silence_count = 0
            await asyncio.sleep(IDLE_LOST_EOT_S)
            if self._ended:
                return
            logger.warning(
                f"[cartesia] lost end-of-turn call={self.call_id} "
                f"— no UserTurnEnded after {IDLE_LOST_EOT_S:.0f}s; re-prompting"
            )
            prompt = _idle_reprompt(self.language)
            yield _speak(prompt)
            self._post_transcript("agent", prompt)
            self._last_agent_text = prompt
            self._silence_count = 1
            async for out in self._idle_watch():
                yield out
            return

        if isinstance(event, CallStarted):
            self._call_started_at = time.monotonic()
            # PASS THROUGH the SDK's native CallStarted introduction (proactive
            # greeting). TEST: does it play audibly (even if absent from the
            # Cartesia transcript)? Recording analysis decides.
            async for out in self.inner.process(env, event):
                gated = self._gate_end_call(out)
                if gated is None:
                    continue
                if isinstance(gated, AgentSendText):
                    if gated.interruptible:
                        gated = gated.model_copy(update={"interruptible": False})
                    if gated.text == self.greeting:
                        self._greeting_emitted = True
                        self._post_transcript("agent", gated.text)
                        self._last_agent_text = gated.text
                yield gated
            # Greeting audio is already yielded. Switch STT/TTS to the start
            # language NOW so a +49 "Hallo" is not transcribed as "Allô" by a
            # French recognizer (2026-08-19 last call).
            if self.voice_id and not self._switched:
                self._switched = True
                logger.info(
                    f"[cartesia] start voice after intro → voice_id={self.voice_id} "
                    f"lang={self.language or '-'}"
                )
                update = self._emit_voice_update(self.voice_id, self.language)
                if update is not None:
                    yield update
            if not self._ended:
                async for out in self._idle_watch():
                    yield out
            return

        # Live transcript: the caller's words arrive on UserTurnEnded.
        caller_text = ""
        extra_ctx = None
        if isinstance(event, UserTurnEnded):
            self._silence_count = 0
            caller_text = " ".join(
                getattr(c, "content", "") for c in (event.content or []) if getattr(c, "content", "")
            ).strip()
            if caller_text:
                self._last_caller_text = caller_text
                self._post_transcript("caller", caller_text)
            # Caller talked over the closed-day line. A yep/ja is NOT agreement
            # to the alternative date they did not hear — replay the facts.
            if self._was_cut_off and self._must_repeat:
                replay = _replay_closed_line(self._must_repeat, self.language) or (
                    self._must_repeat.get("message") or ""
                )
                if replay and (
                    _is_bare_ack(caller_text)
                    or _is_greeting_only(caller_text)
                    or not caller_text
                ):
                    logger.warning(
                        f"[cartesia] replaying cut-off closed facts call={self.call_id} "
                        f"caller={caller_text[:40]!r}"
                    )
                    self._was_cut_off = False
                    self._must_repeat = None
                    pin = self._pin_clone(self.language)
                    if pin is not None:
                        yield pin
                    yield _speak(replay)
                    self._post_transcript("agent", replay)
                    self._last_agent_text = replay
                    if not self._ended:
                        async for out in self._idle_watch():
                            yield out
                    return
                extra_ctx = (
                    "CUT OFF: your last spoken reply did not finish. The caller did "
                    "not hear it, so a yes is not confirmation. Repeat these facts "
                    f"first in one short sentence: {replay}"
                )
                self._was_cut_off = False

        # GREETING is delivered PROACTIVELY by the platform's native CallStarted
        # introduction (see the CallStarted branch above + get_agent's
        # proactive_intro). The agent speaks first, before the caller says
        # anything — confirmed on a real call 2026-07-06. The one hard
        # requirement: the intro text MUST end in terminal punctuation (. ! ? …)
        # or the TTS endpointer never flushes it and the platform drops it
        # silently (this is why "a tenant bonjour" stayed silent while
        # "Platform 68, guten Tag!" played — same code, same agent). get_agent
        # enforces that. We therefore no longer inject the greeting in front of
        # the first response (that produced a DOUBLE greeting once native works).

        # Mid-call language → voice switch BEFORE the LLM turn, so the reply is
        # generated AND spoken in the caller's language with no tool round-trip.
        # This is the same mutation point the tenant-voice swap below already
        # uses; the guards live in _language_switch_allowed.
        if AUTO_LANGUAGE_SWITCH and caller_text and self.candidates:
            detected = _detect_caller_language(
                caller_text, self.candidates, self.language
            )
            if detected and self._language_switch_allowed():
                new_vid = self.voices.get(detected) or self.voice_id
                logger.info(
                    f"[cartesia] auto language switch → {detected} "
                    f"(voice_id={new_vid or '-'}) from caller: {caller_text[:60]!r}"
                )
                self.language = detected
                if self.voices.get(detected):
                    self.voice_id = new_vid
                self._switched = True
                self._switch_count += 1
                self._last_switch_at = time.monotonic()
                update = self._emit_voice_update(new_vid, detected)
                if update is not None:
                    yield update
                lock = (
                    f"LANGUAGE LOCK: the caller is speaking {detected}. "
                    f"Reply in {detected} only — every word, including numbers and dates. "
                    "Do not mix languages. Do not stay in the greeting language."
                )
                extra_ctx = f"{extra_ctx}\n{lock}" if extra_ctx else lock

        if self.voice_id and not self._switched and not isinstance(event, CallEnded):
            self._switched = True
            logger.info(
                f"[cartesia] tenant voice (first caller event) → voice_id={self.voice_id} lang={self.language or '-'}"
            )
            update = self._emit_voice_update(self.voice_id, self.language)
            if update is not None:
                yield update

        # Greeting already played. A short Allô/Hallo/Bonjour is them greeting
        # back - do NOT let the LLM invent "how can I help" (2026-08-19).
        if isinstance(event, UserTurnEnded) and _is_greeting_only(caller_text):
            logger.info(
                f"[cartesia] greeting-only turn, staying quiet call={self.call_id} "
                f"caller={caller_text[:40]!r}"
            )
            if not self._ended:
                async for out in self._idle_watch():
                    yield out
            return

        # Stream complete sentences to TTS as they form. Each one is
        # uninterruptible: Cartesia has no fade, so barge-in would otherwise
        # click the audio off mid-word (Tino 2026-08-19: 4 of 8 turns cut).
        # Scrub each sentence before it is spoken so "how can I help" never
        # reaches the speaker. Incomplete leftovers are dropped on cancel —
        # already-sent sentences keep playing to the end.
        pending = ""
        agent_chunks: list = []
        completed = False
        scrubbed_away = False

        def _flush(units: list[str]) -> list:
            nonlocal scrubbed_away
            events = []
            for unit in units:
                spoken = _scrub_bot_talk(unit)
                if not spoken:
                    if unit.strip():
                        scrubbed_away = True
                        logger.warning(
                            f"[cartesia] scrubbed bot-talk call={self.call_id} "
                            f"was={unit[:100]!r}"
                        )
                    continue
                # Never send French words through the German clone (or vice
                # versa). Match TTS voice_id + language to THIS sentence.
                # An inconclusive guess ("Gerne!", "August.", "Sind Sie noch
                # da?") must fall back to the CONVERSATION language, not to
                # whatever clone the previous sentence left us on — otherwise
                # one French sentence strands every short German line after it
                # on the French clone (call CA5706468…, 2026-08-22).
                guessed = _guess_spoken_language(spoken, self.candidates) or self.language
                upd = self._pin_clone(guessed)
                if upd is not None:
                    events.append(upd)
                agent_chunks.append(spoken)
                self._last_agent_text = spoken
                self._post_transcript("agent", spoken)
                events.append(_speak(spoken))
            return events

        try:
            # Jev hint rides along from the previous turn; nothing waits on it.
            jev = getattr(self, "_jev_hint", "") if caller_text else ""
            if jev:
                extra_ctx = f"{extra_ctx}\n{jev}" if extra_ctx else jev
            inner_kwargs = {"context": extra_ctx} if extra_ctx else {}
            async for out in self.inner.process(env, event, **inner_kwargs):
                facts = _closed_facts_from_tool(out)
                if facts:
                    self._must_repeat = facts
                    self._was_cut_off = False
                # A consult tool result is what tells us the guest is on hold
                # (or no longer is). Without this the guest gets "Sind Sie noch
                # da?" while waiting on the chef we are still ringing.
                if isinstance(out, AgentToolReturned):
                    tname = getattr(out, "tool_name", "") or ""
                    if tname in _CONSULT_TOOL_NAMES:
                        was = self._consulting
                        self._consulting = _consult_still_open(
                            getattr(out, "result", None)
                        )
                        if was != self._consulting:
                            logger.info(
                                f"[cartesia] consult {'open' if self._consulting else 'closed'} "
                                f"call={self.call_id} via {tname}"
                            )
                        if not self._consulting:
                            self._silence_count = 0
                if isinstance(out, AgentSendText) and out.text:
                    pending += out.text
                    units, pending = _split_off_sentences(pending)
                    for ev in _flush(units):
                        yield ev
                    continue
                gated = self._gate_end_call(out)
                if gated is None:
                    continue
                if isinstance(gated, AgentSendText) and gated.interruptible:
                    gated = gated.model_copy(update={"interruptible": False})
                yield gated
            completed = True
        except asyncio.CancelledError:
            if self._must_repeat:
                self._was_cut_off = True
                logger.warning(
                    f"[cartesia] cut off before closed facts landed call={self.call_id}"
                )
            return

        leftover = _scrub_bot_talk(pending)
        if leftover:
            agent_chunks.append(leftover)
            self._last_agent_text = leftover
            self._post_transcript("agent", leftover)
            yield _speak(leftover)
        elif pending.strip():
            scrubbed_away = True

        # They heard this turn. A later "yep" can mean agreement.
        if completed and agent_chunks:
            self._was_cut_off = False
            self._must_repeat = None

        # Tool-only completions leave the caller in dead air (Jul 22 "Hello?"
        # bug). Do NOT fill when we deliberately stayed quiet (greeting-only
        # or scrubbed how-can-I-help).
        if (
            completed
            and not _has_speech(agent_chunks)
            and isinstance(event, UserTurnEnded)
            and not self._ended
            and not scrubbed_away
        ):
            logger.warning(
                f"[cartesia] SILENT TURN call={self.call_id} — completed with no speech "
                f"(caller said: {caller_text[:80]!r}); speaking filler"
            )
            filler = _silent_turn_filler(self.language)
            if filler:
                pin = self._pin_clone(self.language)
                if pin is not None:
                    yield pin
                yield _speak(filler)
                self._post_transcript("agent", filler)
                self._last_agent_text = filler

        if completed and not self._ended:
            async for out in self._idle_watch():
                yield out


# Directives the TTS consumes without ever making a sound: emotion tags and
# the [pause] marker. A reply that is only these is dead air to the caller.
_MARKUP_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")


def _has_speech(chunks) -> bool:
    """True when the turn actually produced audible words."""
    if isinstance(chunks, str):
        chunks = [chunks]
    text = "".join(c for c in (chunks or []) if c)
    return bool(_MARKUP_RE.sub("", text).strip(" \t\r\n.,;:!?…-—–"))


# Language-aware one-liner when a turn finishes with tools and zero speech.
_SILENT_FILLER_BY_LANG = {
    "de": "Einen Moment bitte.",
    "en": "One moment.",
    "fr": "Un instant s'il vous plaît.",
    "it": "Un momento, per favore.",
    "es": "Un momento, por favor.",
    "nl": "Een moment, alstublieft.",
    "pl": "Chwileczkę.",
    "tr": "Bir dakika lütfen.",
    "pt": "Um momento, por favor.",
    "hu": "Egy pillanat.",
    "ru": "Один момент.",
    "ar": "لحظة من فضلك.",
}


def _silent_turn_filler(language: str) -> str:
    return _lang_line(_SILENT_FILLER_BY_LANG, language)



def _require_platform():
    raise RuntimeError(
        "PLATFORM_BASE_URL is not set. Point it at the platform API that resolves "
        "dialed numbers and hosts the voice tool endpoints."
    )

PLATFORM_BASE_URL = (os.environ.get("PLATFORM_BASE_URL") or "").rstrip("/")
# Common API stem. Resolve is {prefix}/resolve; per-call tools are
# {prefix}/{business_id}/tools/{slug}.
PLATFORM_TOOL_PATH_PREFIX = (os.environ.get("PLATFORM_TOOL_PATH_PREFIX") or "/api/voice").rstrip("/") or "/api/voice"
if not PLATFORM_TOOL_PATH_PREFIX.startswith("/"):
    PLATFORM_TOOL_PATH_PREFIX = "/" + PLATFORM_TOOL_PATH_PREFIX


def platform_api(*segments: str) -> str:
    """Join PLATFORM_BASE_URL + PLATFORM_TOOL_PATH_PREFIX + segments."""
    rest = "/".join(str(s).strip("/") for s in segments if s is not None and str(s) != "")
    return f"{PLATFORM_BASE_URL}{PLATFORM_TOOL_PATH_PREFIX}/{rest}" if rest else f"{PLATFORM_BASE_URL}{PLATFORM_TOOL_PATH_PREFIX}"

WEBHOOK_SECRET = os.getenv("PLATFORM_WEBHOOK_SECRET", "")
HTTP_TIMEOUT = 10.0

# One shared keep-alive client for ALL the platform round-trips (resolve, tools,
# live transcript). Every call was paying a fresh TCP+TLS handshake through
# the Cloudflare tunnel (~0.2-0.3s) — on tool calls that sits squarely in the
# caller-audible response path.
_HTTP_CLIENT: Optional[httpx.AsyncClient] = None


def _http() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120.0),
        )
    return _HTTP_CLIENT

FALLBACK_SYSTEM_PROMPT = (
    "You are a warm host taking phone calls at a restaurant. Keep replies to one short "
    "spoken sentence. If you can't identify the restaurant, apologize and ask the caller "
    "to call back in a moment."
)
FALLBACK_INTRODUCTION = "Hello!"


async def _resolve_business(to_number: str, from_number: str = "") -> Optional[dict]:
    if not WEBHOOK_SECRET:
        logger.error("PLATFORM_WEBHOOK_SECRET not set — cannot resolve tenant")
        return None
    try:
        payload = {"to": to_number}
        if from_number:
            payload["from"] = from_number
        r = await _http().post(
            platform_api("resolve"),
            json=payload,
            headers={"X-Cartesia-Secret": WEBHOOK_SECRET},
        )
        if r.status_code != 200:
            logger.error(f"resolve {to_number} → HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        if "error" in data:
            logger.warning(f"resolve {to_number} → {data['error']}")
            return None
        return data
    except Exception as e:
        logger.exception(f"resolve {to_number} failed: {e}")
        return None


def _make_tools(business_id: str, caller_phone: str, call_id: str = ""):
    """Build reservation tools. Caller phone + the Cartesia call_id are
    auto-attached to every payload so the platform can do caller-recognition and,
    crucially, link a booking to THIS exact call (concurrent calls otherwise
    can't be told apart by a time-window guess on the platform's side)."""

    def tool_url(slug: str) -> str:
        return platform_api(business_id, "tools", slug)

    async def post_tool(slug: str, payload: dict) -> str:
        if not WEBHOOK_SECRET:
            return '{"error": "Configuration error. Please transfer the caller to a human."}'
        # Auto-inject caller phone + the call id if not already in payload.
        if caller_phone and "caller_phone" not in payload:
            payload = {**payload, "caller_phone": caller_phone}
        if call_id and "call_id" not in payload:
            payload = {**payload, "call_id": call_id}
        try:
            r = await _http().post(
                tool_url(slug),
                json=payload,
                headers={"X-Cartesia-Secret": WEBHOOK_SECRET},
            )
            if r.status_code >= 500:
                logger.error(f"[{slug}] the platform {r.status_code}: {r.text[:200]}")
                return '{"error": "Sorry, I had trouble with the booking system. Could you try again?"}'
            return r.text
        except httpx.TimeoutException:
            return '{"error": "The booking system is slow right now. Could you try again?"}'
        except Exception as e:
            logger.exception(f"[{slug}] failed: {e}")
            return '{"error": "Sorry, something went wrong. Could you try again?"}'

    @loopback_tool
    async def check_availability(
        ctx: ToolEnv,
        date: Annotated[str, "Date in YYYY-MM-DD format"],
        guests: Annotated[int, "Number of guests"],
        time: Annotated[str, "Time in HH:mm 24-hour format. Empty for a day-status check."] = "",
        private_event: Annotated[
            bool,
            "True if the caller is talking about a privatisation / exclusive hire / private event on that date, not a normal table.",
        ] = False,
        name: Annotated[str, "Caller name if they said it. Empty otherwise."] = "",
    ) -> str:
        """Check whether a date (and optionally a specific time) is open for the party size.
        Call with just the date first to confirm the day is open before asking for a time.
        If they mention a privatisation / exclusive evening matching a date, set private_event=true
        — they are that client even if caller ID differs. A table request stays a stranger."""
        payload: dict = {"date": date, "guests": guests}
        if time:
            payload["time"] = time
        if private_event:
            payload["private_event"] = True
        if name:
            payload["name"] = name
        return await post_tool("check-availability", payload)

    @loopback_tool
    async def book_reservation(
        ctx: ToolEnv,
        date: Annotated[str, "Date in YYYY-MM-DD format"],
        time: Annotated[str, "Time in HH:mm 24-hour format"],
        guests: Annotated[int, "Number of guests"],
        name: Annotated[str, "Guest name as heard"],
        phone: Annotated[str, "Caller phone if explicitly provided. Empty otherwise."] = "",
        comment: Annotated[str, "Special requests or notes. Empty if none."] = "",
    ) -> str:
        """Create a reservation. Call IMMEDIATELY after check_availability confirms a slot
        AND you have the caller's name. Speaking does not create a booking — you must call this."""
        payload: dict = {"date": date, "time": time, "guests": guests, "name": name}
        if phone:
            payload["phone"] = phone
        if comment:
            payload["comment"] = comment
        return await post_tool("book-reservation", payload)

    @loopback_tool
    async def find_reservation_candidates(
        ctx: ToolEnv,
        date: Annotated[str, "Date in YYYY-MM-DD that the caller said they have a reservation on"],
        time: Annotated[str, "Time in HH:mm if the caller mentioned one. Empty otherwise."] = "",
        name: Annotated[str, "Name on the reservation if the caller said one. Empty otherwise."] = "",
        guests: Annotated[int, "Party size if the caller said one. 0 if not."] = 0,
    ) -> str:
        """Find likely matching reservations BEFORE modifying or cancelling. Use this when the
        caller wants to change or cancel a booking. Returns candidate reservation IDs ranked by
        confidence — a candidate with phoneMatch or nameMatch true is the caller's reservation.
        If exactly one candidate matches, proceed with modify_by_id / cancel_by_id; otherwise
        read back the times and ask which one. This tool only FINDS — nothing is changed until
        you also call modify_by_id or cancel_by_id."""
        payload: dict = {"date": date}
        if time:
            payload["time"] = time
        if name:
            payload["name"] = name
        if guests > 0:
            payload["guests"] = guests
        return await post_tool("find-candidates", payload)

    @loopback_tool
    async def modify_by_id(
        ctx: ToolEnv,
        reservation_id: Annotated[str, "The reservation id from find_reservation_candidates"],
        name: Annotated[str, "Name on the reservation as said by the caller (for verification)"] = "",
        new_time: Annotated[str, "New time HH:mm. Empty to keep the same time."] = "",
        new_guests: Annotated[int, "New party size. 0 to keep the same."] = 0,
        note: Annotated[str, "Optional special-request note to append. Empty for none."] = "",
    ) -> str:
        """Modify a specific reservation by id (preferred over modify_reservation when the caller
        confirmed a candidate). Verification happens via caller phone or name match. Call this in
        the SAME turn the caller confirms the change — saying the change out loud saves NOTHING;
        never tell the caller a change is done unless this tool returned success."""
        payload: dict = {"reservation_id": reservation_id}
        if name:
            payload["name"] = name
        if new_time:
            payload["new_time"] = new_time
        if new_guests > 0:
            payload["new_guests"] = new_guests
        if note:
            payload["note"] = note
        return await post_tool("modify-by-id", payload)

    @loopback_tool
    async def cancel_by_id(
        ctx: ToolEnv,
        reservation_id: Annotated[str, "The reservation id from find_reservation_candidates"],
        name: Annotated[str, "Name on the reservation as said by the caller (for verification)"] = "",
    ) -> str:
        """Cancel a specific reservation by id (preferred over cancel_reservation when the caller
        confirmed a candidate). ONLY call after the caller explicitly says cancel / stornieren /
        annuler AND answers yes to one final confirmation question."""
        payload: dict = {"reservation_id": reservation_id}
        if name:
            payload["name"] = name
        return await post_tool("cancel-by-id", payload)

    # Legacy name-based modify/cancel — still useful when the caller can't be matched
    # by phone and find_candidates returns nothing.
    @loopback_tool
    async def modify_reservation(
        ctx: ToolEnv,
        name: Annotated[str, "Name on the existing reservation"],
        existing_date: Annotated[str, "Existing date YYYY-MM-DD"],
        existing_time: Annotated[str, "Existing time HH:mm"],
        new_date: Annotated[str, "New date YYYY-MM-DD. Empty to keep."] = "",
        new_time: Annotated[str, "New time HH:mm. Empty to keep."] = "",
        new_guests: Annotated[int, "New party size. 0 to keep."] = 0,
        new_name: Annotated[str, "Corrected name (mishearing fix)."] = "",
        comment: Annotated[str, "Updated note. Empty if unchanged."] = "",
    ) -> str:
        """Change an existing reservation by name + date + time. Use when find_reservation_candidates
        didn't return a usable match. Only call after the caller has confirmed the change."""
        payload: dict = {"name": name, "existing_date": existing_date, "existing_time": existing_time}
        if new_date: payload["new_date"] = new_date
        if new_time: payload["new_time"] = new_time
        if new_guests > 0: payload["new_guests"] = new_guests
        if new_name: payload["new_name"] = new_name
        if comment: payload["comment"] = comment
        return await post_tool("modify-reservation", payload)

    @loopback_tool
    async def cancel_reservation(
        ctx: ToolEnv,
        name: Annotated[str, "Name on the reservation"],
        date: Annotated[str, "Reservation date YYYY-MM-DD"],
        time: Annotated[str, "Reservation time HH:mm if known. Empty otherwise."] = "",
    ) -> str:
        """Cancel by name + date + time. Use when find_reservation_candidates didn't return a usable
        match. ONLY call after explicit cancel/stornieren/annuler AND one final yes confirmation."""
        payload: dict = {"name": name, "date": date}
        if time:
            payload["time"] = time
        return await post_tool("cancel-reservation", payload)

    @loopback_tool
    async def request_callback(
        ctx: ToolEnv,
        reason: Annotated[str, "Short reason for the call-back — what the caller wants / their question"],
        name: Annotated[str, "Caller's name if given. Empty otherwise."] = "",
        phone: Annotated[str, "Call-back number ONLY if the caller had to say it out loud (their number was withheld/anonymous). Empty otherwise — their number is captured automatically."] = "",
    ) -> str:
        """Arrange for a colleague to call the caller back. Use this when you cannot
        help the caller yourself and they need a person — it instantly rings the team
        with the reason and number. The caller's number is usually captured
        automatically, so you do NOT need to ask for it. ONLY pass `phone` when the
        tool told you the number was missing (withheld/anonymous) and the caller then
        said their number out loud."""
        payload: dict = {"reason": reason}
        if name:
            payload["name"] = name
        if phone:
            # Caller spoke their number (theirs was withheld) — override the
            # auto-injected caller_phone so post_tool doesn't replace it.
            payload["caller_phone"] = phone
        return await post_tool("request-callback", payload)

    @loopback_tool
    async def menu_lookup(
        ctx: ToolEnv,
        query: Annotated[str, "What the caller asked about — a dish name, ingredient, or dietary need (e.g. 'vegan', 'gluten-free', 'pasta')"],
    ) -> str:
        """Search the restaurant's actual menu for items matching the caller's question.
        Use this for any menu, dish, or allergen/diet question instead of guessing —
        never invent menu items. If nothing is found, say so and offer to ask staff."""
        return await post_tool("menu-lookup", {"query": query})

    @loopback_tool
    async def drinks_lookup(
        ctx: ToolEnv,
        query: Annotated[str, "What the caller asked about — a drink name, type (wine/beer/cocktail), or allergen"],
    ) -> str:
        """Search the restaurant's actual drinks list for items matching the caller's
        question. Use this for any drinks or drinks-allergen question instead of
        guessing — never invent drinks."""
        return await post_tool("drinks-lookup", {"query": query})

    @loopback_tool
    async def catering_catalog(
        ctx: ToolEnv,
        query: Annotated[str, "What the caller wants to order — a dish or product name (e.g. 'Gratin', 'Tarte', 'Platte')"],
        category: Annotated[str, "Optional category to narrow it down (e.g. 'Salate', 'Desserts'). Empty if none."] = "",
    ) -> str:
        """Look up catering / take-away products WITH PRICES in the restaurant's webshop.
        Use this — NOT menu_lookup — whenever the caller asks what something costs to
        take away or order for a group ("was kostet ein Blech ...?"). menu_lookup is the
        dine-in menu and has no prices at all.
        Each result has `price` (per portion) and, where the shop states it, `trayPrice`
        for a whole Blech of `portionsPerTray` portions. Read `trayPrice` out EXACTLY as
        given — do NOT multiply price × portions yourself, the shop's own total differs
        for some dishes. If `trayPrice` is absent, quote only the portion price."""
        payload = {"query": query}
        if category:
            payload["category"] = category
        return await post_tool("catering-catalog", payload)

    @loopback_tool
    async def catering_quote(
        ctx: ToolEnv,
        items: Annotated[str, 'JSON list of what the caller wants, e.g. [{"product":"Gratin","persons":20},{"product":"Tarte","quantity":10}]. Use "persons" when they say how many people to feed, "quantity" when they name a number of portions.'],
        guests: Annotated[int, "Total number of guests for the event. 0 if not stated."] = 0,
        requirements: Annotated[str, "Special requirements in the caller's own words — allergies, vegan, delivery, setup, staff, occasion. Empty if none."] = "",
        email: Annotated[str, "Caller's email if they gave one, to send the quote to. Empty otherwise."] = "",
        customer_name: Annotated[str, "Caller's name if known. Empty otherwise."] = "",
    ) -> str:
        """Build a priced, itemised catering quote during the call and send it to the
        caller by WhatsApp and (if given) email. Use as soon as you know roughly what
        they want and for how many people — do not make them wait for a call-back.
        Read the returned `spoken` summary out to the caller. If the result has
        escalate=true the job is too big to close alone: say a colleague will take
        care of it personally and bring a human in."""
        return await post_tool("catering-quote", {
            "items": items,
            "guests": guests,
            "requirements": requirements,
            "email": email,
            "customer_name": customer_name,
        })

    @loopback_tool
    async def send_catering_link(
        ctx: ToolEnv,
        product: Annotated[str, "Product to deep-link, if the caller asked about a specific one. Empty for the whole shop."] = "",
        note: Annotated[str, "Optional short extra line for the message. Empty if none."] = "",
    ) -> str:
        """WhatsApp the catering webshop link to the caller. Use this instead of reading a
        URL out loud — a web address is unusable spoken. Tell the caller you have sent it."""
        payload = {}
        if product:
            payload["product"] = product
        if note:
            payload["note"] = note
        return await post_tool("send-catering-link", payload)

    @loopback_tool
    async def event_inquiry(
        ctx: ToolEnv,
        guests: Annotated[int, "Approximate number of guests"],
        date: Annotated[str, "Date in YYYY-MM-DD format. Empty if the caller has no date yet."] = "",
        time: Annotated[str, "Start time in HH:mm. Empty if not discussed."] = "",
        occasion: Annotated[str, "What the event is, in the caller's words: Geburtstag, Firmenfeier, Hochzeit … Empty if unknown."] = "",
        format: Annotated[str, "seated (served dinner, everyone sits), standing (reception/buffet, people mingle), mixed, or unknown."] = "unknown",
        outdoor_ok: Annotated[bool, "True if the caller is fine with part of the group sitting outside on the terrace."] = False,
        name: Annotated[str, "Contact name for the event. Empty if not given yet."] = "",
        notes: Annotated[str, "Anything else worth passing on: menu wishes, budget, allergies."] = "",
    ) -> str:
        """Record a private event / privatisation request — renting the whole restaurant, or any
        party too big for a normal table (birthday, company do, wedding). Call this INSTEAD of
        book_reservation for those; a group that size is the whole room, not a table, and
        check_availability cannot answer it. It returns an honest capacity verdict to say out
        loud (what fits seated, with the terrace, or only as a standing/casual party) and alerts
        the team. Call it even when a detail or two is still missing."""
        payload: dict = {"guests": guests, "format": format or "unknown"}
        if date:
            payload["date"] = date
        if time:
            payload["time"] = time
        if occasion:
            payload["occasion"] = occasion
        if outdoor_ok:
            payload["outdoor_ok"] = True
        if name:
            payload["name"] = name
        if notes:
            payload["notes"] = notes
        return await post_tool("event-inquiry", payload)

    @loopback_tool
    async def playbook_lookup(
        ctx: ToolEnv,
        situation: Annotated[str, "The situation name exactly as listed in your briefing"],
    ) -> str:
        """Fetch the restaurant's own rule for a situation. The briefing lists
        which situations have owner rules; when the call turns out to be one of them,
        call this FIRST with that exact name and follow the steps it returns, in order.
        Never improvise a rule you were told exists.
        Returns {found, situation, steps, message}."""
        return await post_tool("playbook-lookup", {"situation": situation})

    @loopback_tool
    async def join_waitlist(
        ctx: ToolEnv,
        date: Annotated[str, "Date in YYYY-MM-DD format"],
        guests: Annotated[int, "Number of guests"],
        name: Annotated[str, "Guest name as heard"],
        time: Annotated[str, "Time in HH:mm 24-hour format. Omit if any time works."] = "",
        note: Annotated[str, "Optional note. Empty if none."] = "",
    ) -> str:
        """Put the caller on the waitlist for a date you could not seat them on.
        Use ONLY when the playbook says to offer the waitlist and the caller agreed.
        Do not ask for a phone number; the server fills caller_phone from the call.
        Returns {success, message, waitlistId}."""
        payload: dict = {"date": date, "guests": guests, "name": name}
        if time:
            payload["time"] = time
        if note:
            payload["note"] = note
        return await post_tool("join-waitlist", payload)

    @loopback_tool
    async def ask_supervisor(
        ctx: ToolEnv,
        question: Annotated[str, "The ONE concrete question, as you would ask a colleague out loud"],
        context: Annotated[str, "One sentence of background: who is on the line and what they want."] = "",
        name: Annotated[str, "Caller's name if you have it. Empty otherwise."] = "",
        target: Annotated[str, "Label of the person to ask (e.g. 'Küche', 'Chef') if the caller asked for someone specific. Empty = whoever is first in line."] = "",
    ) -> str:
        """Ask your manager for something above your pay grade. Guest stays with you.
        Say "Einen Moment bitte, ich frage kurz nach - ich bleibe dran." Never say you are
        calling, texting, or transferring. Then poll check_supervisor every ~2s.
        Do NOT use for availability/menu/hours."""
        payload: dict = {"question": question}
        if context:
            payload["context"] = context
        if name:
            payload["name"] = name
        if target:
            payload["target"] = target
        return await post_tool("ask-supervisor", payload)

    @loopback_tool
    async def check_supervisor(
        ctx: ToolEnv,
        consult_id: Annotated[str, "The consult_id ask_supervisor returned. Empty is fine - the current call is used."] = "",
    ) -> str:
        """Poll the chef consult (~every 2s). Guest stays with you.
        pending: patience ("er schaut gerade…"), no silence, no "I'm texting".
        relay: "Also, er sagt: …". Stay ready.
        answered: "Alles klar, die Rückmeldung ist: …".
        taking_over: "Ich verbinde Sie direkt." then silence.
        unreachable: you could not reach him, he'll get back ASAP. No request_callback.
        NEVER invent. NEVER mention twin/texting/system."""
        payload: dict = {}
        if consult_id:
            payload["consult_id"] = consult_id
        return await post_tool("check-supervisor", payload)

    @loopback_tool
    async def relay_to_supervisor(
        ctx: ToolEnv,
        message: Annotated[str, "What the guest just said that the chef should hear"],
        consult_id: Annotated[str, "consult_id from ask_supervisor. Empty uses the current call."] = "",
    ) -> str:
        """Forward what the guest just said to the chef.
        Tell the guest "Ich gebe das weiter…" then check_supervisor in ~2s."""
        payload: dict = {"message": message}
        if consult_id:
            payload["consult_id"] = consult_id
        return await post_tool("relay-to-supervisor", payload)

    # ── To-go orders ──────────────────────────────────────────────────────
    # Only registered when the tenant has the ordering module (see get_agent).
    # A restaurant that does not sell to-go must never be offered these, and a
    # pizzeria must never see the reservation tools above.

    @loopback_tool
    async def order_menu(ctx: ToolEnv) -> str:
        """The live orderable to-go menu with real prices, sizes and toppings.
        ALWAYS call this before taking an order — never quote a price from memory.
        If it returns available=false, you CANNOT take an order: say so plainly
        and offer to have a colleague call back. Never invent a dish or a price."""
        return await post_tool("order-menu", {})

    @loopback_tool
    async def place_order(
        ctx: ToolEnv,
        name: Annotated[str, "Guest name for the order"],
        items: Annotated[
            str,
            'JSON list of what they want, e.g. [{"name":"Pizza Margherita","quantity":2,'
            '"options":[{"name":"Groß"}]}]. Use the exact dish names from order_menu.',
        ],
        fulfilment_type: Annotated[str, "PICKUP or DELIVERY"] = "PICKUP",
        delivery_address: Annotated[str, "Full address. Required for DELIVERY."] = "",
        scheduled_for: Annotated[str, "ISO datetime if they asked for a specific time"] = "",
        note: Annotated[str, "Any special request in the guest's own words"] = "",
    ) -> str:
        """Place the to-go order. Read the full list AND the total back to the guest
        and get a yes BEFORE calling this. If the result is an error, tell the guest
        what went wrong (sold out, kitchen closed, dish not on the menu) and offer
        the alternatives it suggests — do NOT retry the same order blindly."""
        payload: dict = {
            "name": name,
            "items": _parse_json_list(items),
            "fulfilment_type": (fulfilment_type or "PICKUP").upper(),
        }
        if delivery_address:
            payload["delivery_address"] = delivery_address
        if scheduled_for:
            payload["scheduled_for"] = scheduled_for
        if note:
            payload["note"] = note
        return await post_tool("place-order", payload)

    @loopback_tool
    async def find_my_order(ctx: ToolEnv) -> str:
        """Look up this caller's current open order. Use when they ask about an
        order they already placed ('wo bleibt meine Bestellung?') or want to
        change it. Returns null when they have none."""
        return await post_tool("find-order", {})

    @loopback_tool
    async def cancel_my_order(
        ctx: ToolEnv,
        confirmed: Annotated[bool, "True only after they explicitly said cancel AND a final yes"],
    ) -> str:
        """Cancel this caller's current open order. Only after an explicit
        cancel word and one final confirmation question."""
        if not confirmed:
            return '{"error":"not_confirmed","message":"Ask the guest to confirm the cancellation first."}'
        return await post_tool("cancel-order", {})

    return {
        "reservations": [
            check_availability,
            book_reservation,
            find_reservation_candidates,
            modify_by_id,
            cancel_by_id,
            modify_reservation,
            cancel_reservation,
        ],
        "ordering": [
            order_menu,
            place_order,
            find_my_order,
            cancel_my_order,
        ],
        "shared": [
            request_callback,
            menu_lookup,
            drinks_lookup,
            catering_catalog,
            catering_quote,
            send_catering_link,
            event_inquiry,
            playbook_lookup,
            join_waitlist,
            ask_supervisor,
            check_supervisor,
            relay_to_supervisor,
        ],
    }


def _parse_json_list(raw) -> list:
    """Tolerant JSON list parse - the LLM sometimes sends a list, sometimes a string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _transfer_hold_line(language: str) -> str:
    """Spoken the moment a transfer starts, before the redirect round-trip.

    The tool docstring promises the caller "hears a short 'einen Moment' then
    ringing", but that line lives in the TwiML and only plays AFTER the
    redirect lands — the seconds spent placing it were pure silence, and on
    CA58c1d02… the caller hung up into exactly that gap, one second before
    Twilio answered. Same words as the silent-turn filler on purpose: short
    enough to finish before the redirect cuts our audio, and it leads
    naturally into the TwiML's "…ich verbinde Sie".
    """
    return _silent_turn_filler(language)


_HOLD_ASK_BY_LANG = {
    "de": "Einen Moment bitte, ich frage kurz nach — ich bleibe dran.",
    "en": "One moment, I just need to check — stay with me.",
    "fr": "Un instant, je vérifie vite.",
    "it": "Un attimo, controllo subito.",
    "es": "Un momento, lo consulto ahora.",
    "nl": "Een moment, ik vraag even na.",
    "pl": "Chwileczkę, dopytam.",
    "tr": "Bir dakika, hemen soruyorum.",
    "pt": "Um momento, já confirmo.",
    "hu": "Egy pillanat, gyorsan megkérdezem.",
    "ru": "Секунду, уточню.",
    "ar": "لحظة من فضلك، أسأل الآن.",
}


def _hold_ask_line(language: str) -> str:
    """Guest-facing line while a twin Rückfrage is ringing — never 'I'm texting'."""
    return _lang_line(_HOLD_ASK_BY_LANG, language)


def _strip_spoken_secrets(s: str) -> str:
    """Never speak a phone number or long digit run on the owner leg."""
    return re.sub(r"\s{2,}", " ", re.sub(r"\+?\d[\d\s()/.\-]{6,}\d", " ", s or "")).strip()


def _supervisor_intro(meta: dict) -> str:
    """First words the OWNER hears. Two short beats. Never a phone number."""
    question = _strip_spoken_secrets(str(meta.get("question") or "").strip()) or "eine Frage"
    caller_name = _strip_spoken_secrets(str(meta.get("callerName") or "").strip()) or "Ein Gast"
    intro = f"Rückfrage. {caller_name} fragt: {question}. Was sag ich dem Gast?"
    return intro.rstrip(" \t\r\n.!?…,;:") + "!"


# Only spoken when the call-back POST itself fails — the platform supplies the real
# wording otherwise. Still language-aware: this is the last thing a caller who
# asked for a human ever hears.
_CALLBACK_FALLBACK_BY_LANG = {
    "de": "Ich gebe Ihre Anfrage sofort an einen Kollegen weiter, der sich schnellstmöglich bei Ihnen meldet.",
    "en": "I'm passing your request straight to a colleague, who will get back to you as soon as possible.",
    "fr": "Je transmets votre demande tout de suite à un collègue, qui vous rappellera dès que possible.",
    "it": "Passo subito la sua richiesta a un collega, che la richiamerà al più presto.",
    "es": "Le paso su petición ahora mismo a un compañero, que le devolverá la llamada lo antes posible.",
    "nl": "Ik geef uw verzoek nu door aan een collega, die zo snel mogelijk terugbelt.",
    "pl": "Przekazuję prośbę koledze, który oddzwoni jak najszybciej.",
    "tr": "Talebinizi hemen bir meslektaşıma iletiyorum, en kısa sürede sizi arasın.",
    "pt": "Vou passar o pedido a um colega, que ligará o mais depressa possível.",
    "hu": "Azonnal továbbítom a kérést egy kollégának, aki mielőbb visszahívja.",
    "ru": "Сейчас передам запрос коллеге — он перезвонит как можно скорее.",
    "ar": "سأنقل طلبك الآن إلى زميل وسيتصل بك في أقرب وقت.",
}


def _make_transfer_tools(
    business_id: str,
    caller_phone: str,
    transfer_phone: str,
    call_id: str = "",
    transfer_targets: Optional[list] = None,
    lang_state: Optional[_CallLanguage] = None,
    live_connect_policy: str = "",
):
    """Human hand-off tools: LIVE caller-leg redirect first, call-back bridge as fallback.

    ROOT CAUSE of the transfer saga (TRANSFER_INVESTIGATION.md + IE1 call logs):
    Cartesia's native AgentTransferCall bridges fine when it dials the owner's
    mobile directly, but FAILS INSTANTLY (dur=0) when routed through the shared
    whisper-proxy number — an intra-account dial to a Cartesia-registered number.
    So we no longer ask Cartesia to dial anything: transfer_to_human asks the platform
    to REST-redirect the caller's OWN inbound Twilio leg to a whisper+dial TwiML
    (full control of caller-ID, owner whisper, and no-answer fallback).

    RANKED TARGETS: resolve may return several transfer destinations as
    LABELS + notes (e.g. Restaurant, Inhaber) — numbers stay server-side, so the
    LLM picks a label and the platform maps it to a phone; unknown/empty label = the
    ranked default chain (each target is tried in order until one answers).

    transfer_to_human  → live caller-leg redirect via the platform; the caller stays on
                         the line and rings the ranked targets, Telegram
                         heads-up first.
    callback_bridge    → the platform auto call-back (ends this call, backend dials
                         the caller back from the restaurant number and <Dial>s
                         the owner). Automatic fallback if the redirect can't be
                         placed, or when the caller prefers a call-back."""

    targets = [
        {"label": str(t.get("label") or "").strip(), "note": str(t.get("note") or "").strip()}
        for t in (transfer_targets or [])
        if isinstance(t, dict) and str(t.get("label") or "").strip()
    ]
    if len(targets) > 1:
        target_desc = (
            "Which line to connect to — one of: "
            + "; ".join(f"'{t['label']}'" + (f" ({t['note']})" if t["note"] else "") for t in targets)
            + ". Leave EMPTY for the default (targets are tried in order of importance). "
            "Only set it when the caller asks for a specific person/place that matches a label."
        )
    else:
        target_desc = "Leave empty — there is only one transfer destination."

    async def _live_transfer(reason: str, target: str = "") -> dict:
        """Ask the platform to redirect the caller's OWN Twilio leg to a whisper+dial
        TwiML (the platform owns the bridge). Also fires the Telegram heads-up.

        Returns the JSON body. `ok` + not `use_supervisor` → live connect placed.
        `use_supervisor` → twin-only tenant: the platform started a Rückfrage instead;
        the guest stays with us, do NOT fall back to callback_bridge.
        """
        if not WEBHOOK_SECRET:
            return {}
        payload: dict = {"reason": reason, "call_id": call_id}
        if target:
            payload["target"] = target
        if caller_phone:
            payload["caller_phone"] = caller_phone
        # The TwiML speaks to the caller after our audio is gone, so it needs the
        # language THIS call ended up in — the platform falls back to the tenant's
        # language when we send nothing.
        if lang_state and lang_state.code:
            payload["language"] = lang_state.code
        try:
            resp = await _http().post(
                platform_api(business_id, "tools", "live-transfer"),
                json=payload,
                headers={"X-Cartesia-Secret": WEBHOOK_SECRET},
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.warning("[live-transfer] POST failed", exc_info=True)
            return {}

    async def _request_bridge(reason: str) -> dict:
        """POST to the platform; returns {bridging: bool, message: str}. On any failure
        returns a safe fallback that keeps the caller informed."""
        fallback = {
            "bridging": False,
            "message": _lang_line(
                _CALLBACK_FALLBACK_BY_LANG,
                lang_state.code if lang_state else "",
            ),
        }
        if not WEBHOOK_SECRET:
            return fallback
        payload: dict = {"reason": reason}
        if caller_phone:
            payload["caller_phone"] = caller_phone
        try:
            resp = await _http().post(
                platform_api(business_id, "tools", "bridge-callback"),
                json=payload,
                headers={"X-Cartesia-Secret": WEBHOOK_SECRET},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "bridging": bool(data.get("bridging")),
                "message": str(data.get("message") or fallback["message"]),
            }
        except Exception:
            logger.warning("[bridge-callback] POST failed", exc_info=True)
            return fallback

    @passthrough_tool
    async def transfer_to_human(
        ctx: ToolEnv,
        reason: Annotated[
            str,
            "A SHORT 3-5 word summary, in the caller's language, of why they need "
            "a person (e.g. 'große Gruppe Samstag', 'Beschwerde Rechnung', "
            "'Frage zur Allergie'). The team sees this as an instant heads-up.",
        ],
        target: Annotated[str, target_desc] = "",
    ):
        """Transfer the caller LIVE to a human colleague — the caller STAYS on the
        line and hears a short 'einen Moment' then ringing while the colleague's
        phone rings. Use ONLY after you have tried to help and the caller still
        needs a real person. Say NOTHING before or after calling it and never call
        end_call around a transfer — the hold line and the hand-off are spoken for
        you. If it could not connect, the tool automatically offers an immediate
        call-back. Some restaurants convert this into a twin Rückfrage (guest
        stays) — then poll check_supervisor and never say you transferred."""
        data = await _live_transfer(reason, (target or "").strip())
        if data.get("use_supervisor"):
            # Twin-only: the platform started a Rückfrage instead of through-connecting.
            # Guest stays. Speak the hold-ask line (never "I'm texting") and
            # keep the call — the model must poll check_supervisor next.
            yield AgentSendText(
                text=_hold_ask_line(lang_state.code if lang_state else ""),
                interruptible=False,
            )
            return
        # Cover the redirect round-trip with speech. Uninterruptible so a
        # caller who says "ja bitte" over it cannot cancel the only thing
        # standing between them and several seconds of silence.
        yield AgentSendText(
            text=_transfer_hold_line(lang_state.code if lang_state else ""),
            interruptible=False,
        )
        # Redirect the caller's OWN leg (the platform owns the bridge). This replaces
        # the native AgentTransferCall, which failed instantly when dialed via
        # the whisper-proxy number (see TRANSFER_INVESTIGATION.md / IE1 logs).
        # `target` is a LABEL matched server-side — the LLM never passes numbers.
        if data.get("ok"):
            # The caller's leg has moved to the whisper+dial TwiML and our media
            # stream is ending — say nothing; the redirected TwiML handles it.
            return
        # Redirect could not be placed → graceful fallback: immediate call-back.
        result = await _request_bridge(reason)
        yield AgentSendText(text=result["message"], interruptible=False)
        if result["bridging"]:
            yield AgentEndCall(reason="agent_ended", interruptible=False)

    @passthrough_tool
    async def callback_bridge(
        ctx: ToolEnv,
        reason: Annotated[
            str,
            "A SHORT 3-5 word summary, in the caller's language, of why they need "
            "a person. The team sees this as an instant heads-up.",
        ],
    ):
        """FALLBACK human hand-off: an IMMEDIATE automatic call-back that bridges
        the caller to a colleague. Use ONLY if a live transfer (transfer_to_human)
        did not connect, or the caller explicitly prefers being called back. This
        does NOT keep the current call open — it speaks the right closing line and
        ends the call FOR you, so do NOT say anything else and do NOT call
        end_call yourself; the call-back is already on its way."""
        result = await _request_bridge(reason)
        yield AgentSendText(text=result["message"], interruptible=False)
        if result["bridging"]:
            # Free the caller's line so the backend can ring them straight back
            # and bridge the colleague. If not bridging (withheld number), stay on
            # the line so we can still gather a call-back number.
            yield AgentEndCall(reason="agent_ended", interruptible=False)

    # Twin-only restaurants: do not offer live through-connect OR the
    # call-back bridge (that also dumps the guest onto the owner).
    # ask_supervisor is the human path; digit 1 on the Rückfrage takes over.
    if str(live_connect_policy or "").lower() == "never":
        return []
    return [transfer_to_human, callback_bridge]


def _make_switch_voice_tool(voices: dict, lang_state: Optional[_CallLanguage] = None):
    """Build a passthrough tool the LLM can call to switch TTS voice + STT
    language when the caller changes language. Yields AgentUpdateCall so the
    harness picks up the new voice_id / language for the rest of the call, and
    writes the new language into the shared _CallLanguage so everything the
    wrapper speaks on its own (idle re-prompt, silent-turn filler, goodbye)
    follows the caller too."""

    voice_map = {str(k).lower(): str(v) for k, v in voices.items() if v}

    @passthrough_tool
    async def switch_voice_to_language(
        ctx: ToolEnv,
        language: Annotated[
            str,
            "Two-letter language code (ISO 639-1) the caller is now speaking in: "
            "'de' (German), 'fr' (French), 'en' (English), 'it' (Italian), etc.",
        ],
    ):
        """Switch the TTS voice + STT language for the rest of the call. Call this
        immediately when you detect the caller has switched languages. Never
        announce the switch. The tool returns a confirmation — after it you MUST
        speak your normal reply to the caller in their language; a response that
        is only this tool call leaves dead air on the phone."""
        code = (language or "").strip().lower()[:2]
        voice_id = voice_map.get(code)
        if not voice_id:
            logger.info(f"[cartesia] switch_voice → no clone for '{code}', STT/TTS language only")
            if lang_state is not None and code:
                lang_state.code = code
            if code:
                yield AgentUpdateCall(language=code)
            yield (
                f"No dedicated clone for '{code}' — recognizer switched. "
                "Now speak your reply to the caller in their language."
            )
            return
        logger.info(f"[cartesia] switch_voice → {code} (voice_id={voice_id})")
        if lang_state is not None:
            lang_state.code = code
            lang_state.voice_id = voice_id
        yield AgentUpdateCall(voice_id=voice_id, language=code)
        # Same loopback guarantee as above; also means the reply is generated
        # AFTER the switch, so it already plays in the new voice/language.
        yield (
            f"Voice switched to '{code}'. Now speak your reply to the caller "
            "in their language."
        )

    return switch_voice_to_language


# Short "line unavailable" message per tenant language, for billing-suspended
# tenants. Must end in terminal punctuation or the TTS endpointer never flushes
# the native intro (same rule get_agent enforces for greetings). German is the
# primary market → default.
_UNAVAILABLE_BY_LANG = {
    "de": "Dieser Anschluss ist zurzeit leider nicht verfügbar. Bitte versuchen Sie es später erneut.",
    "en": "This number is currently unavailable. Please try again later.",
    "fr": "Ce numéro est actuellement indisponible. Veuillez réessayer plus tard.",
    "it": "Questo numero al momento non è disponibile. La preghiamo di riprovare più tardi.",
    "es": "Este número no está disponible en este momento. Inténtelo más tarde.",
    "nl": "Dit nummer is momenteel niet bereikbaar. Probeer het later opnieuw.",
    "pl": "Ten numer jest obecnie niedostępny. Prosimy spróbować później.",
    "tr": "Bu hat şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.",
    "pt": "Este número está indisponível de momento. Tente mais tarde.",
    "hu": "Ez a szám jelenleg nem elérhető. Kérjük, próbálja később.",
    "ru": "Этот номер сейчас недоступен. Позвоните позже.",
    "ar": "هذا الرقم غير متاح حاليا. يرجى المحاولة لاحقا.",
}
_UNAVAILABLE_DEFAULT = _UNAVAILABLE_BY_LANG["en"]


class _DeclineAgent:
    """Play a short 'currently unavailable' line, then hang up.

    Used to decline calls for billing-suspended tenants (canceled subscription
    or expired trial) so they never get a free AI conversation — the Cartesia
    equivalent of the blocked-number reject. The message is spoken via the
    platform's native CallStarted introduction (the only proactive-greeting
    path proven to play audibly here; see _TenantVoiceAgent), then the call is
    ended. A barge-in fallback guarantees the call still ends even if the caller
    talks over the intro and the opening turn is cancelled. No LLM turn ever
    runs (we end on CallStarted), so the static intro costs no tokens.
    """

    def __init__(self, inner: LlmAgent, message: str):
        self.inner = inner
        self.message = (message or "").strip()
        self._ended = False

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    async def process(self, env, event):
        if isinstance(event, CallEnded) or self._ended:
            return
        self._ended = True
        if isinstance(event, CallStarted):
            # Let the native intro (the unavailable line) flow through, then
            # hang up once it has been queued for playback.
            async for out in self.inner.process(env, event):
                yield out
            yield AgentEndCall(reason="agent_ended", interruptible=False)
            return
        # Caller spoke before/over the intro: say the line once (mid-turn text
        # plays reliably) and end.
        if self.message:
            yield AgentSendText(text=self.message, interruptible=False)
        yield AgentEndCall(reason="agent_ended", interruptible=False)



async def _post_supervisor(
    business_id: str,
    consult_id: str,
    slug: str,
    payload: dict,
    call_id: str = "",
) -> str:
    """One consult call to the platform. Raw JSON text back, never an exception."""
    body = dict(payload or {})
    body["consult_id"] = consult_id
    if call_id:
        body["call_id"] = call_id
    url = platform_api(business_id, "tools", slug)
    try:
        r = await _http().post(
            url,
            headers={"X-Cartesia-Secret": WEBHOOK_SECRET, "Content-Type": "application/json"},
            json=body,
            timeout=HTTP_TIMEOUT,
        )
        text = r.text
        if r.status_code >= 400:
            logger.warning(f"[supervisor-tool] {slug} → {r.status_code} {text[:200]}")
            return json.dumps({"ok": False, "message": f"tool error {r.status_code}"})
        return text
    except Exception as e:
        logger.exception(f"[supervisor-tool] {slug} failed: {e}")
        return json.dumps({"ok": False, "message": str(e)})


def _make_supervisor_tools(business_id: str, consult_id: str, call_id: str = ""):
    """Tools for the OUTBOUND leg - live middleman twin talking to the chef."""

    async def post_sup(slug: str, payload: dict) -> str:
        return await _post_supervisor(business_id, consult_id, slug, payload, call_id)

    @loopback_tool
    async def tell_guest(
        ctx: ToolEnv,
        message: Annotated[str, "What the guest should hear right now - clear decision or reply"],
        final: Annotated[bool, "True only when the middleman is done and the chef can hang up"] = False,
    ) -> str:
        """Relay to the guest twin NOW. Stay on the line - you are the middleman.
        Use final=true only when conversation with guest via you is finished."""
        return await post_sup("supervisor-answer", {"answer": message, "final": bool(final)})

    @loopback_tool
    async def check_guest_reply(
        ctx: ToolEnv,
    ) -> str:
        """Poll for what the guest said next (~every 2s while waiting)."""
        return await post_sup("supervisor-check-guest", {})

    @loopback_tool
    async def take_over_guest_call(
        ctx: ToolEnv,
        reason: Annotated[str, "Why the chef wants the live guest"] = "",
    ) -> str:
        """Connect the chef to the guest live. Tell chef to hang up - phone rings with guest."""
        return await post_sup("supervisor-takeover", {"reason": reason})

    return [tell_guest, check_guest_reply, take_over_guest_call, end_call]


_TAKEOVER_KEY = "1"
_TAKEOVER_SPOKEN = "Ich verbinde. Dein Telefon klingelt gleich mit dem Gast."
_TAKEOVER_FAILED_SPOKEN = "Das Verbinden klappt gerade nicht - sag mir einfach, was ich dem Gast ausrichten soll."


class _SupervisorKeypadAgent:
    """The chef's keypad on the Rückfrage leg. Owns the take-over key.

    the platform tells the chef "Taste 1 übernimmt" (Telegram, and the TwiML brief
    says it out loud), and on the TwiML leg the <Gather> really does read the
    digit. On the TWIN leg it was a dead key: the SDK delivers a keypress as
    its own UserDtmfSent event, which is neither in the default run filter
    (CallStarted / UserTurnEnded / CallEnded) nor rendered into the prompt by
    LlmAgent._build_messages - only UserTextSent becomes a user message. So
    the model was told to watch for "digit 1" and could never see one. The
    chef pressed, nothing happened, and the guest kept waiting.

    We take the key here instead of teaching the model to: a keypress is a
    decision, not something to reason about, and routing it through STT and an
    LLM turn is latency and luck we don't need. Everything else falls through
    to the dialog agent untouched.
    """

    def __init__(self, inner, business_id: str, consult_id: str, call_id: str = ""):
        self.inner = inner
        self.business_id = (business_id or "").strip()
        self.consult_id = (consult_id or "").strip()
        self.call_id = (call_id or "").strip()
        self._taking_over = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def process(self, env, event):
        if not isinstance(event, UserDtmfSent):
            async for out in self.inner.process(env, event):
                yield out
            return

        button = str(getattr(event, "button", "") or "").strip()
        if button != _TAKEOVER_KEY:
            # Nothing else is bound - stay silent rather than derail the
            # Rückfrage over a mis-hit key.
            logger.info(f"[cartesia] supervisor keypad ignored '{button}' consult={self.consult_id}")
            return
        # A held key repeats; the take-over is placed exactly once.
        if self._taking_over:
            return
        self._taking_over = True

        logger.info(f"[cartesia] supervisor pressed {button} → take-over consult={self.consult_id}")
        raw = await _post_supervisor(
            self.business_id,
            self.consult_id,
            "supervisor-takeover",
            {"reason": "Taste 1 auf der Rückfrage"},
            self.call_id,
        )
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            data = None
        # Only an explicit refusal counts - an unparseable body means the platform
        # answered, and the transfer is more likely placed than not.
        ok = data.get("ok") is not False if isinstance(data, dict) else True
        if not ok:
            # The guest is still on their own call and the chef is still on
            # this one - let them just say the answer instead of hanging up
            # into a transfer that was never placed.
            self._taking_over = False
            logger.warning(f"[cartesia] take-over refused consult={self.consult_id}: {raw[:200]}")
            yield _speak(_TAKEOVER_FAILED_SPOKEN)
            return

        # End this leg ourselves. the platform is dialing the chef's own number for
        # the guest, and that number is busy until this call is down.
        yield _speak(_TAKEOVER_SPOKEN)
        yield AgentEndCall(reason="agent_ended", interruptible=False)


async def _build_supervisor_agent(meta: dict, call_request: CallRequest):
    """Outbound call TO the restaurant boss - live concierge dialog, not IVR."""
    business_id = str(meta.get("businessId") or "").strip()
    consult_id = str(meta.get("consultId") or "").strip()
    question = str(meta.get("question") or "").strip()
    context = str(meta.get("context") or "").strip()
    caller_name = str(meta.get("callerName") or "").strip() or "ein Gast"
    caller_phone = str(meta.get("callerPhone") or "").strip()
    brief = str(meta.get("briefText") or "").strip()

    intro = _supervisor_intro(meta)

    who = _strip_spoken_secrets(caller_name)
    system_prompt = (
        "You are calling the restaurant SUPERVISOR as a short Rückfrage. "
        "German. Punchy. Two sentences max per turn. NEVER read a phone number out loud "
        "(you may know it; do not speak the digits).\n"
        f"Guest (NOT on this line): {who}\n"
        f"Question: {question}\n"
        f"Context: {context or '-'}\n"
        "The introduction already said this is a Rückfrage. Do NOT repeat the disclaimer. "
        "Listen, then relay.\n"
        "Loop:\n"
        "1) After they answer, call tell_guest (stay on the line).\n"
        "2) Poll check_guest_reply every ~2s; relay with tell_guest.\n"
        "3) tell_guest(final=true) + end_call when done.\n"
        "4) verbinde mich / connect me / ich nehm das Gespräch → take_over_guest_call, "
        "tell them to hang up, end_call. (The keypad 1 is handled without you - "
        "never ask them to press it again, and never claim they did.)\n"
        "Never invent policy. Never book tables on this leg."
    )
    if brief:
        system_prompt += f"\nSpoken brief (already played): {brief}"

    tools = _make_supervisor_tools(business_id or "unknown", consult_id, call_request.call_id or "")
    model = os.getenv("CARTESIA_LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
    agent = LlmAgent(
        model=model,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=tools,
        config=LlmConfig(system_prompt=system_prompt, introduction=intro),
        max_tool_iterations=8,
    )
    de_voice = str(meta.get("voiceId") or "").strip()
    if de_voice:
        agent = _TenantVoiceAgent(
            agent,
            voice_id=de_voice,
            language="de",
            greeting="",
            business_id=business_id,
            call_id=call_request.call_id or "",
            voices={"de": de_voice},
            languages=["de"],
            leg="supervisor",
        )
    # Outermost, so the take-over key is read before the voice wrapper can
    # hand a keypress to the LLM as an empty turn.
    agent = _SupervisorKeypadAgent(
        agent,
        business_id=business_id,
        consult_id=consult_id,
        call_id=call_request.call_id or "",
    )
    logger.info(
        f"[cartesia] SUPERVISOR leg consult={consult_id} biz={business_id} "
        f"q={question[:60]!r} call={call_request.call_id} voice={de_voice or 'default'}"
    )

    # UserDtmfSent is NOT in the SDK's default run filter, so without this the
    # wrapper above never runs and the key stays dead.
    def _supervisor_run(ev):
        return isinstance(ev, (CallStarted, UserTurnEnded, CallEnded, UserDtmfSent))

    def _supervisor_cancel(ev):
        return isinstance(ev, UserTurnStarted)

    return (agent, _supervisor_run, _supervisor_cancel)


async def get_agent(env: AgentEnv, call_request: CallRequest):
    to_number = call_request.to or ""
    from_number = call_request.from_ or ""
    logger.info(f"[cartesia] new call {call_request.call_id} to={to_number} from={from_number}")

    meta = call_request.metadata or {}
    # OUTBOUND supervisor consult: we rang the chef while the guest holds.
    # Metadata was set by the platform when placing the Cartesia outbound call.
    if str(meta.get("mode") or "") == "supervisor_consult" and meta.get("consultId"):
        return await _build_supervisor_agent(meta, call_request)

    # BOOT LATENCY: pre_call already resolved this tenant and shipped the full
    # payload back to us via call metadata (the platform echoes PreCallResult
    # metadata on the WebSocket start message). Re-resolving here put a second
    # the platform round-trip between the caller and the greeting — the caller was
    # talking into dead air meanwhile. Fall back to a live resolve only if the
    # metadata is missing (older harness, direct WS connects).
    resolved: Optional[dict] = None
    cached = meta.get("resolved")
    if isinstance(cached, dict) and cached.get("businessId"):
        resolved = cached
        logger.info(f"[cartesia] resolved {to_number} from pre_call metadata (no second round-trip)")
    else:
        resolved = await _resolve_business(to_number, from_number)

    # Billing gate — resolve returns `suspended: true` when the tenant's
    # subscription is canceled or its trial expired (same checkCallLimit() the
    # phone-ai-profile endpoint uses). Decline the call gracefully instead of
    # running the assistant, so a suspended tenant gets no free AI calls. This
    # is the Cartesia path's equivalent of the blocked-number reject; the Grok
    # path is gated server-side. Older platform deploys omit the field → absent
    # → falsy → no gate, so this is a safe no-op until the platform ships it.
    if resolved and resolved.get("suspended"):
        business_id = resolved.get("businessId", "")
        logger.info(
            f"[cartesia] tenant {business_id or to_number} billing-suspended → "
            f"declining call {call_request.call_id}"
        )
        lang = (resolved.get("language") or "").strip().lower()[:2]
        message = _UNAVAILABLE_BY_LANG.get(lang, _UNAVAILABLE_DEFAULT)
        decline_inner = LlmAgent(
            model=os.getenv("CARTESIA_LLM_MODEL", "anthropic/claude-haiku-4-5-20251001"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            tools=[end_call],
            config=LlmConfig(system_prompt=FALLBACK_SYSTEM_PROMPT, introduction=message),
            max_tool_iterations=1,
        )
        return _DeclineAgent(decline_inner, message)

    if resolved:
        business_id = resolved["businessId"]
        system_prompt = resolved.get("systemPrompt") or FALLBACK_SYSTEM_PROMPT
        introduction = resolved.get("introduction") or FALLBACK_INTRODUCTION
        transfer_phone = (resolved.get("transferPhone") or "").strip()
        # Per-language voice map (one clone/catalogue voice per language).
        # Mid-call switches are applied by _TenantVoiceAgent before the LLM
        # turn (zero tool latency). Keep a light prompt note so the model
        # replies in the caller's language without calling a switch tool.
        voices = resolved.get("voices") or {}
        if not isinstance(voices, dict):
            voices = {}

        # ONE language state for the whole call: the wrapper writes to it when
        # it detects a switch pre-turn, the tool writes to it when the model
        # asks for one, and both read it back.
        lang_state = _CallLanguage(
            (resolved.get("language") or ""), (resolved.get("voiceId") or "")
        )

        raw_langs = resolved.get("supportedLanguages") or []
        supported_langs = sorted(
            {
                str(x).lower()[:2]
                for x in (raw_langs if isinstance(raw_langs, list) else [])
                if x and str(x).strip()
            }
            | {str(k).lower()[:2] for k in voices.keys() if k}
        )
        if supported_langs:
            clones = ", ".join(sorted(str(k).lower()[:2] for k in voices.keys() if k)) or "none"
            offered = ", ".join(supported_langs)
            system_prompt += (
                "\n\nLANGUAGE / VOICE: Reply in the language the caller is actually speaking "
                "(every word, including numbers/dates). Do NOT use their phone country code. "
                f"This restaurant answers in: {offered}. Dedicated clones: {clones}. "
                "The voice and recognizer follow automatically; if still wrong after your "
                "reply, call switch_voice_to_language once with their 2-letter code. If they "
                "only said hallo/allô/bonjour/hi, speak NOTHING. Never announce a switch.\n"
            )

        # Tool set follows what the tenant actually sells. `features` comes from
        # the platform's /resolve; a pizzeria gets ordering and NO reservation tools,
        # so the agent cannot steer a pizza customer toward booking a table.
        # Missing/!dict features (older API) => reservations only, which is the
        # behaviour every tenant had before ordering existed.
        raw_features = resolved.get("features")
        features = raw_features if isinstance(raw_features, dict) else {}
        allow_reservations = features.get("reservations", True) is not False
        allow_ordering = features.get("ordering") is True

        tool_groups = _make_tools(business_id, from_number, call_request.call_id or "")
        tools = list(tool_groups["shared"])
        if allow_reservations:
            tools.extend(tool_groups["reservations"])
        if allow_ordering:
            tools.extend(tool_groups["ordering"])
        # Tenant knowledge base (Cartesia folders + metadata.businessId filter).
        # Facts (address, gift cards, policies, owner notes) are retrieved on
        # demand so the system prompt stays under the voice token budget.
        # is_background=True: agent can keep speaking while retrieval runs.
        # @see https://docs.cartesia.ai/line/knowledge-base
        tools.append(
            knowledge_base(
                filters={"businessId": business_id},
                top_k=3,
                timeout_s=2.0,
                is_background=True,
                description=(
                    "Look up restaurant facts: address, parking, dogs, outdoor seating, "
                    "accessibility, gift cards/Gutscheine, about the restaurant, owner notes, "
                    "policies, hiring. Use when the caller asks a fact question. "
                    "Do NOT use for booking availability, open/closed, or placing reservations — "
                    "those require the booking tools."
                ),
            )
        )
        # Voice switching: always registered. With the auto path on it is a
        # backstop for what detection missed (and usually never called, so it
        # costs nothing); with it off it is the only mechanism. Either way its
        # loopback yield guarantees the turn still produces speech.
        if voices or supported_langs:
            tools.append(_make_switch_voice_tool(voices, lang_state))
        # Voicemail detection (end the call gracefully if a machine answers).
        tools.append(voicemail)
        # DTMF (for "press 1 to confirm" style IVR interactions if anyone routes through one).
        tools.append(send_dtmf)
        # Human hand-off. Custom tools: fire an instant heads-up to the team
        # (reason + caller number), then LIVE-transfer by redirecting the caller's
        # own leg via the platform (see _make_transfer_tools), with the auto call-back
        # bridge as explicit fallback. The call_id carries the caller's Twilio
        # CallSid so the platform can redirect the right leg; transferTargets carries
        # the ranked destination LABELS the LLM may pick from (numbers stay
        # server-side).
        if transfer_phone:
            tools.extend(
                _make_transfer_tools(
                    business_id,
                    from_number,
                    transfer_phone,
                    call_request.call_id or "",
                    resolved.get("transferTargets") or [],
                    lang_state,
                    str(resolved.get("liveConnectPolicy") or ""),
                )
            )
        else:
            # Without a pinned number we skip the tools entirely so the agent
            # doesn't try to transfer to nowhere.
            pass
        tools.append(end_call(description=END_CALL_DESCRIPTION, interruptible=False))
        logger.info(
            f"[cartesia] resolved {to_number} → business {business_id} "
            f"transfer={transfer_phone or 'none'} tools={len(tools)} "
            f"mode={features.get('serviceModel', 'reservations')}"
        )
    else:
        system_prompt = FALLBACK_SYSTEM_PROMPT
        introduction = FALLBACK_INTRODUCTION
        tools = [end_call(description=END_CALL_DESCRIPTION, interruptible=False)]
        lang_state = _CallLanguage()
        logger.warning(f"[cartesia] no business for {to_number}; running in fallback mode")

    # Build config directly so our per-tenant, tz-aware, live-date prompt
    # always wins — `from_call_request` would prefer the harness-supplied
    # `agent.system_prompt` (the Cartesia agent's STORED text, which is frozen
    # at edit time and would re-hardcode the date and re-leak tenant names).
    #
    # GREETING — per-tenant, resolved greeting FIRST. Every tenant must hear
    # its own restaurant's greeting, never the shared agent's stored
    # `llm_introduce` (which is a tenant's text). Current Line docs state the
    # introduction the agent emits on CallStarted is played proactively; the
    # old June observation ("must match the stored llm_introduce or it's held
    # until inbound audio") predates harness 2026-04-03. For a tenant the
    # resolved greeting equals the stored one, so this is a no-op there —
    # verify proactive playback on a NEW tenant's number after deploying
    # (greeting-monitor C6 + a test call to Platform 68).
    # Fallback chain: tenant greeting → agent's stored greeting.
    harness_intro = (call_request.agent.introduction or "").strip()
    proactive_intro = introduction or harness_intro
    # CRITICAL for the proactive CallStarted greeting: the intro text MUST end in
    # "!" or the TTS endpointer never flushes it and the platform drops it
    # silently (root cause of the "a tenant bonjour" dead-greeting — "Platform
    # 68, guten Tag!" played only because of its trailing "!"). Force a single
    # trailing "!" as a backstop, in case a greeting reaches here unpunctuated.
    _stripped = proactive_intro.rstrip(" \t\r\n.!?…,;:·—–-")
    if _stripped:
        proactive_intro = f"{_stripped}!"
    # A3: model is env-overridable so we can A/B Sonnet vs Haiku on the live
    # main loop without a code change. Defaults to Haiku (current prod model)
    # if unset, so this is a no-op until CARTESIA_LLM_MODEL is explicitly set.
    main_loop_model = os.getenv("CARTESIA_LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
    logger.info(f"[cartesia] main-loop model={main_loop_model}")
    agent = LlmAgent(
        model=main_loop_model,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        tools=tools,
        config=LlmConfig(
            # Native intro path — the ONLY mechanism that audibly plays a
            # proactive greeting (Jul 2 2026). Requires that no voice override
            # is in flight at call start; see _TenantVoiceAgent.
            system_prompt=system_prompt,
            introduction=proactive_intro,
            extra={
                "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"},
            },
        ),
        max_tool_iterations=10,
    )
    tenant_voice_id = (resolved.get("voiceId") or "").strip() if resolved else ""
    tenant_language = (resolved.get("language") or "").strip().lower()[:2] if resolved else ""
    tenant_voices = resolved.get("voices") if resolved else None
    if not isinstance(tenant_voices, dict):
        tenant_voices = {}
    # ALWAYS wrap — the greeting re-arm must protect every tenant (P68 has no
    # voice clone but still needs its greeting guaranteed); the voice switch
    # inside the wrapper is a no-op when voice_id is empty.
    #
    # UserTurnStarted is in the run filter (not only cancel) so barge-in
    # starts a lost-EOT watchdog instead of leaving the line silent when
    # UserTurnEnded never arrives.
    def _tenant_run(ev):
        return isinstance(ev, (CallStarted, UserTurnEnded, CallEnded, UserTurnStarted))

    def _tenant_cancel(ev):
        return isinstance(ev, UserTurnStarted)

    return (
        _TenantVoiceAgent(
            agent,
            tenant_voice_id,
            tenant_language,
            greeting=proactive_intro,
            business_id=business_id if resolved else "",
            call_id=call_request.call_id or "",
            voices=tenant_voices,
            lang_state=lang_state,
            lang_hint="",
            languages=resolved.get("supportedLanguages") if resolved else [],
        ),
        _tenant_run,
        _tenant_cancel,
    )


async def pre_call_handler(call_request: CallRequest):
    """Fired before the WebSocket connects. Deliberately returns NO tts/stt
    config: ANY voice override in flight at call start silences the proactive
    greeting entirely (Jul 2 2026 — four independent mechanisms tested, all
    silent; June's override-free native intro is the only path that plays).
    The tenant voice is applied on the caller's first utterance instead
    (_TenantVoiceAgent).

    The FULL resolve payload rides along in metadata so get_agent can skip
    its own resolve round-trip — the platform echoes this metadata back on
    the WebSocket start message (voice_agent_app.create_chat_session)."""
    incoming = call_request.metadata or {}
    # Outbound supervisor leg: keep the platform's consult metadata intact.
    if str(incoming.get("mode") or "") == "supervisor_consult" and incoming.get("consultId"):
        return PreCallResult(metadata=dict(incoming), config={})

    resolved = await _resolve_business(call_request.to or "", call_request.from_ or "")
    if not resolved:
        return PreCallResult(metadata={}, config={})
    return PreCallResult(
        metadata={"businessId": resolved.get("businessId", ""), "resolved": resolved},
        config={},
    )


app = VoiceAgentApp(get_agent=get_agent, pre_call_handler=pre_call_handler)

if __name__ == "__main__":
    print("Starting multi-tenant Cartesia voice agent")
    app.run()
