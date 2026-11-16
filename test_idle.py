#!/usr/bin/env python3
"""No-SDK tests for idle / end_call gating helpers in main.py."""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys

# Load main.py without executing VoiceAgentApp.run. The module constructs the
# app at import; that's fine — we only need the helpers.
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("cartesia_main", ROOT / "main.py")
m = importlib.util.module_from_spec(spec)
# Importing main talks to nothing if we don't call run().
spec.loader.exec_module(m)


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(name + (": " + detail if detail else ""))


def drain(agen) -> list:
    """Everything an agent yields for one event."""

    async def _run():
        return [out async for out in agen]

    return asyncio.run(_run())


class _RecordingAgent:
    """Stand-in for the Rückfrage dialog agent: records what it was handed."""

    def __init__(self):
        self.seen = []

    async def process(self, env, event):
        self.seen.append(event)
        yield m.AgentSendText(text="inner ran", interruptible=False)


def main() -> int:
    failed = 0

    def run(name, fn):
        nonlocal failed
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  ({e})")

    def farewells():
        for t in (
            "Auf Wiederhören!",
            "Tschüss, bis dann",
            "Okay, bye!",
            "Au revoir et à bientôt",
            "Goodbye, see you Saturday",
            "Ciao!",
        ):
            check(t, m._is_farewell(t), t)

    def not_farewells():
        for t in (
            "Ja",
            "Okay, vier Personen",
            "Danke",
            "Wie ist dein Name?",
            "Einen Moment bitte.",
        ):
            check(t, not m._is_farewell(t), t)

    def bare_acks():
        for t in ("Ja", "ja.", "Okay", "Danke!", "merci", "mhm", "genau", "yes"):
            check(t, m._is_bare_ack(t), t)

    def not_bare_acks():
        for t in ("Ja, auf Wiederhören", "Agil", "vier Personen", "Ja, Samstag um sieben"):
            check(t, not m._is_bare_ack(t), t)

    def block_ja_mid_call():
        check("ja after name ask", m._should_block_end_call("Ja", "Wie ist dein Name, bitte?"))
        check("okay after question", m._should_block_end_call("Okay", "Für welchen Tag denn?"))
        check("danke after confirm", m._should_block_end_call("Danke", "Ihr Tisch ist reserviert."))

    def allow_real_goodbye():
        check("caller tschüss", not m._should_block_end_call("Tschüss", "Ihr Tisch ist reserviert."))
        check(
            "ja after our goodbye",
            not m._should_block_end_call("Ja", "Perfekt, auf Wiederhören!"),
        )
        check("bye", not m._should_block_end_call("Bye!", "See you Saturday"))

    # ── language detection / state (call CA58c1d02…, 2026-08-14) ────────────
    VOICES = {"de", "en", "fr", "it", "es", "nl", "pl", "tr"}

    def detects_each_language_mid_call():
        cases = (
            ("Vous êtes quoi?", "fr"),
            ("Oui ou non.", "fr"),
            ("Ich möchte einen Tisch für vier Personen", "de"),
            ("I'd like a table for four people tomorrow night", "en"),
            ("Vorrei prenotare un tavolo per quattro persone", "it"),
            ("Quiero una mesa para cuatro personas mañana", "es"),
            ("Ik wil graag een tafel voor vier personen reserveren", "nl"),
            ("Chciałbym zarezerwować stolik na jutro na cztery osoby", "pl"),
            ("Yarın dört kişilik masa istiyorum lütfen", "tr"),
        )
        for t, lang in cases:
            other = "en" if lang == "de" else "de"
            check(t, m._detect_caller_language(t, VOICES, other) == lang, repr(t))
        check(
            "unoffered language stays put",
            m._detect_caller_language("Quiero una mesa para cuatro personas", {"de", "fr"}, "de") is None,
        )

    def ignores_ambiguous_openers():
        # "Allô" and "Hallo" are one sound to a de-configured STT, and "Sure"
        # is what it actually transcribed. None may move the voice without a
        # country hint. "Hi" is international and must not flip to English.
        for t in ("Hallo?", "Allô.", "Sure", "Ja", "Hallo, hallo?", "Hi.", "Hey"):
            check(t, m._detect_caller_language(t, VOICES, "de") is None, repr(t))

    def unique_openers_switch_voice():
        # These are not the same sound as hallo; they MUST move the clone
        # so a French caller is not answered in the German accent.
        check("bonjour", m._detect_caller_language("Bonjour.", VOICES, "de") == "fr")
        check("bonsoir", m._detect_caller_language("Bonsoir !", VOICES, "de") == "fr")
        check("guten abend", m._detect_caller_language("Guten Abend", VOICES, "fr") == "de")
        check("guten tag", m._detect_caller_language("Guten Tag!", VOICES, "fr") == "de")
        check("buenos dias", m._detect_caller_language("Buenos días", VOICES, "de") == "es")
        check("buongiorno", m._detect_caller_language("Buongiorno", VOICES, "de") == "it")
        check("merhaba", m._detect_caller_language("Merhaba", VOICES, "de") == "tr")
        check("dzien dobry", m._detect_caller_language("Dzień dobry", VOICES, "de") == "pl")
        check("goedendag", m._detect_caller_language("Goedendag", VOICES, "de") == "nl")

    def mixed_french_keeps_french_clone():
        # 2026-08-22 François: French sentence + "guten Tag" + ü in Müller
        # used to steal the German clone; the model then spoke French with
        # a German accent.
        stay = "Je suis François Müller, guten Tag."
        check(
            "mixed je suis + guten Tag stays fr",
            m._detect_caller_language(stay, VOICES, "fr") is None,
            stay,
        )
        check(
            "umlaut in a name is not a switch",
            m._detect_caller_language("Müller", VOICES, "fr") is None,
        )
        check(
            "clear German after French greeting still switches",
            m._detect_caller_language(
                "Ich hab eine Frage. Haben Sie heute und morgen geschlossen?",
                VOICES,
                "fr",
            )
            == "de",
        )
        check(
            "spoken French is guessed fr",
            m._guess_spoken_language(
                "Oui, aujourd'hui et demain nous sommes fermés.", VOICES
            )
            == "fr",
        )
        check(
            "spoken German is guessed de",
            m._guess_spoken_language("Einen Moment, ich schaue das für Sie nach.", VOICES)
            == "de",
        )

    def short_lines_never_inherit_the_previous_clone():
        # Replays the agent side of CA5706468833eb526508717a2fe520c762. Matching
        # the clone per sentence is not enough: "Sind Sie noch da?" and "Gerne!"
        # are too short to classify, and used to keep the FR clone the previous
        # French sentence switched to — German words, French accent.
        conversation = "de"
        clone = "de"
        stuck = []
        for line in (
            "Ah, François!",
            "Einen Moment, ich schaue das für Sie nach.",
            "Oui, aujourd'hui et demain nous sommes fermés pour congés.",
            "Sind Sie noch da?",
            "Gerne!",
            "Donnerstag, verstanden — das ist der 27.",
            "August.",
            "Gerne, François — bis bald!",
        ):
            guessed = m._guess_spoken_language(line, VOICES) or conversation
            if guessed in VOICES:
                clone = guessed
            french_words = "oui" in line.lower() or "aujourd" in line.lower()
            if clone == "fr" and not french_words:
                stuck.append(line)
        check(
            "no German line is left on the French clone",
            not stuck,
            f"stranded on fr: {stuck}",
        )

    def a_language_without_a_clone_still_switches():
        """A tenant answers ES/TR/PL but has clones only for DE/FR/EN.

        Gating the switch on having a clone meant those languages got no
        update at all, so Spanish went out through the German clone in German
        — the accent bug, just for the languages nobody test-calls.
        """
        agent = m._TenantVoiceAgent(
            _RecordingAgent(), voice_id="v-de", language="de",
            voices={"de": "v-de", "fr": "v-fr", "en": "v-en"},
            languages=["es", "tr", "pl"],
        )
        check("clone-less langs are candidates", {"es", "tr", "pl"} <= agent.candidates)

        upd = agent._pin_clone("es")
        check("switching to a clone-less language still emits", upd is not None)
        check("it is a language-only update", getattr(upd, "language", "") == "es", repr(upd))
        check("the clone is left alone", agent.voice_id == "v-de", agent.voice_id)
        check("and we remember TTS is on es", agent._spoken_lang == "es")
        check("asking again is a no-op", agent._pin_clone("es") is None)

        back = agent._pin_clone("de")
        check("back to a real clone switches the voice", getattr(back, "voice_id", "") == "v-de")
        check("and the language with it", getattr(back, "language", "") == "de")

        # The guessed-language path must reach the same place.
        es = m._guess_spoken_language("Sí, estamos abiertos el jueves.", agent.candidates)
        check("spoken Spanish is guessed es", es == "es", str(es))

    def country_is_not_a_language_hint():
        # +33 / +49 must not move the voice. A German number often speaks French.
        check(
            "+33 hallo ignored",
            m._detect_caller_language("Hallo?", VOICES, "de", "fr") is None,
        )
        check(
            "+49 allô ignored",
            m._detect_caller_language("Allô.", VOICES, "fr", "de") is None,
        )

    def phone_prefix_hint():
        check("+33", m._language_from_phone("TEST_PHONE") == "fr")
        check("+49", m._language_from_phone("TEST_PHONE") == "de")
        check("anon", m._language_from_phone("anonymous") is None)

    def greeting_only_turns():
        for t in ("Allô.", "Hallo.", "Hi.", "Bonjour.", "Bonsoir !", "Guten Abend"):
            check(t, m._is_greeting_only(t), t)
        for t in (
            "Haben Sie geöffnet?",
            "Bonjour, une table pour deux",
            "Oui bonjour, on aimerait réserver",
        ):
            check(t, not m._is_greeting_only(t), t)

    def scrubs_last_call_bot_line():
        raw = "Mhm, bonjour! [pause] Qu'est-ce que je peux faire pour vous?"
        check("last call", m._scrub_bot_talk(raw) == "")
        check("help fr", m._scrub_bot_talk("Comment puis-je vous aider ?") == "")
        check("help de", m._scrub_bot_talk("Wie kann ich Ihnen helfen?") == "")
        check("keeps real", "geöffnet" in m._scrub_bot_talk("Ja, wir haben heute geöffnet."))
        check(
            "drops stage direction",
            m._scrub_bot_talk(
                "[switching to German as the caller has switched mid-sentence]Sehr gerne, Tino."
            )
            == "Sehr gerne, Tino.",
        )
        check("keeps pause markup", "[pause]" in m._scrub_bot_talk("Ja [pause] genau."))

    def sentence_wind_down_split():
        done, rest = m._split_off_sentences(
            "Let me check if that works for you.I'm sorry – we're closed."
        )
        check("missing space after period", done[0].startswith("Let me check"), str(done))
        check("second sentence", any("sorry" in s.lower() for s in done) or "sorry" in rest.lower(), str((done, rest)))
        done, rest = m._split_off_sentences("Hello! How")
        check("complete first", done == ["Hello!"], str(done))
        check("partial rest", rest == "How", rest)
        done, rest = m._split_off_sentences("I'm sorry – we're")
        check("no fake split", done == [], str(done))
        check("keeps fragment", rest.startswith("I'm sorry"), rest)
        ev = m._speak("Booked! Wednesday the 26th.")
        check("never hard-cuts", ev.interruptible is False, str(ev.interruptible))

    def replays_tino_closed_friday_not_yep():
        raw = (
            '{"available":false,'
            '"message":"Unfortunately we\'re closed on 2026-08-21 — it\'s a holiday. '
            'The next day we\'re open is 2026-08-26.",'
            '"nextOpenDate":"2026-08-26","closureReason":"HOLIDAY"}'
        )
        facts = m._closed_facts_from_payload(
            "check_availability", raw, {"date": "2026-08-21", "guests": 3, "time": "19:00"}
        )
        check("parsed", facts is not None, str(facts))
        check("reason", facts["reason"] == "HOLIDAY", str(facts))
        check("next", facts["next"] == "2026-08-26", str(facts))
        check(
            "open slot is not closed",
            m._closed_facts_from_payload(
                "check_availability",
                '{"available":true,"message":"We have a table"}',
            )
            is None,
        )
        line = m._replay_closed_line(facts, "en")
        check("says holiday", "holiday" in line.lower(), line)
        check("not wednesday-booked", "booked" not in line.lower(), line)
        check("next day", "2026-08-26" in line, line)
        de = m._replay_closed_line(facts, "de")
        check("german holiday", "Feiertag" in de, de)
        check("yep is an ack", m._is_bare_ack("Yep."))

    def no_switch_when_already_there():
        check("fr→fr", m._detect_caller_language("Oui ou non.", VOICES, "fr") is None)
        check("de→de", m._detect_caller_language("Ich möchte einen Tisch", VOICES, "de") is None)

    def german_still_detected():
        check(
            "de from fr",
            m._detect_caller_language("Ich möchte einen Tisch für vier Personen", VOICES, "fr") == "de",
        )

    def tool_switch_reaches_our_own_lines():
        # The bug: the tool switched the voice, the wrapper kept saying
        # "Sind Sie noch da?" in German for the rest of the call.
        state = m._CallLanguage("de", "voice-de")
        check("starts German", m._idle_reprompt(state.code) == "Sind Sie noch da?")
        state.code = "fr"
        check("idle follows", m._idle_reprompt(state.code) == "Vous êtes toujours là ?")
        check("goodbye follows", m._idle_goodbye(state.code) == "Au revoir !")
        check("filler follows", m._silent_turn_filler(state.code) == "Un instant s'il vous plaît.")
        check("hold line follows", m._transfer_hold_line(state.code).startswith("Un instant"))

    def markup_only_turns_are_silent():
        for chunks in (
            ['<emotion value="honest and warm"/>'],
            ["[pause]"],
            ["<emotion value=\"warm\"/>", " [pause] "],
            [],
            [" "],
        ):
            check(str(chunks), not m._has_speech(chunks), str(chunks))

    def real_speech_is_speech():
        for chunks in (
            ['<emotion value="warm"/>Ja, guten Abend!'],
            ["[pause] Oui, c'est vrai"],
            ["Ag", "il"],
        ):
            check(str(chunks), m._has_speech(chunks), str(chunks))

    def idle_backs_off():
        delays = [m.IDLE_S * (1.0 + m.IDLE_BACKOFF * n) for n in range(m.MAX_SILENCE_PROMPTS)]
        check("first wait is not the old 7s", delays[0] >= 10, str(delays))
        check("each wait grows", all(b > a for a, b in zip(delays, delays[1:])), str(delays))
        check("hangup takes >40s of silence", sum(delays) > 40, str(sum(delays)))

    def rueckfrage_waits_instead_of_nagging():
        """the operator got "Sind Sie noch da?" twice while thinking about a Rückfrage,
        then the twin hung up on him (~30s). The chef is answering OUR question;
        silence there is thought, not absence.
        """
        guest = [m.IDLE_S * (1.0 + m.IDLE_BACKOFF * n) for n in range(m.MAX_SILENCE_PROMPTS)]
        sup = [m.SUPERVISOR_IDLE_S * (1.0 + m.IDLE_BACKOFF * n) for n in range(m.MAX_WAIT_PROMPTS)]
        check("chef gets more than one prompt before the drop", m.MAX_WAIT_PROMPTS > 2, str(m.MAX_WAIT_PROMPTS))
        check("first nudge is not the guest's 12s", sup[0] > guest[0], f"{sup[0]} vs {guest[0]}")
        check(
            "chef gets minutes, not the old ~30s",
            sum(sup) > 180,
            f"{sum(sup):.0f}s",
        )
        check("the wait is still bounded", sum(sup) < 900, f"{sum(sup):.0f}s")

        # The line itself must not ask the chef whether they are present.
        nudge = m._supervisor_nudge("de")
        check("nudge is not 'sind Sie noch da'", "noch da" not in nudge.lower(), nudge)
        check("nudge gives them time", "zeit" in nudge.lower(), nudge)

        # A guest holding for the chef is waiting on US.
        hold = m._consult_hold_line("de")
        check("hold line is not 'sind Sie noch da'", "noch da" not in hold.lower(), hold)
        check("hold line says we are working", "frage" in hold.lower(), hold)
        for lang in ("de", "en", "fr", "it", "es", "nl", "pl", "tr", "pt", "hu", "ru", "ar"):
            check(f"hold line exists for {lang}", bool(m._consult_hold_line(lang)), lang)

    def consult_status_drives_the_hold():
        open_now = ('{"status": "pending"}', '{"status": "ringing"}', '{"status": "relay"}')
        for payload in open_now:
            check(f"open: {payload}", m._consult_still_open(payload) is True, payload)
        closed = ('{"status": "answered"}', '{"status": "taking_over"}', '{"status": "unreachable"}')
        for payload in closed:
            check(f"closed: {payload}", m._consult_still_open(payload) is False, payload)
        # ask_supervisor only ever STARTS a consult, so an opaque reply holds.
        check("opaque reply holds the guest", m._consult_still_open("sc_17") is True)
        check("dict payload works too", m._consult_still_open({"status": "answered"}) is False)

    def leg_picks_the_idle_line():
        sup = m._TenantVoiceAgent(
            _RecordingAgent(), voice_id="v-de", language="de",
            voices={"de": "v-de"}, leg="supervisor",
        )
        check("supervisor always waits", sup._waiting_on_an_answer() is True)
        check("supervisor nudges, not re-prompts", sup._idle_line() == m._supervisor_nudge("de"))

        guest = m._TenantVoiceAgent(
            _RecordingAgent(), voice_id="v-de", language="de",
            voices={"de": "v-de"},
        )
        check("guest is not waiting by default", guest._waiting_on_an_answer() is False)
        check("guest gets the normal re-prompt", guest._idle_line() == m._idle_reprompt("de"))
        guest._consulting = True
        check("guest on hold is waiting", guest._waiting_on_an_answer() is True)
        check("guest on hold gets the hold line", guest._idle_line() == m._consult_hold_line("de"))

    def supervisor_intro_is_not_a_live_connect():
        text = m._supervisor_intro({
            "question": "IEK Energiekonzept",
            "callerName": "Elias",
            "targetLabel": "the operator",
            "callerPhone": "TEST_PHONE",
        })
        check("names Rückfrage", "Rückfrage" in text, text)
        check("names the guest", "Elias" in text, text)
        check("asks the question", "IEK" in text, text)
        check("short", len(text) < 160, text)
        check("does not read the phone", "+49" not in text and "170123" not in text, text)
        check("does not say on the line", "in der Leitung" not in text, text)
        hold = m._hold_ask_line("de")
        check("hold-ask is not texting", "schreib" not in hold.lower() and "text" not in hold.lower(), hold)

    def keypad_one_takes_over_the_twin_call():
        """the operator pressed 1 on a Rückfrage and nothing happened (2026-08-20).

        The key reaches neither the SDK's default run filter nor the prompt
        (LlmAgent renders only UserTextSent), so the twin leg could never see
        it. Pinned here: the wrapper places the take-over itself and frees the
        chef's line, and never quietly turns a keypress into an LLM turn.
        """
        posted = []
        reply = ['{"ok": true, "message": "Verbinde."}']

        async def fake_post(business_id, consult_id, slug, payload, call_id=""):
            posted.append((business_id, consult_id, slug, call_id))
            return reply[0]

        original = m._post_supervisor
        m._post_supervisor = fake_post
        try:
            inner = _RecordingAgent()
            agent = m._SupervisorKeypadAgent(inner, "biz_1", "sc_1", "CA" + "a" * 32)
            out = drain(agent.process(None, m.UserDtmfSent(button="1")))

            check("take-over placed", [p[2] for p in posted] == ["supervisor-takeover"], str(posted))
            check(
                "consult and call travel with it",
                (posted[0][0], posted[0][1], posted[0][3]) == ("biz_1", "sc_1", "CA" + "a" * 32),
                str(posted),
            )
            check("the model never saw the key", inner.seen == [], str(inner.seen))
            check(
                "chef is told what happens",
                any(isinstance(o, m.AgentSendText) and "verbinde" in o.text.lower() for o in out),
                str(out),
            )
            # the platform dials the chef's OWN number for the guest - it stays busy
            # until this leg is down, so ending it is part of connecting.
            check("leg is freed for the transfer", any(isinstance(o, m.AgentEndCall) for o in out), str(out))

            drain(agent.process(None, m.UserDtmfSent(button="1")))
            check("a held key still transfers once", len(posted) == 1, str(posted))

            stray = m._SupervisorKeypadAgent(_RecordingAgent(), "biz_1", "sc_1")
            check("stray key stays silent", drain(stray.process(None, m.UserDtmfSent(button="7"))) == [], "spoke")
            check("stray key places nothing", len(posted) == 1, str(posted))

            speech_inner = _RecordingAgent()
            speaking = m._SupervisorKeypadAgent(speech_inner, "biz_1", "sc_1")
            drain(speaking.process(None, m.UserTurnEnded()))
            check("speech still reaches the model", len(speech_inner.seen) == 1, str(speech_inner.seen))

            # Consult already closed: do not hang the chef up into a transfer
            # that was never placed - let them just say the answer.
            reply[0] = '{"ok": false, "message": "consult unknown"}'
            refused = m._SupervisorKeypadAgent(_RecordingAgent(), "biz_1", "sc_gone")
            out = drain(refused.process(None, m.UserDtmfSent(button="1")))
            check("refusal keeps the chef on the line", not any(isinstance(o, m.AgentEndCall) for o in out), str(out))
            check("refusal is spoken", any(isinstance(o, m.AgentSendText) for o in out), str(out))
        finally:
            m._post_supervisor = original

    def supervisor_run_filter_includes_dtmf():
        """The wrapper only ever runs if get_agent asks for DTMF events.

        The SDK's default filter is CallStarted / UserTurnEnded / CallEnded,
        so a bare agent (what this leg returned before) drops keypresses at
        the harness - the take-over code would sit there and never be called.
        """
        from line.voice_agent_app import AgentConfig

        req = m.CallRequest(
            call_id="cartesia_CA" + "a" * 32,
            **{"from": "+4930999"},
            to="+4930111",
            agent_call_id="ac_1",
            agent=AgentConfig(),
        )
        spec = asyncio.run(
            m._build_supervisor_agent(
                {"businessId": "biz_1", "consultId": "sc_1", "question": "50 Personen?"},
                req,
            )
        )
        check("returns a filter, not a bare agent", isinstance(spec, tuple) and len(spec) == 3, str(type(spec)))
        agent, run_filter, cancel_filter = spec
        check("keypad wrapper is outermost", isinstance(agent, m._SupervisorKeypadAgent), str(type(agent)))
        check("keypresses run the agent", run_filter(m.UserDtmfSent(button="1")), "dtmf dropped")
        check("speech still runs the agent", run_filter(m.UserTurnEnded()), "speech dropped")
        check("call start still runs the agent", run_filter(m.CallStarted()), "start dropped")
        check("barge-in still cancels", cancel_filter(m.UserTurnStarted()), "no cancel")

    def twin_only_registers_no_live_connect_tools():
        none = m._make_transfer_tools("biz", "+49", "+49", live_connect_policy="never")
        check("never has no transfer tools", none == [], str(none))
        both = m._make_transfer_tools("biz", "+49", "+49", live_connect_policy="insist")
        check("insist still has both live-connect tools", len(both) == 2, str(len(both)))

    cases = [
        ("farewell phrases match", farewells),
        ("non-farewells do not match", not_farewells),
        ("bare acks match", bare_acks),
        ("real answers are not acks", not_bare_acks),
        ("block end_call on ja mid-call", block_ja_mid_call),
        ("allow end_call on real goodbye", allow_real_goodbye),
        ("each language detected mid-call", detects_each_language_mid_call),
        ("ambiguous openers never switch", ignores_ambiguous_openers),
        ("unique openers switch voice", unique_openers_switch_voice),
        ("mixed French keeps French clone", mixed_french_keeps_french_clone),
        ("short lines never inherit the previous clone", short_lines_never_inherit_the_previous_clone),
        ("a language without a clone still switches", a_language_without_a_clone_still_switches),
        ("country is not a language hint", country_is_not_a_language_hint),
        ("phone prefix hint", phone_prefix_hint),
        ("greeting-only turns stay quiet", greeting_only_turns),
        ("scrubs last-call how-can-I-help", scrubs_last_call_bot_line),
        ("sentence split winds down, never hard-cuts", sentence_wind_down_split),
        ("cut-off closed Friday is replayed, yep is not agreement", replays_tino_closed_friday_not_yep),
        ("no switch to the current language", no_switch_when_already_there),
        ("German still detected", german_still_detected),
        ("tool switch reaches our own lines", tool_switch_reaches_our_own_lines),
        ("markup-only turns count as silent", markup_only_turns_are_silent),
        ("real speech is not silent", real_speech_is_speech),
        ("idle backs off", idle_backs_off),
        ("Rückfrage waits instead of nagging", rueckfrage_waits_instead_of_nagging),
        ("consult status drives the hold", consult_status_drives_the_hold),
        ("leg picks the idle line", leg_picks_the_idle_line),
        ("supervisor intro is Rückfrage not Durchstellen", supervisor_intro_is_not_a_live_connect),
        ("twin-only registers no live-connect tools", twin_only_registers_no_live_connect_tools),
        ("pressing 1 on the twin call takes over", keypad_one_takes_over_the_twin_call),
        ("the twin leg listens for the keypad at all", supervisor_run_filter_includes_dtmf),
    ]
    for name, fn in cases:
        run(name, fn)

    total = len(cases)
    print(f"\n{('FAIL', 'PASS')[failed == 0]}  {total - failed}/{total} idle cases")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
