#!/usr/bin/env python3
"""No-SDK tests for the Jev hint plumbing in main.py."""
import importlib.util, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("cartesia_main", ROOT / "main.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class R:
    def __init__(self, code, body=None, boom=False):
        self.status_code, self._b, self._boom = code, body, boom
    def json(self):
        if self._boom: raise ValueError("bad json")
        return self._b

assert m._jev_hint_from(R(200, {"ok": True, "jev_hint": " [hint] x "})) == "[hint] x"
assert m._jev_hint_from(R(200, {"ok": True})) == ""
assert m._jev_hint_from(R(500, {"jev_hint": "x"})) == ""
assert m._jev_hint_from(R(200, boom=True)) == ""
assert m._jev_hint_from(R(200, {"jev_hint": 5})) == ""
assert m._jev_hint_from(None) == ""
assert len(m._jev_hint_from(R(200, {"jev_hint": "a" * 1000}))) == 400
print("test_jev ok")
