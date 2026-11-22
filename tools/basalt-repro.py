#!/usr/bin/env python3
"""Deterministischer Crash-Repro fuer die Basalt Hy4-Flash Lane (:30000).

Rezept (aus der Session vom 21.11., reproduziert den `prefill mmq: quantize`-Fault):
    R1 = base        -> 16522 Prompt-Tokens
    R2 = base*2      -> 33022 Prompt-Tokens (frisch, 0 reused)
    R3 = base*2+extra-> 33390 (33053 reused + ~337 neu)  == Crash

Aufruf:
    python3 basalt-repro.py crash          # R1/R2/R3
    python3 basalt-repro.py ladder N...    # Prompts para*N der Reihe nach
    python3 basalt-repro.py pair N         # para*N, dann para*N+extra
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = os.environ.get("PORT", "30000")                     # andere Lane auf einem anderen Port messen
API = f"http://127.0.0.1:{PORT}/v1/chat/completions"
LOG = os.environ.get("LANE_LOG", "/home/owc/basalt-serve-dbg.log")
ENGINE_LOG = "/home/owc/.cache/basalt/9f5b71de33f93963/engine.log"

PARA = ("Der Audio-Codec arbeitet mit einer Transformationsstufe, die Koeffizienten quantisiert und "
        "anschliessend entropiecodiert. Jede Stufe hat eigene Tabellen, die aus den Rohdaten gelernt werden. ")
EXTRA = ("Nun die entscheidende Zusatzfrage: wie verhaelt sich die Bitrate, wenn die Stufe ihre Tabellen "
         "aus einem anderen Korpus lernt und die Koeffizienten feiner quantisiert werden? ") * 8


def chat(text, max_tokens=16, timeout=2400):
    body = json.dumps({"model": "hy4-flash", "max_tokens": max_tokens, "temperature": 0.0,
                       "messages": [{"role": "user", "content": text}]}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        u = d.get("usage") or {}
        return round(time.time() - t0, 1), u.get("prompt_tokens"), u.get("completion_tokens"), "OK"
    except Exception as e:                                   # noqa: BLE001
        return round(time.time() - t0, 1), None, None, "ERR: " + str(e)[:200]


def state():
    """Zustand der Lane: Slot-Belegung, Tier, letzte Engine-Zeilen."""
    out = []
    try:
        props = json.loads(urllib.request.urlopen("http://127.0.0.1:30000/props", timeout=10).read())
        out.append("props: " + json.dumps({k: props.get(k) for k in
                                           ("n_ctx", "total_slots", "kv_unified", "n_ctx_slot") if k in props}))
    except Exception as e:                                   # noqa: BLE001
        out.append("props: ERR " + str(e)[:80])
    r = subprocess.run(f"tail -12 {LOG}", shell=True, capture_output=True, text=True).stdout
    out.append("--- serve.log ---\n" + r.rstrip())
    e = subprocess.run(f"grep -E '^(PROMPT|PMATCH)|illegal memory|basalt: error|chunk|TIER' {ENGINE_LOG} | tail -14",
                       shell=True, capture_output=True, text=True).stdout
    out.append("--- engine.log ---\n" + e.rstrip())
    crashes = subprocess.run(f"grep -c 'illegal memory' {LOG}", shell=True, capture_output=True, text=True).stdout.strip()
    out.append(f"crashes so far (serve.log): {crashes}")
    return "\n".join(out)


def step(label, text):
    dt, pt, ct, st = chat(text)
    print(f"{label}: {dt}s prompt_tokens={pt} completion={ct} -> {st}", flush=True)
    print(state(), flush=True)
    print("=" * 70, flush=True)
    return st


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "crash"
    if mode == "crash":
        step("R1 base (16522)", PARA * 300)
        step("R2 base*2 (33022)", PARA * 600)
        step("R3 base*2+extra (33390)", PARA * 600 + EXTRA)
    elif mode == "ladder":
        for n in [int(x) for x in sys.argv[2:]]:
            step(f"para*{n}", PARA * n)
    elif mode == "pair":
        n = int(sys.argv[2])
        step(f"pair A para*{n}", PARA * n)
        step(f"pair B para*{n}+extra", PARA * n + EXTRA)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
