#!/usr/bin/env python3
"""Basalt Hy4-flash NVFP4 auf der Spark starten (:30000).

Startet den Python-Server (`bin/basalt serve`) aus der UNBEGRENZTEN
hermes-gateway-cgroup.  Der 4-GiB-Scope des Terminal-Tools laesst die 85 GiB
Gewichte nicht laden ("RAM ran out loading") - das ist die cgroup, nicht die
Engine (Skill basalt-arm-sm121).

Vorher: TensorFold teilt den CUDA-Pool und muss fuer die Laufzeit aus sein.

    PORT=30000 PARALLEL=4 CTX=262144 ./start-basalt-hy4.sh
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

REPO = "/home/owc/basalt-hy4-port"
PACK = f"{REPO}/models/Hy4-flash-NVFP4.basalt"
LOG = os.environ.get("LOG", "/home/owc/basalt-serve.log")
PORT = int(os.environ.get("PORT", "30000"))
PARALLEL = int(os.environ.get("PARALLEL", "4"))
CTX = int(os.environ.get("CTX", "262144"))
KV = os.environ.get("KV", "")          # leer = die Vorgabe des Packs (fp8 beim Hy4)
HOST = os.environ.get("HOST", "0.0.0.0")
# 1: jeder Slot bekommt den vollen --max-context (Agent-Benchmarks brauchen ~220k Tokens in EINEM
# Request; ohne das teilt der Server den Kontext auf die Slots auf, hier also 65536 je Slot).
SLOT_FULL = os.environ.get("SLOT_FULL", "")


def stale_basalt():
    """Alte Engine-/Server-Prozesse per PID beenden.

    NICHT `pkill -f basalt`: das Muster steht in der eigenen Kommandozeile und
    toetet die aufrufende Shell mit (Exit -15).
    """
    me = os.getpid()
    killed = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cl = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        exe = cl[0].decode(errors="replace") if cl else ""
        is_engine = exe.endswith("/bin/basalt")
        is_server = any(a.endswith(b"serve/server") or a.endswith(b"serve/server.py") for a in cl[1:3])
        if is_engine or is_server:
            try:
                os.kill(int(d), signal.SIGTERM)
                killed.append(int(d))
            except ProcessLookupError:
                pass
    return killed


def port_free(port):
    import socket
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main():
    if not os.path.exists(PACK):
        sys.exit(f"Pack fehlt: {PACK}")

    killed = stale_basalt()
    if killed:
        print(f"alte basalt-Prozesse beendet: {killed}")
        time.sleep(3)
    if not port_free(PORT):
        sys.exit(f":{PORT} ist belegt - eine andere Lane laeuft (TensorFold/Hy4-llama.cpp). "
                 f"Erst stoppen (Panel :9001); Basalt und TensorFold teilen den CUDA-Pool.")

    args = ["./bin/basalt", "serve", PACK,
            "--alias", "hy4-flash", "--host", HOST, "--port", str(PORT),
            "--parallel", str(PARALLEL), "--max-context", str(CTX),
            # Ohne das lehnt die Engine Requests ab, deren Prompt + max_tokens den Slot
            # sprengt ("requests are never truncated") - bei 32k-Prompts und 32k-Cap
            # sind das reihenweise Fehlschläge, statt eines gekuerzten Budgets.
            "--fit-max-tokens"]
    if SLOT_FULL:
        args += ["--slot-full-context"]
    if KV:
        args += ["--kv", KV]
    print("start:", " ".join(args))

    log = open(LOG, "w")
    p = subprocess.Popen(args, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"PID {p.pid} -> {LOG}")

    t0 = time.time()
    while time.time() - t0 < 600:
        if p.poll() is not None:
            print(f"Prozess beendet rc={p.returncode}")
            subprocess.run(["tail", "-25", LOG])
            return 1
        try:
            d = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read())
            st = d["data"][0].get("status", {}).get("value")
            if st == "loaded":
                print(f"loaded nach {time.time() - t0:.0f}s  http://{HOST}:{PORT}/v1")
                return 0
        except Exception:
            pass
        time.sleep(3)
    print("Timeout beim Laden - Log:")
    subprocess.run(["tail", "-25", LOG])
    return 1


if __name__ == "__main__":
    sys.exit(main())