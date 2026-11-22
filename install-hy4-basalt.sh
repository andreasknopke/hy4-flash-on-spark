#!/usr/bin/env bash
# install-hy4-basalt.sh — build the Hy4-Flash NVFP4 lane on a DGX Spark (GB10, compute capability 12.1).
#
# What it does, in order:
#   1. checks the machine (aarch64, CUDA 13, cmake, gcc, python, free disk)
#   2. clones yanmindi/basalt-arm-sm121 at the pinned commit we validated
#   3. applies patches/0001-spark-sm121-fixes.patch (the sm_121 fixes: NVFP4 tiled MMQ path,
#      expert ring off, product names in MMQ launch errors)
#   4. builds it (Release, CMAKE_CUDA_ARCHITECTURES=121)
#   5. creates the server venv and installs the pinned requirements
#   6. downloads the NVFP4 pack from jesdga/Hy4-flash-Basalt (147.8 GiB, resumable)
#   7. writes the lane start script, optionally the systemd user unit, then smoke-tests it
#
# Nothing here needs root. The pack download is the long part (1-3 h on a 100 Mbit line);
# it resumes with --pack / -C -, so a broken run is cheap to continue.
#
# Usage:
#   ./install-hy4-basalt.sh                        # ~/basalt-hy4-port, port 30000, no unit
#   ./install-hy4-basalt.sh --install-unit         # also install the systemd --user unit
#   ./install-hy4-basalt.sh --pack /mnt/x.basalt   # use a pack you already have
#   ./install-hy4-basalt.sh --skip-pack            # build only (no pack)
#   ./install-hy4-basalt.sh --port 30001 --jobs 8
set -euo pipefail

# --- pinned versions (the combination this recipe was measured with) -------------
# Our fork: yanmindi's Arm64/sm_121 port + the upstream v0.5.0 merge (that is where the Hy4-flash
# support comes from) + the MMQ read-past fix + the two sm_121 prompt-fault fixes. This commit is
# byte-identical to the tree the numbers in the README were measured on.
PORT_REPO="https://github.com/andreasknopke/basalt-arm-sm121.git"
PORT_COMMIT="8172978ddc04beaaae15e70c77f6cc075833d784"   # branch hy4-port, 2026-11-22
PACK_URL="https://huggingface.co/jesdga/Hy4-flash-Basalt/resolve/main/Hy4-flash-NVFP4.basalt"
PACK_NAME="Hy4-flash-NVFP4.basalt"
PACK_BYTES=158746168309                                  # 147.8 GiB, verified on our box
PATCH_RAW="https://raw.githubusercontent.com/andreasknopke/hy4-flash-on-spark/main/patches/0001-spark-sm121-fixes.patch"

TARGET="${HOME}/basalt-hy4-port"
PORT=30000
JOBS="$(nproc 2>/dev/null || echo 4)"
PACK=""
SKIP_PACK=0
SMOKE=1
UNIT=0
KV=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) TARGET="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --pack) PACK="$2"; shift 2 ;;
        --skip-pack) SKIP_PACK=1; shift ;;
        --no-smoke) SMOKE=0; shift ;;
        --install-unit) UNIT=1; shift ;;
        --kv) KV="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- 1. machine checks -----------------------------------------------------------
say 1/7 "checking the machine"
[ "$(uname -m)" = "aarch64" ] || die "this recipe is for Arm64 (aarch64); found $(uname -m)"

NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || NVCC="/usr/local/cuda/bin/nvcc"
[ -x "$NVCC" ] || die "nvcc not found — install the CUDA toolkit (13.x) and/or set PATH=/usr/local/cuda/bin:\$PATH"
CUDA_VER="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
case "$CUDA_VER" in 13.*) : ;; *) echo "warning: built and measured with CUDA 13, found ${CUDA_VER:-unknown}" ;;
esac

for t in cmake git curl python3; do have "$t" || die "$t is missing"; done
CMAKE_VER="$(cmake --version | sed -n 's/cmake version \([0-9][0-9.]*\)/\1/p')"
awk -v v="$CMAKE_VER" 'BEGIN{split(v,a,"."); exit !(a[1]>3 || (a[1]==3 && a[2]>=24))}' \
    || die "cmake 3.24+ required, found $CMAKE_VER"
GCC_VER="$(gcc -dumpversion | cut -d. -f1)"
[ "$GCC_VER" -ge 13 ] || die "gcc 13+ required, found $GCC_VER"
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' \
    || die "python 3.11+ required, found $PY_VER"
echo "  aarch64, CUDA ${CUDA_VER:-?}, cmake ${CMAKE_VER}, gcc ${GCC_VER}, python ${PY_VER}, ${JOBS} jobs"

if [ "$SKIP_PACK" = 0 ] && [ -z "$PACK" ]; then
    FREE_GIB="$(df -BG --output=avail "$(dirname "$TARGET")" 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [ -n "$FREE_GIB" ] && [ "$FREE_GIB" -lt 160 ]; then
        die "the pack needs 147.8 GiB; only ${FREE_GIB} GiB free under $(dirname "$TARGET").
       Free space, or point --pack at an existing Hy4-flash-NVFP4.basalt, or use --skip-pack."
    fi
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "  note: port ${PORT} is already in use — stop that lane before starting this one"
fi

# --- 2. the port at the pinned commit --------------------------------------------
say 2/7 "port: andreasknopke/basalt-arm-sm121 (fork, branch hy4-port) @ ${PORT_COMMIT:0:8}"
if [ -d "$TARGET/.git" ]; then
    echo "  reusing $TARGET"
    git -C "$TARGET" fetch --quiet origin || true
else
    git clone --quiet "$PORT_REPO" "$TARGET"
fi
git -C "$TARGET" checkout --quiet "$PORT_COMMIT" || die "cannot check out ${PORT_COMMIT:0:8} in $TARGET"
echo "  at $(git -C "$TARGET" rev-parse --short HEAD) — $(git -C "$TARGET" log -1 --format=%s | cut -c1-60)"

# --- 3. our patch ----------------------------------------------------------------
say 3/7 "checking the sm_121 fixes"
# The pinned commit of our fork already carries them; the patch is applied only if you point this
# script at the upstream port instead (--repo is not exposed, so edit PORT_REPO above if you want that).
PATCH="$(dirname "$(readlink -f "$0")")/patches/0001-spark-sm121-fixes.patch"
if [ ! -f "$PATCH" ]; then
    PATCH="$(mktemp /tmp/0001-spark-sm121-fixes.XXXXXX.patch)"
    curl -fsSL "$PATCH_RAW" -o "$PATCH" || die "patch not found next to this script and could not be downloaded"
fi
if git -C "$TARGET" apply --check "$PATCH" 2>/dev/null; then
    git -C "$TARGET" apply "$PATCH"
    echo "  applied $(basename "$PATCH")"
elif git -C "$TARGET" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "  already in this commit (the fork carries them) — nothing to do"
else
    die "the patch does not apply cleanly to ${PORT_COMMIT:0:8}. Your tree may be modified; check 'git -C $TARGET status'."
fi

# --- 4. build --------------------------------------------------------------------
say 4/7 "building (Release, sm_121) — about 10 minutes"
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX="${NVCC}"
cmake -S "$TARGET" -B "$TARGET/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121 >/dev/null
cmake --build "$TARGET/build" -j "$JOBS"
[ -x "$TARGET/bin/basalt" ] || die "build finished without bin/basalt"
echo "  $(du -h "$TARGET/bin/basalt" | cut -f1) binary at $TARGET/bin/basalt"

# --- 5. server venv --------------------------------------------------------------
say 5/7 "server venv and requirements"
if [ ! -x "$TARGET/.venv/bin/python" ]; then
    python3 -m venv "$TARGET/.venv"
fi
"$TARGET/.venv/bin/pip" install --quiet --upgrade pip
"$TARGET/.venv/bin/pip" install --quiet -r "$TARGET/requirements.txt"
echo "  $("$TARGET/.venv/bin/python" -c 'import numpy, jinja2; print("numpy", numpy.__version__)')"

# --- 6. the pack -----------------------------------------------------------------
say 6/7 "NVFP4 pack"
mkdir -p "$TARGET/models"
if [ "$SKIP_PACK" = 1 ]; then
    echo "  skipped (--skip-pack)"
else
    if [ -z "$PACK" ]; then
        PACK="$TARGET/models/$PACK_NAME"
    fi
    if [ -f "$PACK" ] && [ "$(stat -c %s "$PACK")" = "$PACK_BYTES" ]; then
        echo "  already there and complete: $PACK"
    else
        echo "  downloading $(awk -v b="$PACK_BYTES" 'BEGIN{printf "%.1f", b/1073741824}') GiB to $PACK (resumable, this is the long part)"
        echo "  from $PACK_URL"
        curl -fL -C - --retry 5 --retry-delay 10 -o "$PACK" "$PACK_URL" \
            || die "download failed — rerun the same command to resume"
    fi
    SIZE="$(stat -c %s "$PACK")"
    [ "$SIZE" = "$PACK_BYTES" ] || die "pack has $SIZE bytes, expected $PACK_BYTES — incomplete or wrong file"
    echo "  $(awk -v b="$SIZE" 'BEGIN{printf "%.1f", b/1073741824}') GiB, size matches"
fi

# --- 7. lane script, unit, smoke test --------------------------------------------
# The lane must point at the pack we actually have: with --pack the file can live outside $TARGET.
if [ "$SKIP_PACK" = 1 ]; then
    PACK_RESOLVED="${TARGET}/models/${PACK_NAME}"
else
    PACK_RESOLVED="$(readlink -f "$PACK")"
    if [ "$PACK_RESOLVED" != "${TARGET}/models/${PACK_NAME}" ]; then
        echo "  note: the pack stays at $PACK_RESOLVED (outside the build tree); start-lane.sh points there"
    fi
fi
say 7/7 "lane start script and smoke test"
cat > "$TARGET/start-lane.sh" <<EOF
#!/bin/sh
# Hy4-Flash NVFP4 via Basalt in the foreground (for the systemd unit and for manual use).
# The server starts the engine itself (--engine-bin); :${PORT} is shared with other lanes, so a
# busy port is a hard error instead of a silent start into someone else's server.
set -e
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "port ${PORT} is in use - stop the other lane on it first" >&2
    exit 1
fi
cd "${TARGET}"
exec "${TARGET}/.venv/bin/python" -m serve.server \\
    --engine-bin "${TARGET}/bin/basalt" \\
    --model "${PACK_RESOLVED}" \\
    --alias hy4-flash --host 0.0.0.0 --port "${PORT}" \\
    --parallel 4 --max-context 1048576 --fit-max-tokens ${KV}
EOF
chmod +x "$TARGET/start-lane.sh"
echo "  $TARGET/start-lane.sh  (4 slots x 262144 tokens, --fit-max-tokens)"

if [ "$UNIT" = 1 ]; then
    UNIT_DIR="${HOME}/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/hy4-basalt.service" <<EOF
[Unit]
Description=Hy4-Flash NVFP4 (Basalt) on :${PORT}

[Service]
Type=simple
WorkingDirectory=${TARGET}
ExecStart=/bin/sh ${TARGET}/start-lane.sh
# No Restart: the port is shared with other lanes, so an automatic restart would race them.
Restart=no
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    echo "  unit hy4-basalt.service installed (left disabled on purpose:"
    echo "  systemctl --user start hy4-basalt / enable it only if it owns :${PORT} exclusively)"
fi

if [ "$SMOKE" = 1 ] && [ "$SKIP_PACK" = 0 ]; then
    # The smoke test loads the whole pack into unified memory (~86 GiB) and then builds the KV on top.
    # On a box that is already running another lane (or downloading something) that ends in
    # "RAM ran out loading" — which says nothing about the install. Skip it instead of failing.
    AVAIL_GIB="$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
    if [ "${AVAIL_GIB:-0}" -lt 100 ]; then
        echo "  skipped: only ${AVAIL_GIB} GiB RAM available, the smoke test needs ~100 GiB free"
        echo "           (stop other lanes and run: $TARGET/start-lane.sh)"
        SMOKE=0
    fi
fi

if [ "$SMOKE" = 1 ] && [ "$SKIP_PACK" = 0 ]; then
    echo "  starting the lane for a smoke test (the model load takes ~30 s)"
    setsid nohup "$TARGET/start-lane.sh" > "$TARGET/smoke.log" 2>&1 &
    SMOKE_PID=$!
    cleanup_smoke() { kill "$SMOKE_PID" 2>/dev/null || true; pkill -f "${TARGET}/bin/basalt" 2>/dev/null || true; }
    trap cleanup_smoke EXIT
    for _ in $(seq 1 60); do
        sleep 5
        if curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then break; fi
        kill -0 "$SMOKE_PID" 2>/dev/null || { tail -20 "$TARGET/smoke.log"; die "the lane died during startup (see $TARGET/smoke.log)"; }
    done
    curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" | head -c 200; echo
    curl -fsS -m 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"hy4-flash","max_tokens":16,"temperature":0,"messages":[{"role":"user","content":"Say OK."}]}' \
        | head -c 300; echo
    echo "  smoke test done — stopping the lane again"
    kill "$SMOKE_PID" 2>/dev/null || true
    sleep 3
    pkill -f "${TARGET}/bin/basalt" 2>/dev/null || true
fi

cat <<EOF

Done.
  port        : ${PORT}  (4 slots x 262144 tokens, KV ${KV:-default fp8})
  start       : ${TARGET}/start-lane.sh
  $( [ "$UNIT" = 1 ] && echo "or          : systemctl --user start hy4-basalt" || echo "unit        : not installed (--install-unit to add it)" )
  acceptance  : tools/basalt-repro.py crash    -> R1/R2/R3 must pass, then
                grep -c 'illegal memory' ${TARGET}/smoke.log   # 0
  benchmark   : see the README (opencode-bench-plusplus, codec-reverse-bench)
EOF
