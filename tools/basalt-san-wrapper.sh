#!/bin/bash
# Diagnose-Wrapper: die ENGINE laeuft unter compute-sanitizer (memcheck), der Server startet sie als Kind.
# Nur der dienende Prozess wird instrumentiert; Flag-/Plan-Proben laufen unveraendert durch.
#
# Die folgende Zeile ist kein Kommentar fuer den Menschen, sondern fuer serve/config.py:engine_flags_of():
# der Server liest die vom Engine-Binary akzeptierten Flags aus dessen Bytes. Ein Shell-Skript hat keine,
# also stehen sie hier als Text -- sonst lehnt der Server jeden Engine-Flag ab ("is not an engine flag").
# engine flags: --adapt-decay --adapt-every --adapt-swaps --batch --config --control-vector --control-vector-layer-range --control-vector-scaled --conversation-cache-dir --conversation-cache-disk-mib --conversation-cache-mib --conversation-cache-min-free-mib --conversation-cache-min-tokens --conversation-cache-slots --cvec-dir --cvec-mode --decode-timing --draft-experts --draft-log --dump-mixed --dump-residual --dump-routing --eos-ids --exp --expert-cache --expert-profile --expert-profile-save --expert-profile-save-every --force-zero-doorbell --gpu --gpu-only-full --gpu-plan --gpu-stages --grammar --graph-only --greedy --kv --kv-resident --max-context --max-new --mmap-experts --model --mtp-max-t --mtp-window --no-capture --no-peer-prefill --no-pool --no-prompt-skip --peer-adapt-swaps --peer-exclusive --peer-reserve-mib --ple-delay-us --ple-inflight --ple-io --ple-row-cache --pool-affinity --pool-workers --ppl-base --ppl-out --prefill --prefill-cpu-share --profile --prompt-cache --prompt-cache-every --prompt-cache-root --python --ram-reserve-mib --relaxed-drafting --resident-budget-gib --resident-cpu-experts --resident-experts --rope-freq-base --rope-freq-scale --rope-scale --rope-scaling --route-prior --route-tail-skip --seed --serve --shared-expert-arena --short-read --slots --spec --stage-timing --stats --stop-eos --suffix-draft --tail-role-token --temperature --tokens --tokens-file --top-k --top-p --turn-token --vision --vision-external --vram-elastic --vram-reserve-mib --vram-segment-mib --wait-stats --yarn-attn-factor --yarn-beta-fast --yarn-beta-slow --yarn-ext-factor --yarn-orig-ctx
for a in "$@"; do
    if [ "$a" = "--serve" ]; then
        exec env LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
          /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
          --kernel-name "${SAN_KERNEL:-kns=mul_mat_q}" \
          --launch-timeout 600 --print-limit 8 --error-exitcode 99 \
          --log-file "${SAN_LOG:-/home/owc/basalt-san-mmq-%p.log}" \
          /home/owc/basalt-hy4-port/bin/basalt "$@"
    fi
done
exec /home/owc/basalt-hy4-port/bin/basalt "$@"

