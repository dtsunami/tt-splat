#!/bin/bash
# A/B perf probe: cost of the host grad round-trip at device_resident.py:236
# (= the headroom the on-device sanitize, fix #1, would recover).
#
#   Arm A = today's behavior   : TT_GRAD_CLIP=10  (round-trip ON,  grads -> host -> Adam)
#   Arm B = resident fast path : TT_GRAD_CLIP=0   (round-trip OFF, grads stay on device)
#
# densify OFF (TT_DENSIFY=0) so N is constant -> the only variable is the round-trip.
# The cost shows up in the C stage (t7-t6 wraps both the to_torch and Adam).
#
# usage:  STEPS=200 TT_SIZE=1600 ./bench_ab.sh
set -e
cd ~/tt-splat
export TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal
PY=~/tt-metal/python_env/bin/python
STEPS=${STEPS:-200}
SIZE=${TT_SIZE:-1600}

run() {  # $1=label  $2=gradclip
  TT_DEVICE_RESIDENT=1 TT_SIZE=$SIZE TT_DENSIFY=0 TT_GRAD_CLIP=$2 \
    "$PY" server/train_tt.py --dataset work/scene --output "work/bench_$1" --steps "$STEPS" 2>/dev/null \
    | grep "per-step ms"
}

echo "### A/B grad round-trip probe — $STEPS steps @ ${SIZE}px, densify OFF ###"
A=$(run A 10); echo "ARM A (round-trip ON,  TT_GRAD_CLIP=10): $A"
B=$(run B 0);  echo "ARM B (resident path,  TT_GRAD_CLIP=0 ): $B"

# diff the C and step fields
python3 - "$A" "$B" <<'PY'
import sys, re
def parse(s):
    return {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', s)}
a, b = parse(sys.argv[1]), parse(sys.argv[2])
print("\n  stage     ARM A (rt on)   ARM B (resident)   delta (A-B)")
for k in ("D", "C", "step"):
    da = a.get(k, 0); db = b.get(k, 0)
    print(f"  {k:<6} {da:>10.1f} ms {db:>13.1f} ms {da-db:>11.1f} ms")
c = a.get("C", 0) - b.get("C", 0)
print(f"\n  >> round-trip cost recovered by fix #1 ≈ {c:.1f} ms/step "
      f"({100*c/max(a.get('step',1),1e-9):.1f}% of step)")
PY
