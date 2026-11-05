#!/bin/bash
# WSDM'27 short re-experiments. Usage: bash scripts/run_wsdm.sh <P0|P1|P2|agg> [device]
# Logs go to logs/ so scripts/aggregate.py can read the [Rewire] lines.
set -u
STAGE=${1:?stage P0|P1|P2|agg}
DEV=${2:-cuda:0}
mkdir -p logs outputs

alloc() {  # $1 = label, rest = strategy flags
  local label=$1; shift
  python run.py --mode fairgate --device "$DEV" --run_name wsdm_alloc \
      --strict_label_budget "$@" 2>&1 | tee "logs/wsdm_alloc_${label}.log"
}

case $STAGE in
  P0)
    alloc uniform  --fiw_weight_mode uniform
    alloc random   --gating_mode_override random
    alloc boundary --gating_mode_override boundary
    alloc adaptive --gating_mode_override adaptive
    ;;
  P1)
    alloc degree     --gating_mode_override degree
    alloc loss       --gating_mode_override loss
    alloc confidence --gating_mode_override confidence
    ;;
  P2)
    for f in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.8; do
      for st in uniform boundary adaptive; do
        if [ "$st" = uniform ]; then F="--fiw_weight_mode uniform"; else F="--gating_mode_override $st"; fi
        python run.py --mode fairgate --device "$DEV" --run_name wsdm_sweep \
            --datasets pokec_z --strict_label_budget --rewire_frac "$f" --rewire_seed 0 $F \
            2>&1 | tee "logs/wsdm_sweep_f${f}_${st}.log"
      done
    done
    ;;
  agg)
    python scripts/aggregate.py --sweep_logs "logs/wsdm_sweep_*.log"
    ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
