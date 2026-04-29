#!/bin/bash
# ============================================================
# FairGate — Ablation (model_dualhead 기준, 최종 FAIRGATE_CONFIGS)
#
# 변형:
#   A5: Full FairGate  (adaptive gating + dual σ(v) + full loss)
#   A0: No fairness    (GCN baseline, λ_fair=0)
#   R4: w/o FIW        (uniform weight + full loss, λ_fair 동일)
#   R5: Fixed gating   (boundary 고정 + dual σ(v) + full loss)
#
# 대표 데이터셋: pokec_z / german / income
# runs=5, seed=27~31
# ============================================================

set -e

RUNS=5
BACKBONE=GCN
DEVICE=cuda:1

# ── 데이터셋별 하이퍼파라미터 (최종 FAIRGATE_CONFIGS 기준) ────────────────────
declare -A LAMBDA_MAP SBRS_MAP DROP_MAP WARMUP_MAP

LAMBDA_MAP["pokec_z"]=0.20;  SBRS_MAP["pokec_z"]=0.9;  DROP_MAP["pokec_z"]=0.5;  WARMUP_MAP["pokec_z"]=400
LAMBDA_MAP["pokec_n"]=0.20;  SBRS_MAP["pokec_n"]=0.5;  DROP_MAP["pokec_n"]=0.5;  WARMUP_MAP["pokec_n"]=400
LAMBDA_MAP["pokec_z_g"]=0.10;  SBRS_MAP["pokec_z_g"]=0.9;  DROP_MAP["pokec_z_g"]=0.5;  WARMUP_MAP["pokec_z_g"]=100
LAMBDA_MAP["pokec_n_g"]=0.10;  SBRS_MAP["pokec_n_g"]=0.8;  DROP_MAP["pokec_n_g"]=0.5;  WARMUP_MAP["pokec_n_g"]=400
LAMBDA_MAP["credit"]=0.05;   SBRS_MAP["credit"]=0.5;  DROP_MAP["credit"]=0.7;   WARMUP_MAP["credit"]=200
LAMBDA_MAP["recidivism"]=0.20;   SBRS_MAP["recidivism"]=0.9;  DROP_MAP["recidivism"]=0.2;   WARMUP_MAP["recidivism"]=100
LAMBDA_MAP["income"]=0.20;   SBRS_MAP["income"]=0.5;   DROP_MAP["income"]=0.7;   WARMUP_MAP["income"]=200
LAMBDA_MAP["german"]=0.20;   SBRS_MAP["german"]=0.95;  DROP_MAP["german"]=0.2;   WARMUP_MAP["german"]=100
LAMBDA_MAP["nba"]=0.20;   SBRS_MAP["nba"]=0.8;   DROP_MAP["nba"]=0.2;   WARMUP_MAP["nba"]=200

# FAIRGATE_CONFIGS = {
#     "pokec_z":     dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
#     "pokec_n":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.5, warm_up=400),
#     "pokec_z_g":   dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.5, warm_up=100),
#     "pokec_n_g":   dict(lambda_fair=0.10, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),
#     "credit":      dict(lambda_fair=0.05, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
#     "recidivism":  dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
#     "income":      dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
#     "german":      dict(lambda_fair=0.20, sbrs_quantile=0.95, struct_drop=0.2, warm_up=100),
#     "nba":         dict(lambda_fair=0.20, sbrs_quantile=0.8, struct_drop=0.2, warm_up=200),
# }


# DATASETS=("pokec_z" "german" "income")
DATASETS=("pokec_z" "pokec_n" "pokec_z_g" "pokec_n_g"
          "credit" "recidivism" "income" "german" "nba")

# ── 공통 실행 함수 ─────────────────────────────────────────────────────────────
run_variant() {
    local TAG=$1
    local GATE=$2
    local WMODE=$3
    local UNC=$4
    local LUQ=$5
    local LAMBDA_OVERRIDE=${6:-""}

    for DS in "${DATASETS[@]}"; do
        local LAM=${LAMBDA_OVERRIDE:-${LAMBDA_MAP[$DS]}}
        echo "  -> $DS  lambda_fair=$LAM  gate=$GATE  wmode=$WMODE  unc=$UNC"

        python -m utils.train_dualhead \
            --dataset       $DS \
            --backbone      $BACKBONE \
            --device        $DEVICE \
            --save_dir      outputs/ \
            --run_name      ${TAG} \
            --lr            1e-3 \
            --weight_decay  1e-5 \
            --epochs        500 \
            --patience      501 \
            --seed          27 \
            --runs          $RUNS \
            --lambda_fair   $LAM \
            --sbrs_quantile ${SBRS_MAP[$DS]} \
            --struct_drop   ${DROP_MAP[$DS]} \
            --warm_up       ${WARMUP_MAP[$DS]} \
            --gating_mode_override   $GATE \
            --fiw_weight_mode        $WMODE \
            --uncertainty_type       $UNC \
            --lambda_uq              $LUQ \
            --uq_width_penalty       0.05 \
            --adaptive_probe_epochs  20 \
            --adaptive_eta           1.0 \
            --adaptive_auc_tol       0.005
    done
}

echo "=========================================="
echo "  FairGate Ablation -- $(date '+%Y-%m-%d %H:%M')"
echo "  datasets : ${DATASETS[*]}"
echo "  runs     : $RUNS"
echo "  FAIRGATE_CONFIGS:"
echo "    pokec_z  lambda=0.20  q=0.90  drop=0.5  warm=400"
echo "    german   lambda=0.20  q=0.95  drop=0.2  warm=100"
echo "    income   lambda=0.20  q=0.50  drop=0.7  warm=200"
echo "=========================================="

# [1/4] A5: Full FairGate
echo ""
echo "[1/4] A5 -- Full FairGate"
echo "      adaptive gating + dual sigma(v) + full 3-level loss"
run_variant "abl_A5" "adaptive" "continuous_uncert" "dual" "0.01"

# [2/4] A0: No fairness
echo ""
echo "[2/4] A0 -- No fairness (GCN baseline)"
echo "      lambda_fair=0, uniform weight, no fairness loss"
run_variant "abl_A0" "none" "uniform" "entropy" "0.0" "0.0"

# [3/4] R4: w/o FIW
echo ""
echo "[3/4] R4 -- Remove FIW (uniform weight + full 3-level loss)"
echo "      lambda_fair same, uniform weight, dual UQ kept"
run_variant "abl_R4" "none" "uniform" "dual" "0.01"

# [4/4] R5: Fixed boundary gating
echo ""
echo "[4/4] R5 -- Fixed boundary gating (no topology adaptation)"
echo "      boundary fixed, dual sigma(v), full 3-level loss"
run_variant "abl_R5" "boundary" "continuous_uncert" "dual" "0.01"

echo ""
echo "=========================================="
echo "  Ablation complete -- $(date '+%Y-%m-%d %H:%M')"
echo "  Results:"
for TAG in A5 A0 R4 R5; do
    F="outputs/abl_${TAG}.csv"
    if [ -f "$F" ]; then echo "    OK $F"
    else               echo "    MISSING $F"; fi
done
echo "=========================================="
echo ""
echo "Analyze:"
echo "  python summarize_ablation.py"