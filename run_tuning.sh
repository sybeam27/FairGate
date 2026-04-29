#!/bin/bash
# ============================================================
# FairGate — λ_fair Re-tuning (dual-head 기준)
#
# 목적: model_dualhead.py 기준으로 각 데이터셋의 최적 λ_fair 탐색
# 탐색 범위: [0.05, 0.10, 0.20] × 9 datasets
# runs=3 (빠른 탐색), seed=27~29
#
# 결과: outputs/tune_lambda_*.csv
# 분석: python summarize_tuning.py
# ============================================================

set -e

RUNS=3
BACKBONE=GCN
LAMBDAS=("0.05" "0.10" "0.20")

# 데이터셋별 고정 하이퍼파라미터 (기존 FAIRGATE_CONFIGS 유지)
declare -A SBRS_MAP
SBRS_MAP["pokec_z"]=0.9;  SBRS_MAP["pokec_z_g"]=0.9
SBRS_MAP["pokec_n"]=0.5;  SBRS_MAP["pokec_n_g"]=0.8
SBRS_MAP["credit"]=0.5;   SBRS_MAP["recidivism"]=0.9
SBRS_MAP["income"]=0.5;   SBRS_MAP["german"]=0.95
SBRS_MAP["nba"]=0.8

declare -A DROP_MAP
DROP_MAP["pokec_z"]=0.5;  DROP_MAP["pokec_z_g"]=0.5
DROP_MAP["pokec_n"]=0.5;  DROP_MAP["pokec_n_g"]=0.5
DROP_MAP["credit"]=0.7;   DROP_MAP["recidivism"]=0.2
DROP_MAP["income"]=0.7;   DROP_MAP["german"]=0.2
DROP_MAP["nba"]=0.2

declare -A WARMUP_MAP
WARMUP_MAP["pokec_z"]=400; WARMUP_MAP["pokec_z_g"]=100
WARMUP_MAP["pokec_n"]=400; WARMUP_MAP["pokec_n_g"]=400
WARMUP_MAP["credit"]=200;  WARMUP_MAP["recidivism"]=100
WARMUP_MAP["income"]=200;  WARMUP_MAP["german"]=100
WARMUP_MAP["nba"]=200

# 튜닝 대상 데이터셋 (전체)
DATASETS=("pokec_z" "pokec_n" "pokec_z_g" "pokec_n_g"
          "credit" "recidivism" "income" "german" "nba")

echo "=========================================="
echo "  λ_fair Re-tuning — $(date '+%Y-%m-%d %H:%M')"
echo "  λ_fair: ${LAMBDAS[*]}"
echo "  datasets: ${DATASETS[*]}"
echo "  runs=$RUNS"
echo "=========================================="

TOTAL=$((${#DATASETS[@]} * ${#LAMBDAS[@]}))
COUNT=0

for DS in "${DATASETS[@]}"; do
    for LAM in "${LAMBDAS[@]}"; do
        COUNT=$((COUNT+1))
        TAG="lam${LAM/./}_${DS}"   # e.g. lam010_german
        echo ""
        echo "[$COUNT/$TOTAL] dataset=$DS  λ_fair=$LAM"

        python -m utils.train_dualhead \
            --dataset      $DS \
            --backbone     $BACKBONE \
            --device       cuda:1 \
            --save_dir     outputs/ \
            --run_name     tune_lambda \
            --lr           1e-3 \
            --weight_decay 1e-5 \
            --epochs       500 \
            --patience     501 \
            --seed         27 \
            --runs         $RUNS \
            --lambda_fair  $LAM \
            --sbrs_quantile  ${SBRS_MAP[$DS]} \
            --struct_drop    ${DROP_MAP[$DS]} \
            --warm_up        ${WARMUP_MAP[$DS]} \
            --gating_mode_override adaptive \
            --fiw_weight_mode continuous_uncert \
            --uncertainty_type dual \
            --lambda_uq    0.01 \
            --uq_width_penalty 0.05 \
            --adaptive_probe_epochs 20 \
            --adaptive_eta 1.0 \
            --adaptive_auc_tol 0.005
    done
done

echo ""
echo "=========================================="
echo "  Tuning complete — $(date '+%Y-%m-%d %H:%M')"
echo "=========================================="
echo ""
echo "결과 분석:"
echo "  python summarize_tuning.py"
