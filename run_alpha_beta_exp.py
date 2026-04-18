"""
run_alpha_beta_exp.py — α,β 계산 방식 비교 배치 실험

variance / mutual_info / uniform 세 방식을 모든 데이터셋에 대해 실행한다.
최종 FAIRGATE_CONFIGS 하이퍼파라미터를 고정하고 alpha_beta_mode만 변경한다.

출력:
    outputs/ablation/exp_ab_variance.csv
    outputs/ablation/exp_ab_mutual_info.csv
    outputs/ablation/exp_ab_uniform.csv

실행:
    # 전체 데이터셋 × 3개 방식
    python run_alpha_beta_exp.py

    # 특정 데이터셋만
    python run_alpha_beta_exp.py --modes mutual_info --datasets german credit recidivism nba income

    # dry run으로 명령어 확인
    python run_alpha_beta_exp.py --dry_run

    # 완료 후 분석 자동 실행
    python run_alpha_beta_exp.py --analyze
"""

import os
import sys
import argparse
import subprocess

# ── 설정 ───────────────────────────────────────────────────────────────────────
DEVICE = "cuda:0"

ALL_DATASETS = [
    "pokec_z", "pokec_z_g",
    "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism",
    "nba", "income",
]

# 최종 확정 하이퍼파라미터 (FAIRGATE_CONFIGS 기준)
FAIRGATE_CONFIGS = {
    "pokec_z":    dict(lambda_fair=0.05, sbrs_quantile=0.7, struct_drop=0.5, warm_up=200),
    "pokec_n":    dict(lambda_fair=0.20, sbrs_quantile=0.7, struct_drop=0.5, warm_up=400),
    "pokec_n_g": dict(lambda_fair=0.01, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),  # 변경
    "pokec_z_g": dict(lambda_fair=0.20, sbrs_quantile=0.6, struct_drop=0.5, warm_up=100),  # 변경
    # "german":    dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),  # 변경
    "german":     dict(lambda_fair=0.15, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),
    "credit":     dict(lambda_fair=0.10, sbrs_quantile=0.8, struct_drop=0.2, warm_up=100),
    "income":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "nba":       dict(lambda_fair=0.15, sbrs_quantile=0.8, struct_drop=0.2, warm_up=200),  # 유지
    "recidivism": dict(lambda_fair=0.07, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),
}

# α,β 방식별 설정
AB_MODES = {
    "variance"   : dict(alpha_beta_mode="variance",    edge_intervention="drop"),
    "mutual_info": dict(alpha_beta_mode="mutual_info", edge_intervention="drop"),
    "uniform"    : dict(alpha_beta_mode="uniform",     edge_intervention="drop"),
}

# 고정 학습 파라미터
FIXED = dict(
    backbone         = "GCN",
    hidden_dim       = 128,
    dropout          = 0.5,
    sgc_k            = 2,
    lr               = 1e-3,
    weight_decay     = 1e-5,
    epochs           = 500,
    patience         = 501,
    runs             = 5,
    seed             = 27,
    fips_lam         = 1.0,
    mmd_alpha        = 0.3,
    dp_eo_ratio      = 0.3,
    uncertainty_type = "entropy",
    ramp_epochs      = 0,
    recal_interval   = 200,
)

# outputs/ablation 하위에 저장
OUTPUT_DIR = "outputs/ablation"
LOG_DIR    = "logs/alpha_beta"


# ── 명령어 생성 ────────────────────────────────────────────────────────────────

def build_cmd(dataset: str, mode_name: str, output_file: str) -> list:
    cfg    = FAIRGATE_CONFIGS[dataset]
    ab_cfg = AB_MODES[mode_name]
    run_name = f"ab_{mode_name}_{dataset}"

    return [
        sys.executable, "-m", "utils.train",
        "--dataset",           dataset,
        "--backbone",          FIXED["backbone"],
        "--save_dir",          ".",           # output_file이 전체 경로를 포함하므로
                                              # save_dir이 앞에 붙지 않도록 '.'으로 고정
        "--output_file",       output_file,
        "--run_name",          run_name,
        "--device",            DEVICE,
        "--hidden_dim",        str(FIXED["hidden_dim"]),
        "--dropout",           str(FIXED["dropout"]),
        "--sgc_k",             str(FIXED["sgc_k"]),
        "--lr",                str(FIXED["lr"]),
        "--weight_decay",      str(FIXED["weight_decay"]),
        "--epochs",            str(FIXED["epochs"]),
        "--patience",          str(FIXED["patience"]),
        "--runs",              str(FIXED["runs"]),
        "--seed",              str(FIXED["seed"]),
        "--lambda_fair",       str(cfg["lambda_fair"]),
        "--sbrs_quantile",     str(cfg["sbrs_quantile"]),
        "--struct_drop",       str(cfg["struct_drop"]),
        "--warm_up",           str(cfg["warm_up"]),
        "--fips_lam",          str(FIXED["fips_lam"]),
        "--mmd_alpha",         str(FIXED["mmd_alpha"]),
        "--dp_eo_ratio",       str(FIXED["dp_eo_ratio"]),
        "--uncertainty_type",  FIXED["uncertainty_type"],
        "--ramp_epochs",       str(FIXED["ramp_epochs"]),
        "--recal_interval",    str(FIXED["recal_interval"]),
        "--alpha_beta_mode",   ab_cfg["alpha_beta_mode"],
        "--edge_intervention", ab_cfg["edge_intervention"],
    ]


def run_cmd(cmd: list, dry_run: bool, log_path: str) -> bool:
    if dry_run:
        print("    $ " + " ".join(cmd))
        return True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout)
        for ln in proc.stdout.strip().splitlines()[-5:]:
            print(f"      {ln}")
        if proc.returncode != 0:
            print(f"    [WARN] 실패 — 로그: {log_path}")
            return False
    return True


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="α,β 계산 방식 비교 배치 실험",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS)
    p.add_argument("--modes", nargs="+", default=list(AB_MODES.keys()),
                   choices=list(AB_MODES.keys()),
                   help="실험할 방식 (기본: 전체 3개)")
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                   help="결과 저장 디렉토리 (기본: outputs/ablation)")
    p.add_argument("--log_dir",    type=str, default=LOG_DIR)
    p.add_argument("--dry_run",    action="store_true")
    p.add_argument("--analyze",    action="store_true",
                   help="실험 완료 후 analyze_alpha_beta.py 자동 실행")
    return p.parse_args()


def main():
    args = parse_args()

    # outputs/ablation 폴더 생성 보장
    os.makedirs(args.output_dir, exist_ok=True)

    total = len(args.datasets) * len(args.modes)
    print(f"\n{'='*65}")
    print(f"α,β Mode Comparison Experiment")
    print(f"  modes      : {args.modes}")
    print(f"  datasets   : {args.datasets}")
    print(f"  total      : {total} runs  (각 {FIXED['runs']} seeds)")
    print(f"  device     : {DEVICE}")
    print(f"  output_dir : {args.output_dir}")
    print(f"{'='*65}")

    results = {}   # mode → (ok, fail)
    step = 0

    for mode_name in args.modes:
        # 저장 경로: outputs/ablation/exp_ab_{mode}.csv
        output_file = os.path.join(args.output_dir, f"exp_ab_{mode_name}.csv")
        ok = fail = 0

        print(f"\n{'─'*65}")
        print(f"  [Mode: {mode_name}]  →  {output_file}")
        print(f"{'─'*65}")

        for ds in args.datasets:
            step += 1
            cmd      = build_cmd(ds, mode_name, output_file)
            log_path = os.path.join(args.log_dir, mode_name, f"{ds}.log")
            print(f"  [{step:2d}/{total}] {ds:<14}  mode={mode_name}")

            if run_cmd(cmd, args.dry_run, log_path):
                ok += 1
            else:
                fail += 1

        results[mode_name] = (ok, fail)
        print(f"\n  → {mode_name} 완료: {ok}/{ok+fail} 성공")

    # 최종 요약
    print(f"\n{'='*65}")
    print("전체 완료 요약")
    for mode_name, (ok, fail) in results.items():
        status = "OK" if fail == 0 else f"WARN ({fail} 실패)"
        print(f"  {mode_name:<14}: {ok}/{ok+fail}  [{status}]")
        print(f"    → {args.output_dir}/exp_ab_{mode_name}.csv")
    print(f"{'='*65}")

    # 분석 자동 실행
    if args.analyze and not args.dry_run:
        print("\n[자동 분석 실행]")
        csv_files = [
            os.path.join(args.output_dir, f"exp_ab_{m}.csv")
            for m in args.modes
        ]
        analyze_cmd = [
            sys.executable, "analyze_alpha_beta.py",
            "--csv", *csv_files,
        ]
        print("  $ " + " ".join(analyze_cmd))
        subprocess.run(analyze_cmd)


if __name__ == "__main__":
    main()