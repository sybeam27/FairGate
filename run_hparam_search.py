"""
run_hparam_search.py — FairGate 핵심 공정성 하이퍼파라미터 탐색

탐색 대상 (우선순위 순):
    1) lambda_fair   : 공정성-정확도 tradeoff 직접 제어 (가장 중요)
    2) sbrs_quantile : 개입 노드 비율 결정
    3) struct_drop   : 구조 손실 강도
    4) fips_lam      : 불확실성 증폭 계수
    5) warm_up       : warm-up 길이
    6) mmd_alpha     : representation loss 혼합 비율
    7) dp_eo_ratio   : DP/EO 균형

전략:
    - "random" (기본): 전체 grid에서 무작위 N개 샘플링. 빠른 탐색에 권장.
    - "full"         : 전체 grid 실행. 조합 수가 많으므로 단일 데이터셋에만 사용.

결과: outputs/hparam/{dataset}.csv 에 데이터셋별로 누적 저장.
완료 후 각 데이터셋의 Top-5 config를 출력하고 run.py FAIRGATE_CONFIGS에 반영.

실행 예:
    # 단일 데이터셋 random search (30개 조합)
    python run_hparam_search.py --datasets german --n_samples 30

    # 여러 데이터셋 동시 탐색
    python run_hparam_search.py --datasets german recidivism nba --n_samples 20

    # 전체 데이터셋 탐색
    python run_hparam_search.py --n_samples 20

    # dry run으로 명령어 확인
    python run_hparam_search.py --datasets german --dry_run
"""

import os
import sys
import random
import argparse
import subprocess
import itertools

import pandas as pd


# ── 디바이스 ───────────────────────────────────────────────────────────────────
DEVICE = "cuda:1"

# ── 탐색 대상 데이터셋 ─────────────────────────────────────────────────────────
ALL_DATASETS = [
    "pokec_z", "pokec_n", "pokec_z_g", "pokec_n_g",
    "german", "credit", "income", "recidivism", "nba",
]

# ── 탐색 범위 ──────────────────────────────────────────────────────────────────
# 우선순위 높은 파라미터일수록 더 촘촘하게 설정
GRID = {
    "lambda_fair"  : [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],  # 가장 중요
    "sbrs_quantile": [0.5, 0.6, 0.7, 0.8, 0.9],
    "struct_drop"  : [0.2, 0.3, 0.5, 0.7],
    "fips_lam"     : [0.5, 1.0, 2.0],
    "warm_up"      : [100, 200, 400],
    "mmd_alpha"    : [0.1, 0.3, 0.5],
    "dp_eo_ratio"  : [0.2, 0.3, 0.5],
}

# ── 고정 파라미터 ──────────────────────────────────────────────────────────────
FIXED = {
    "hidden_dim"       : 128,
    "dropout"          : 0.5,
    "lr"               : 1e-3,
    "weight_decay"     : 1e-5,
    "epochs"           : 1000,
    "patience"         : 100,
    "runs"             : 3,       # 탐색 속도용; 최종 실험은 5
    "seed"             : 27,
    "recal_interval"   : 200,
    "uncertainty_type" : "entropy",
    "ramp_epochs"      : 0,
    "alpha_beta_mode"  : "variance",
    "edge_intervention": "drop",
}


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

def all_combinations() -> list:
    keys   = list(GRID.keys())
    values = list(GRID.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def build_cmd(dataset: str, backbone: str, output_file: str,
              cfg: dict, run_name: str) -> list:
    return [
        sys.executable, "train.py",
        "--dataset",           dataset,
        "--backbone",          backbone,
        "--output_file",       output_file,
        "--run_name",          run_name,
        "--device",            DEVICE,
        "--hidden_dim",        str(cfg["hidden_dim"]),
        "--dropout",           str(cfg["dropout"]),
        "--lr",                str(cfg["lr"]),
        "--weight_decay",      str(cfg["weight_decay"]),
        "--epochs",            str(cfg["epochs"]),
        "--patience",          str(cfg["patience"]),
        "--runs",              str(cfg["runs"]),
        "--seed",              str(cfg["seed"]),
        "--lambda_fair",       str(cfg["lambda_fair"]),
        "--sbrs_quantile",     str(cfg["sbrs_quantile"]),
        "--fips_lam",          str(cfg["fips_lam"]),
        "--mmd_alpha",         str(cfg["mmd_alpha"]),
        "--struct_drop",       str(cfg["struct_drop"]),
        "--warm_up",           str(cfg["warm_up"]),
        "--dp_eo_ratio",       str(cfg["dp_eo_ratio"]),
        "--recal_interval",    str(cfg["recal_interval"]),
        "--uncertainty_type",  cfg["uncertainty_type"],
        "--ramp_epochs",       str(cfg["ramp_epochs"]),
        "--alpha_beta_mode",   cfg["alpha_beta_mode"],
        "--edge_intervention", cfg["edge_intervention"],
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
        lines = proc.stdout.strip().splitlines()
        for ln in lines[-4:]:
            print(f"      {ln}")
        if proc.returncode != 0:
            print(f"    [WARN] 실패 — 로그: {log_path}")
            return False
    return True


def cfg_tag(hcfg: dict) -> str:
    return (f"lf{hcfg['lambda_fair']}_q{hcfg['sbrs_quantile']}_"
            f"sd{hcfg['struct_drop']}_fl{hcfg['fips_lam']}_"
            f"wu{hcfg['warm_up']}_ma{hcfg['mmd_alpha']}_"
            f"dp{hcfg['dp_eo_ratio']}")


def print_top5(output_file: str, dataset: str):
    """탐색 완료 후 해당 데이터셋의 Top-5 config 출력."""
    try:
        df = pd.read_csv(output_file)
        df = df[df["dataset"] == dataset].copy()
        if df.empty:
            return
        if not {"acc_mean", "dp_mean", "eo_mean"}.issubset(df.columns):
            return

        df["score"] = df["acc_mean"] - df["dp_mean"].abs() - df["eo_mean"].abs()
        cols = ["lambda_fair", "sbrs_quantile", "struct_drop",
                "fips_lam", "warm_up", "mmd_alpha", "dp_eo_ratio",
                "acc_mean", "dp_mean", "eo_mean", "score"]
        cols = [c for c in cols if c in df.columns]
        top  = df.nlargest(5, "score")[cols]

        print(f"\n  Top-5 configs [{dataset}]  (acc - |dp| - |eo|):")
        print(top.to_string(index=False))

        best = top.iloc[0]
        print(f"\n  → run.py FAIRGATE_CONFIGS['{dataset}'] 추천:")
        print(f"    \"lambda_fair\":   {best.get('lambda_fair', '?')},")
        print(f"    \"sbrs_quantile\": {best.get('sbrs_quantile', '?')},")
        print(f"    \"fips_lam\":      {best.get('fips_lam', '?')},")
        print(f"    \"mmd_alpha\":     {best.get('mmd_alpha', '?')},")
        print(f"    \"struct_drop\":   {best.get('struct_drop', '?')},")
        print(f"    \"warm_up\":       {int(best.get('warm_up', 200))},")
        print(f"    \"dp_eo_ratio\":   {best.get('dp_eo_ratio', '?')},")

    except Exception as e:
        print(f"  [결과 요약 실패] {e}")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="FairGate 하이퍼파라미터 탐색"
    )
    p.add_argument("--datasets",  nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS,
                   help="탐색할 데이터셋 목록 (기본: 전체 9개)")
    p.add_argument("--backbone",  type=str, default="GCN",
                   choices=["GCN", "GraphSAGE", "SGC"])
    p.add_argument("--mode",      type=str, default="random",
                   choices=["full", "random"],
                   help="full=전체 grid, random=무작위 샘플링")
    p.add_argument("--n_samples", type=int, default=30,
                   help="random 모드에서 데이터셋당 샘플링 수")
    p.add_argument("--output_dir", type=str, default="outputs/hparam",
                   help="결과 저장 디렉토리 (데이터셋별 CSV 분리)")
    p.add_argument("--log_dir",   type=str, default="logs/hparam")
    p.add_argument("--search_seed", type=int, default=42,
                   help="random 샘플링 재현성용 시드")
    p.add_argument("--dry_run",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_combos = all_combinations()
    total_full = len(all_combos)

    print(f"\n{'='*70}")
    print(f"FairGate Hparam Search")
    print(f"  mode      : {args.mode}"
          + (f" (데이터셋당 {args.n_samples}/{total_full} 조합)"
             if args.mode == "random" else f" (전체 {total_full} 조합)"))
    print(f"  datasets  : {args.datasets}")
    print(f"  backbone  : {args.backbone}")
    print(f"  device    : {DEVICE}")
    print(f"  output    : {args.output_dir}/hparam_{{dataset}}.csv")
    print(f"{'='*70}")

    for dataset in args.datasets:
        output_file = os.path.join(args.output_dir, f"hparam_{dataset}.csv")

        # 데이터셋별 독립 시드 (재현 가능)
        random.seed(args.search_seed + hash(dataset) % 10000)
        if args.mode == "random":
            combos = random.sample(all_combos, min(args.n_samples, total_full))
        else:
            combos = list(all_combos)

        print(f"\n{'─'*70}")
        print(f"  [{dataset}]  {len(combos)} 조합  →  {output_file}")
        print(f"{'─'*70}")

        ok = 0
        for i, hcfg in enumerate(combos, 1):
            cfg      = {**FIXED, **hcfg}
            tag      = f"hp_{dataset}_{args.backbone}_{cfg_tag(hcfg)}"
            log_path = os.path.join(args.log_dir, dataset, f"{tag}.log")
            cmd      = build_cmd(dataset, args.backbone, output_file, cfg, tag)

            print(f"  [{i:3d}/{len(combos)}] {cfg_tag(hcfg)}")
            if run_cmd(cmd, args.dry_run, log_path):
                ok += 1

        print(f"\n  [{dataset}] 완료: {ok}/{len(combos)} 성공")
        if not args.dry_run:
            print_top5(output_file, dataset)

    print(f"\n{'='*70}")
    print(f"전체 탐색 완료. 결과: {args.output_dir}/hparam_{{dataset}}.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
