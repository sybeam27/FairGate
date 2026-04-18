"""
run.py — FairGate + 비교 모델 전체 실험 실행기

실행:
    python run.py --run_name exp --mode all
    python run.py --run_name exp_baselines --mode baselines --models FairGT
    python run.py --run_name exp_fairgate_v3 --mode fairgate --datasets pokec_z pokec_n pokec_z_g pokec_n_g
    python run.py --run_name exp_v2 --datasets pokec_n recidivism
    python run.py --run_name exp_baselines --mode baselines --models NIFTY FairGB FairGNN EDITS FairEdit --datasets income
    python run.py --run_name exp_v2 --dry_run
    
결과:
    FairGate + 비교 모델 결과 모두 누적 저장 (dataset, model 열로 구분)
"""

# # 기본
# python run.py --run_name exp_fairgate_v2 --mode fairgate --datasets pokec_z_g pokec_n_g german
# python run.py --run_name exp_fairgate_sage --mode fairgate --backbone GraphSAGE
# python run.py --run_name exp_fairgate_sgc --mode fairgate --backbone SGC

# # 3번 한계 ablation
# python train.py --dataset pokec_z --backbone GCN --alpha_beta_mode mutual_info
# python train.py --dataset pokec_z --backbone GCN --alpha_beta_mode uniform

# # 4번 한계 ablation
# python train.py --dataset credit --backbone GCN --edge_intervention scale

import subprocess
import sys
import argparse
from datetime import datetime


DEVICE = 'cuda:1'

# ── 공통 학습 설정 (train.py / train_baselines.py 동일) ──────────────────
COMMON_TRAIN = {
    "lr":           1e-3,
    "weight_decay": 1e-5,
    "epochs":       500,
    "patience":     501,
    "seed":         27,
    "runs":         5,
}

# ── FairGate 데이터셋별 하이퍼파라미터 ───────────────────────────────────
# FAIRGATE_CONFIGS = {
#     "pokec_z":    dict(lambda_fair=0.05, sbrs_quantile=0.7, struct_drop=0.5, warm_up=200),
#     "pokec_n":    dict(lambda_fair=0.20, sbrs_quantile=0.7, struct_drop=0.5, warm_up=400),
#     "pokec_n_g": dict(lambda_fair=0.01, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),  # 변경
#     "pokec_z_g": dict(lambda_fair=0.20, sbrs_quantile=0.6, struct_drop=0.5, warm_up=100),  # 변경
#     # "german":    dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),  # 변경
#     "german":     dict(lambda_fair=0.15, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),
#     "credit":     dict(lambda_fair=0.10, sbrs_quantile=0.8, struct_drop=0.2, warm_up=100),
#     "income":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
#     "nba":       dict(lambda_fair=0.15, sbrs_quantile=0.8, struct_drop=0.2, warm_up=200),  # 유지
#     "recidivism": dict(lambda_fair=0.07, sbrs_quantile=0.7, struct_drop=0.3, warm_up=100),

# }

# 수정본
FAIRGATE_CONFIGS = {
    # ── Pokec 계열 ──────────────────────────────────────────────────────────
    "pokec_z":    dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
    "pokec_z_g":  dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.5, warm_up=100),
    "pokec_n":    dict(lambda_fair=0.15, sbrs_quantile=0.5, struct_drop=0.5, warm_up=400),
    "pokec_n_g":  dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
    # ── 소규모 그래프 ────────────────────────────────────────────────────────
    "credit":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "recidivism": dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "income":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "german":     dict(lambda_fair=0.20, sbrs_quantile=0.7, struct_drop=0.2, warm_up=100),
    "nba":        dict(lambda_fair=0.40, sbrs_quantile=0.5, struct_drop=0.3, warm_up=200),
}


# ── 비교 모델 목록 ────────────────────────────────────────────────────────
ALL_DATASETS = [
    "pokec_z", "pokec_n", "pokec_z_g", "pokec_n_g",
    "german", "credit", "income", "recidivism", "nba",
]

BASELINE_MODELS = [
    # "GNN",
    "NIFTY", "FairGB", "FairGT",
    "EDITS", "FairEdit",
    "FairGNN", "FairVGNN",
    "FairWalk", "CrossWalk",
]


# ============================================================
# Command builders
# ============================================================

def build_fairgate_cmd(dataset: str, backbone: str, run_name: str) -> list:
    cfg = FAIRGATE_CONFIGS.get(dataset, FAIRGATE_CONFIGS["pokec_z"])
    cmd = [sys.executable, "-m", "utils.train",
           "--dataset",  dataset,
           "--backbone", backbone,
           "--device",   DEVICE,
           "--save_dir", "outputs/",
           "--run_name", run_name]
    for k, v in {**COMMON_TRAIN, **cfg}.items():
        cmd += [f"--{k}", str(v)]
    return cmd


def build_baseline_cmd(model: str, dataset: str, run_name: str) -> list:
    cmd = [sys.executable, "-m", "utils.train_baselines",
           "--model",      model,
           "--dataset",    dataset,
           "--device",     DEVICE,
           "--save_dir",   "outputs/",
           "--run_name",   run_name,
           "--hidden_dim", "128"]
    for k, v in COMMON_TRAIN.items():
        cmd += [f"--{k}", str(v)]
    return cmd


# ============================================================
# Runner
# ============================================================

def run_cmd(cmd: list, dry_run: bool) -> tuple:
    """(status, elapsed_sec) 반환"""
    print(f"\n  $ {' '.join(cmd)}")
    if dry_run:
        print("  (dry_run — skipped)")
        return "skipped", 0

    print(f"  started {datetime.now().strftime('%H:%M:%S')}")
    t0      = datetime.now()
    proc    = subprocess.run(cmd, text=True)
    elapsed = (datetime.now() - t0).seconds
    status  = "OK" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"  {status} | {elapsed//60}m {elapsed%60}s")
    return status, elapsed


def run_all(args):
    run_name        = args.run_name
    datasets        = args.datasets
    backbone        = args.backbone
    baseline_models = args.models
    mode            = args.mode
    dry_run         = args.dry_run

    results = {}

    print(f"\n{'='*65}")
    print(f"  run_name : {run_name}")
    print(f"  output   : outputs/{run_name}.csv")
    print(f"  mode     : {mode}")
    if mode in ("fairgate", "all"):
        print(f"  FairGate : backbone={backbone}, {len(datasets)} datasets")
    if mode in ("baselines", "all"):
        print(f"  baselines: {len(baseline_models)} models x {len(datasets)} datasets")
    print(f"  common   : lr={COMMON_TRAIN['lr']}  wd={COMMON_TRAIN['weight_decay']}  "
          f"epochs={COMMON_TRAIN['epochs']}  patience={COMMON_TRAIN['patience']}  "
          f"runs={COMMON_TRAIN['runs']}  seed={COMMON_TRAIN['seed']}")
    print(f"{'='*65}")

    # ── OOM 조합: 실행 자체를 건너뜀 ────────────────────────────
    OOM_SKIP = {
        # "EDITS":    {"pokec_z", "pokec_n"},
        # "FairEdit": {"pokec_z", "pokec_n"},
    }

    # ── 비교 모델 ─────────────────────────────────────────────
    if mode in ("baselines", "all"):
        print(f"\n{'─'*65}")
        print(f"  [Baselines]")
        print(f"{'─'*65}")
        for model in baseline_models:
            for ds in datasets:
                label = f"{model}/{ds}"
                if ds in OOM_SKIP.get(model, set()):
                    print(f"\n[{label}]  skipped (OOM)")
                    results[label] = ("skipped_oom", 0)
                    continue
                print(f"\n[{label}]")
                status, elapsed = run_cmd(
                    build_baseline_cmd(model, ds, run_name), dry_run)
                results[label] = (status, elapsed)

    # ── FairGate ─────────────────────────────────────────────
    if mode in ("fairgate", "all"):
        print(f"\n{'─'*65}")
        print(f"  [FairGate / {backbone}]")
        print(f"{'─'*65}")
        for ds in datasets:
            label = f"FairGate/{ds}"
            print(f"\n[{label}]")
            status, elapsed = run_cmd(
                build_fairgate_cmd(ds, backbone, run_name), dry_run)
            results[label] = (status, elapsed)

    # ── Summary ───────────────────────────────────────────────
    if not dry_run:
        ok    = [k for k, (v, _) in results.items() if v == "OK"]
        skip_oom = [k for k, (v, _) in results.items() if v == "skipped_oom"]
        fail  = [k for k, (v, _) in results.items() if v not in ("OK", "skipped", "skipped_oom")]
        times = {k: e for k, (v, e) in results.items() if v == "OK"}

        print(f"\n{'='*65}")
        print(f"  Done  {len(ok)}/{len(results)}  |  "
              f"Results -> outputs/{run_name}.csv")

        if times:
            elapsed_vals = list(times.values())
            total   = sum(elapsed_vals)
            avg     = total / len(elapsed_vals)
            print(f"  Time  total={total//60}m{total%60}s  "
                  f"avg={avg//60:.0f}m{avg%60:.0f}s per experiment")
            # 가장 오래 걸린 top-3 출력
            top3 = sorted(times.items(), key=lambda x: x[1], reverse=True)[:3]
            for label, sec in top3:
                print(f"    {label}: {sec//60}m{sec%60}s")

        if skip_oom:
            print(f"  OOM skip: {', '.join(skip_oom)}")
        if fail:
            print(f"  FAILED: {', '.join(fail)}")
        print(f"{'='*65}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_name", type=str, required=True,
                   help="결과 파일 이름 (확장자 제외). 예: exp_0413_v1")
    p.add_argument("--mode",     type=str, default="fairgate",
                   choices=["fairgate", "baselines", "all"])
    p.add_argument("--backbone", type=str, default="GCN",
                   choices=["GCN", "GraphSAGE", "SGC"])
    p.add_argument("--datasets", nargs="+",
                   default=ALL_DATASETS,
                   help="실행할 데이터셋 목록 (기본: 전체 9개)")
    p.add_argument("--models",   nargs="+",
                   default=BASELINE_MODELS,
                   help="비교 모델 목록 (baselines/all 모드)")
    p.add_argument("--dry_run",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args)