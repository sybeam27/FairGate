"""
analyze_fiw_gating.py — FIW Gating 분석

학습된 FairGate 모델에서 FIW가 어떤 노드를 선택했는지,
boundary score, uncertainty, 개입 강도의 분포를 분석한다.

분석 항목:
    1) Gated 노드 비율 및 집단 간 분포
    2) Boundary score vs fairness 위반 상관관계
    3) Gated/non-gated 노드의 예측 정확도 비교
    4) 개입 전후 representation 변화 (MMD)

출력:
    outputs/analysis/fiw_gating_stats.csv
    outputs/analysis/fiw_gating_plot.py

실행:
    python analyze_fiw_gating.py --dataset pokec_z
    python analyze_fiw_gating.py --dataset pokec_z german nba
"""

import os, sys, argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data  import get_dataset
from utils.model import (FairGate, compute_fiw_weights,
                          _compute_edge_homophily, _compute_graph_stats)

DEVICE = "cuda:0"

FAIRGATE_CONFIGS = {
    # ── Pokec 계열 ──────────────────────────────────────────────────────────
    "pokec_z":    dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.5, warm_up=400),
    "pokec_z_g":  dict(lambda_fair=0.20, sbrs_quantile=0.6, struct_drop=0.5, warm_up=100),
    "pokec_n":    dict(lambda_fair=0.15, sbrs_quantile=0.5, struct_drop=0.5, warm_up=400),
    "pokec_n_g":  dict(lambda_fair=0.01, sbrs_quantile=0.8, struct_drop=0.5, warm_up=400),
    # ── 소규모 그래프 ────────────────────────────────────────────────────────
    "credit":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "recidivism": dict(lambda_fair=0.10, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "income":     dict(lambda_fair=0.20, sbrs_quantile=0.5, struct_drop=0.7, warm_up=200),
    "german":     dict(lambda_fair=0.20, sbrs_quantile=0.9, struct_drop=0.2, warm_up=100),
    "nba":        dict(lambda_fair=0.40, sbrs_quantile=0.5, struct_drop=0.3, warm_up=200),
}

FIXED = dict(
    hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=1, seed=27, fips_lam=1.0, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200,
    alpha_beta_mode="variance", edge_intervention="drop",
)


def train_model(dataset: str, device: str) -> tuple:
    """모델 학습 후 data, model, node_weights 반환"""
    cfg  = FAIRGATE_CONFIGS[dataset]
    data, sens_idx, x_min, x_max = get_dataset(dataset)
    data = data.to(device)

    model = FairGate(
        in_feats         = data.x.size(1),
        h_feats          = FIXED["hidden_dim"],
        device           = device,
        backbone         = "GCN",
        dropout          = FIXED["dropout"],
        sgc_k            = FIXED["sgc_k"],
        lambda_fair      = cfg["lambda_fair"],
        sbrs_quantile    = cfg["sbrs_quantile"],
        fips_lam         = FIXED["fips_lam"],
        mmd_alpha        = FIXED["mmd_alpha"],
        struct_drop      = cfg["struct_drop"],
        warm_up          = cfg["warm_up"],
        dp_eo_ratio      = FIXED["dp_eo_ratio"],
        uncertainty_type = FIXED["uncertainty_type"],
        recal_interval   = FIXED["recal_interval"],
        alpha_beta_mode  = FIXED["alpha_beta_mode"],
        edge_intervention= FIXED["edge_intervention"],
    )

    torch.manual_seed(FIXED["seed"])
    model.fit(data, epochs=FIXED["epochs"], lr=FIXED["lr"],
              weight_decay=FIXED["weight_decay"],
              patience=FIXED["patience"], verbose=False)
    return data, model


def analyze_gating(dataset: str, data, model, output_dir: str) -> dict:
    """FIW gating 통계 분석"""
    device = data.x.device
    data.alpha_beta_mode = FIXED["alpha_beta_mode"]

    # FIW 가중치 재계산 (학습된 모델 기준)
    node_w, meta = compute_fiw_weights(
        data, model=model.model,
        sbrs_quantile=model.sbrs_quantile,
        fips_lam=model.fips_lam,
        uncertainty_type=model.uncertainty_type,
    )

    N    = data.x.size(0)
    sens = data.sens.cpu()
    src, dst = data.edge_index.cpu()

    # 경계 노드 여부
    is_inter  = (sens[src] != sens[dst])
    has_inter = torch.zeros(N, dtype=torch.bool)
    has_inter[src[is_inter]] = True

    # Degree
    deg = torch.zeros(N).scatter_add_(0, src, torch.ones(src.size(0)))

    node_w_cpu = node_w.cpu()

    # Gated 노드 (threshold: sbrs_quantile)
    threshold = torch.quantile(node_w_cpu, model.sbrs_quantile)
    is_gated  = (node_w_cpu >= threshold)

    # 집단별 gated 비율
    g0_gated = is_gated[sens==0].float().mean().item()
    g1_gated = is_gated[sens==1].float().mean().item()

    # 경계 노드 중 gated 비율
    bnd_gated    = is_gated[has_inter].float().mean().item() if has_inter.any() else 0.0
    nonbnd_gated = is_gated[~has_inter].float().mean().item() if (~has_inter).any() else 0.0

    # Gated vs non-gated 노드의 예측 정확도
    model.model.eval()
    with torch.no_grad():
        out = model.model(data)
        if isinstance(out, tuple): out = out[0]
        pred = (torch.sigmoid(out.view(-1)) > 0.5).cpu().long()
        correct = (pred == data.y.cpu().long())

    gated_acc    = correct[is_gated].float().mean().item()
    nongated_acc = correct[~is_gated].float().mean().item()

    # 가중치 분포 통계
    w_mean = node_w_cpu.mean().item()
    w_std  = node_w_cpu.std().item()
    w_gated_mean = node_w_cpu[is_gated].mean().item()

    # 집단 간 gated 비율 차이 (fairness of gating)
    gating_bias = abs(g0_gated - g1_gated)

    stats = {
        "dataset"       : dataset,
        "n_total"       : N,
        "n_gated"       : int(is_gated.sum().item()),
        "gated_ratio"   : round(is_gated.float().mean().item(), 4),
        "g0_gated_ratio": round(g0_gated, 4),
        "g1_gated_ratio": round(g1_gated, 4),
        "gating_bias"   : round(gating_bias, 4),
        "bnd_gated_ratio"   : round(bnd_gated, 4),
        "nonbnd_gated_ratio": round(nonbnd_gated, 4),
        "gated_acc"     : round(gated_acc, 4),
        "nongated_acc"  : round(nongated_acc, 4),
        "w_mean"        : round(w_mean, 4),
        "w_std"         : round(w_std, 4),
        "w_gated_mean"  : round(w_gated_mean, 4),
        "alpha"         : round(meta.get("alpha", 0), 4),
        "beta"          : round(meta.get("beta", 0), 4),
        "boundary_threshold": round(meta.get("boundary_threshold", 0), 4),
        "homophily"     : round(meta.get("homophily", 0), 4),
    }

    print(f"\n[{dataset}] FIW Gating 분석")
    print(f"  Gated: {stats['n_gated']}/{N} ({stats['gated_ratio']*100:.1f}%)")
    print(f"  집단별 gated 비율: G0={stats['g0_gated_ratio']:.4f}  G1={stats['g1_gated_ratio']:.4f}  bias={stats['gating_bias']:.4f}")
    print(f"  경계 노드 gated={stats['bnd_gated_ratio']:.4f}  비경계={stats['nonbnd_gated_ratio']:.4f}")
    print(f"  Gated 노드 acc={stats['gated_acc']:.4f}  Non-gated acc={stats['nongated_acc']:.4f}")
    print(f"  α(boundary)={stats['alpha']:.3f}  β(degree)={stats['beta']:.3f}")

    return stats


def write_plot_script(output_dir: str, datasets: list):
    code = f'''"""
fiw_gating_plot.py — FIW Gating 분포 시각화
실행: python {output_dir}/fiw_gating_plot.py
"""
import os, pandas as pd, matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(os.path.join("{output_dir}", "fiw_gating_stats.csv"))
DNAME = {{"pokec_z":"Pokec-Z","german":"German","nba":"NBA",
          "credit":"Credit","recidivism":"Recidivism",
          "pokec_n":"Pokec-N","income":"Income"}}
datasets = df["dataset"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# (1) Gated ratio by group
ax = axes[0]
x = np.arange(len(datasets))
w = 0.35
ax.bar(x - w/2, df["g0_gated_ratio"], w, label="Group 0", color="#2563EB", alpha=0.8)
ax.bar(x + w/2, df["g1_gated_ratio"], w, label="Group 1", color="#DC2626", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([DNAME.get(d,d) for d in datasets], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Gated node ratio")
ax.set_title("Gated ratio by sensitive group")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

# (2) Boundary vs non-boundary gated ratio
ax = axes[1]
ax.bar(x - w/2, df["bnd_gated_ratio"],    w, label="Boundary",     color="#059669", alpha=0.8)
ax.bar(x + w/2, df["nonbnd_gated_ratio"], w, label="Non-boundary", color="#9CA3AF", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([DNAME.get(d,d) for d in datasets], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Gated node ratio")
ax.set_title("Boundary vs Non-boundary gating")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

# (3) Gated vs non-gated accuracy
ax = axes[2]
ax.scatter(df["gated_acc"], df["nongated_acc"],
           s=80, color="#7C3AED", zorder=3)
for _, row in df.iterrows():
    ax.annotate(DNAME.get(row["dataset"], row["dataset"]),
                (row["gated_acc"], row["nongated_acc"]),
                textcoords="offset points", xytext=(4,3), fontsize=7)
lims = [min(df["gated_acc"].min(), df["nongated_acc"].min())-0.02,
        max(df["gated_acc"].max(), df["nongated_acc"].max())+0.02]
ax.plot(lims, lims, "--", color="#9CA3AF", linewidth=1)
ax.set_xlabel("Gated node accuracy")
ax.set_ylabel("Non-gated node accuracy")
ax.set_title("Prediction accuracy: gated vs non-gated")
ax.grid(True, alpha=0.3)

plt.suptitle("FairGate FIW Gating Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join("{output_dir}", "fiw_gating_plot.pdf")
plt.savefig(out, bbox_inches="tight", dpi=150)
print(f"[Saved] {{out}}")
plt.show()
'''
    path = os.path.join(output_dir, "fiw_gating_plot.py")
    with open(path, "w") as f:
        f.write(code)
    print(f"[Saved] {path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets",   nargs="+",
                   default=["pokec_z","pokec_z_g","pokec_n","pokec_n_g","german","credit","recidivism","nba","income"])
    p.add_argument("--output_dir", type=str, default="outputs/analysis")
    p.add_argument("--device",     type=str, default=DEVICE)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n[FIW Gating Analysis]  datasets={args.datasets}")
    all_stats = []
    for ds in args.datasets:
        print(f"\n  Training {ds}...")
        try:
            data, model = train_model(ds, args.device)
            stats = analyze_gating(ds, data, model, args.output_dir)
            all_stats.append(stats)
        except Exception as e:
            print(f"  [FAIL] {ds}: {e}")

    if all_stats:
        df = pd.DataFrame(all_stats)
        csv_path = os.path.join(args.output_dir, "fiw_gating_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")
        write_plot_script(args.output_dir, [s["dataset"] for s in all_stats])


if __name__ == "__main__":
    main()