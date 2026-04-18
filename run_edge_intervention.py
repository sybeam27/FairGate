"""
run_edge_intervention.py — Edge Intervention 방식 비교

drop  : inter-group 엣지를 struct_drop 비율로 제거 (기본값)
scale : inter-group 엣지 가중치를 감쇠 (bridge 보존)

그래프 구조(homophily, boundary_ratio)에 따라 어떤 방식이 유리한지 분석.

출력:
    outputs/exp_edge_drop.csv
    outputs/exp_edge_scale.csv
    outputs/analysis/edge_intervention_comparison.csv
    outputs/analysis/edge_intervention_comparison.tex
    outputs/analysis/edge_intervention_plot.py

실행:
    python run_edge_intervention.py
    python run_edge_intervention.py --datasets pokec_z german nba --dry_run
"""

import os, sys, argparse, subprocess
import numpy as np
import pandas as pd

DEVICE = "cuda:1"

ALL_DATASETS = [
    "pokec_z", "pokec_z_g", "pokec_n", "pokec_n_g",
    "german", "credit", "recidivism", "nba", "income",
]

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

FIXED = dict(
    backbone="GCN", hidden_dim=128, dropout=0.5, sgc_k=2,
    lr=1e-3, weight_decay=1e-5, epochs=500, patience=501,
    runs=5, seed=27, fips_lam=1.0, mmd_alpha=0.3,
    dp_eo_ratio=0.3, uncertainty_type="entropy",
    ramp_epochs=0, recal_interval=200, alpha_beta_mode="variance",
)

INTERVENTIONS = ["drop", "scale"]

# 그래프 구조 특성 (analyze_graph_stats 결과 기반)
GRAPH_STATS = {
    "pokec_z"   : dict(homophily=0.9532, boundary_ratio=0.3689, regime="clustered"),
    "pokec_z_g" : dict(homophily=0.4792, boundary_ratio=0.8889, regime="mixed"),
    "pokec_n"   : dict(homophily=0.9559, boundary_ratio=0.3070, regime="clustered"),
    "pokec_n_g" : dict(homophily=0.4889, boundary_ratio=0.8861, regime="mixed"),
    "german"    : dict(homophily=0.8092, boundary_ratio=0.9700, regime="saturated"),
    "credit"    : dict(homophily=0.9600, boundary_ratio=0.6768, regime="degree-skewed"),
    "recidivism": dict(homophily=0.5361, boundary_ratio=0.9983, regime="saturated"),
    "nba"       : dict(homophily=0.7288, boundary_ratio=0.9777, regime="saturated"),
    "income"    : dict(homophily=0.8844, boundary_ratio=0.3343, regime="clustered"),
}


def build_cmd(dataset: str, intervention: str, output_file: str) -> list:
    cfg = FAIRGATE_CONFIGS[dataset]
    run_name = f"edge_{intervention}_{dataset}"
    return [
        sys.executable, "-m", "utils.train",
        "--dataset",           dataset,
        "--backbone",          FIXED["backbone"],
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
        "--alpha_beta_mode",   FIXED["alpha_beta_mode"],
        "--edge_intervention", intervention,
    ]


def run_cmd(cmd, dry_run, log_path):
    if dry_run:
        print("    $ " + " ".join(cmd)); return True
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout)
        for ln in proc.stdout.strip().splitlines()[-4:]:
            print(f"      {ln}")
        if proc.returncode != 0:
            print(f"    [WARN] 실패 — {log_path}"); return False
    return True


def analyze(datasets: list, output_dir: str):
    DNAME = {"pokec_z":"Pokec-Z","pokec_z_g":"Pokec-Z (g)",
             "pokec_n":"Pokec-N","pokec_n_g":"Pokec-N (g)",
             "german":"German","credit":"Credit","recidivism":"Recidivism",
             "nba":"NBA","income":"Income"}

    dfs = {}
    for iv in INTERVENTIONS:
        fpath = f"outputs/exp_edge_{iv}.csv"
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df["edge_intervention"] = iv
            dfs[iv] = df

    if not dfs:
        print("[WARN] 결과 파일 없음"); return

    # 비교 테이블
    rows = []
    print(f"\n{'='*75}")
    print(f"{'Dataset':<20} {'Regime':<15} {'─drop─':>18} {'─scale─':>18} {'Winner'}")
    print(f"{'':20} {'':15} {'AUC':>6} {'ΔDP':>6} {'ΔEO':>6}  {'AUC':>6} {'ΔDP':>6} {'ΔEO':>6}")
    print(f"{'='*75}")

    for ds in datasets:
        gs = GRAPH_STATS.get(ds, {})
        regime = gs.get("regime","?")
        row = {"dataset": ds, "display": DNAME.get(ds,ds),
               "regime": regime,
               "homophily": gs.get("homophily", None),
               "boundary_ratio": gs.get("boundary_ratio", None)}
        vals = {}
        for iv in INTERVENTIONS:
            df = dfs.get(iv, pd.DataFrame())
            sub = df[df["dataset"]==ds] if not df.empty else pd.DataFrame()
            if sub.empty:
                vals[iv] = (float("nan"),)*3
            else:
                r = sub.iloc[0]
                vals[iv] = (r.get("roc_auc_mean",float("nan")),
                            r.get("dp_mean",float("nan")),
                            r.get("eo_mean",float("nan")))
            row[f"{iv}_auc"] = vals[iv][0]
            row[f"{iv}_dp"]  = vals[iv][1]
            row[f"{iv}_eo"]  = vals[iv][2]
            row[f"{iv}_fair"]= vals[iv][1] + vals[iv][2]

        # 승자 판정 (ΔDP+ΔEO 기준)
        f_drop  = vals["drop"][1]  + vals["drop"][2]
        f_scale = vals["scale"][1] + vals["scale"][2]
        winner  = "drop" if f_drop < f_scale else "scale"
        row["winner"] = winner

        d_str = f"{vals['drop'][0]:.4f} {vals['drop'][1]:.4f} {vals['drop'][2]:.4f}"
        s_str = f"{vals['scale'][0]:.4f} {vals['scale'][1]:.4f} {vals['scale'][2]:.4f}"
        print(f"{DNAME.get(ds,ds):<20} {regime:<15} {d_str:>18}  {s_str:>18}  {winner}")
        rows.append(row)

    print(f"{'='*75}")

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "edge_intervention_comparison.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # 패턴 분석: regime별 승자
    print("\n[Regime별 drop vs scale 비교]")
    for regime, grp in df_out.groupby("regime"):
        winners = grp["winner"].value_counts().to_dict()
        print(f"  {regime:<15}: {winners}")

    # 상관관계: homophily vs (scale_fair - drop_fair)
    if "homophily" in df_out.columns and df_out["homophily"].notna().any():
        df_out["scale_advantage"] = df_out["drop_fair"] - df_out["scale_fair"]
        corr = df_out["homophily"].corr(df_out["scale_advantage"])
        bnd_corr = df_out["boundary_ratio"].corr(df_out["scale_advantage"])
        print(f"\n[Scale 방식 이점과 그래프 특성 상관관계]")
        print(f"  homophily vs scale_advantage    : r = {corr:+.4f}")
        print(f"  boundary_ratio vs scale_advantage: r = {bnd_corr:+.4f}")
        if corr > 0.2:
            print("  → 동질성 높은 그래프에서 scale 방식이 유리한 경향")
        elif corr < -0.2:
            print("  → 동질성 낮은 그래프에서 scale 방식이 유리한 경향")

    # LaTeX 생성
    to_latex(df_out, output_dir)


def to_latex(df: pd.DataFrame, output_dir: str):
    lines = [r"\begin{table}[t]", r"\centering",
        r"\caption{엣지 개입 방식(\texttt{drop} vs \texttt{scale}) 비교. "
        r"\texttt{drop}: inter-group 엣지 제거, "
        r"\texttt{scale}: 엣지 가중치 감쇠(bridge 보존). "
        r"각 데이터셋에서 $\Delta\mathrm{DP}+\Delta\mathrm{EO}$가 낮은 방식을 \textbf{굵게} 표시.}",
        r"\label{tab:edge_intervention}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{llrrrrrrrl}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{Regime} & "
        r"\multicolumn{3}{c}{\texttt{drop}} & \multicolumn{3}{c}{\texttt{scale}} & \multirow{2}{*}{Winner} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r" & & AUC$\uparrow$ & $\Delta$DP$\downarrow$ & $\Delta$EO$\downarrow$ "
        r"& AUC$\uparrow$ & $\Delta$DP$\downarrow$ & $\Delta$EO$\downarrow$ & \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        winner = row.get("winner","—")
        # 수치
        def fmt(v, met, winner_iv):
            s = f"{v:.4f}" if not pd.isna(v) else "—"
            return r"\textbf{" + s + "}" if winner_iv == winner else s

        d_auc = fmt(row.get("drop_auc",float("nan")),  "auc",  "drop")
        d_dp  = fmt(row.get("drop_dp", float("nan")),  "dp",   "drop")
        d_eo  = fmt(row.get("drop_eo", float("nan")),  "eo",   "drop")
        s_auc = fmt(row.get("scale_auc",float("nan")), "auc",  "scale")
        s_dp  = fmt(row.get("scale_dp", float("nan")), "dp",   "scale")
        s_eo  = fmt(row.get("scale_eo", float("nan")), "eo",   "scale")

        lines.append(
            f"{row['display']} & {row['regime']} & "
            f"{d_auc} & {d_dp} & {d_eo} & "
            f"{s_auc} & {s_dp} & {s_eo} & {winner} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    tex_path = os.path.join(output_dir, "edge_intervention_comparison.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Saved] {tex_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--datasets",    nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--interventions",nargs="+", default=INTERVENTIONS, choices=INTERVENTIONS)
    p.add_argument("--output_dir",  type=str,  default="outputs/analysis")
    p.add_argument("--log_dir",     type=str,  default="logs/edge_intervention")
    p.add_argument("--dry_run",     action="store_true")
    p.add_argument("--analyze_only",action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    total = len(args.datasets) * len(args.interventions)

    print(f"\n{'='*65}")
    print(f"Edge Intervention Comparison")
    print(f"  interventions : {args.interventions}")
    print(f"  datasets      : {args.datasets}")
    print(f"  total         : {total} runs")
    print(f"{'='*65}")

    if not args.analyze_only:
        step = 0
        for iv in args.interventions:
            output_file = f"exp_edge/{iv}.csv"
            print(f"\n{'─'*65}")
            print(f"  [edge_intervention: {iv}]  →  {output_file}")
            print(f"{'─'*65}")
            for ds in args.datasets:
                step += 1
                cmd = build_cmd(ds, iv, output_file)
                log_path = os.path.join(args.log_dir, iv, f"{ds}.log")
                print(f"  [{step:2d}/{total}] {ds:<14}  edge={iv}")
                run_cmd(cmd, args.dry_run, log_path)

    if not args.dry_run:
        analyze(args.datasets, args.output_dir)

    print(f"\n완료.")


if __name__ == "__main__":
    main()
