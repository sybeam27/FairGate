"""
train_baselines.py — 비교 모델 실험 실행기

train.py(FairGate)와 동일한 학습 설정 인자 및 save_summary 구조를 사용.

Usage:
    python train_baselines.py --model FairGNN --dataset credit
    python train_baselines.py --model FairWalk --dataset pokec_z --runs 5 --seed 27
    python train_baselines.py --model NIFTY --dataset german --run_name exp_0412_v2

공통 인자 (train.py와 동일):
    --lr, --weight_decay, --epochs, --patience, --seed, --runs
    --run_name, --output_file, --save_dir
"""

import os
import json
import time
import random
import argparse
import tracemalloc
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from .dataloading import load_data
from .data import get_dataset
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from algorithms.GNN      import GNN
from algorithms.FairGNN  import FairGNN
from algorithms.FairVGNN import FairVGNN
from algorithms.FairWalk import FairWalk
from algorithms.CrossWalk import CrossWalk
from algorithms.EDITS    import EDITS
from algorithms.FairEdit import FairEdit
from algorithms.NIFTY    import NIFTY
from algorithms.FairGB_alg   import FairGB
from algorithms.FairGT_alg   import FairGT

# ── 성능 메트릭 컬럼명 (train.py와 동일한 순서) ─────────────────────────
METRIC_NAMES = [
    "acc", "roc_auc", "f1",
    "acc_sens0", "roc_auc_sens0", "f1_sens0",
    "acc_sens1", "roc_auc_sens1", "f1_sens1",
    "dp", "eo",
]

# ── Scalability 분석 메트릭 컬럼명 ──────────────────────────────────────
SCALE_METRIC_NAMES = [
    "n_nodes",            # 그래프 노드 수
    "n_edges",            # 그래프 엣지 수 (nnz of adj)
    "avg_degree",         # 평균 차수
    "n_params",           # 학습 가능한 파라미터 수
    "peak_mem_mb",        # 학습 중 최대 메모리 사용량 (MB)
    "epochs_run",         # 실제 수행된 에폭 수 (early stopping 반영)
    "time_per_epoch_ms",  # 에폭당 평균 학습 시간 (ms)
]


# ============================================================
# Utilities
# ============================================================

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# Scalability 측정 유틸리티
# ============================================================

def count_params(model) -> int:
    """학습 가능한 파라미터 수 반환. parameters() 미지원 모델은 0."""
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except (AttributeError, TypeError):
        return 0


def get_graph_stats_adj(adj) -> dict:
    """
    adj → 그래프 통계 dict 반환.
    scipy sparse matrix, torch sparse tensor, dense torch tensor 모두 지원.
    """
    import torch as _torch
    n_nodes = int(adj.shape[0])

    if hasattr(adj, "nnz"):                        # scipy sparse
        n_edges = int(adj.nnz)
    elif isinstance(adj, _torch.Tensor):
        if adj.is_sparse:                          # torch sparse tensor
            n_edges = int(adj._nnz())
        else:                                      # dense torch tensor
            n_edges = int((adj != 0).sum().item())
    else:
        n_edges = 0                                # fallback

    avg_deg = round(n_edges / n_nodes, 2) if n_nodes > 0 else 0.0
    return {"n_nodes": n_nodes, "n_edges": n_edges, "avg_degree": avg_deg}


def get_graph_stats_pyg(data) -> dict:
    """PyG Data 객체 → 그래프 통계 dict 반환."""
    n_nodes  = int(data.x.size(0))
    n_edges  = int(data.edge_index.size(1))
    avg_deg  = round(n_edges / n_nodes, 2) if n_nodes > 0 else 0.0
    return {"n_nodes": n_nodes, "n_edges": n_edges, "avg_degree": avg_deg}


def _infer_epochs_run(model, fallback: int) -> int:
    """
    모델 속성에서 실제 수행된 에폭 수를 추론.
    early stopping이 있는 모델은 best_epoch / epochs_trained 속성을 통해 확인.
    없으면 fallback(=args.epochs) 사용.
    """
    for attr in ("epochs_trained_", "best_epoch_", "best_epoch",
                 "epochs_trained", "n_iter_", "num_epochs"):
        val = getattr(model, attr, None)
        if val is not None and isinstance(val, (int, float)) and val > 0:
            return int(val)
    return fallback


class MemTracker:
    """
    GPU(CUDA) 또는 CPU 메모리 피크 측정 컨텍스트.

    Usage::
        tracker = MemTracker(device)
        tracker.start()
        ... heavy computation ...
        peak_mb = tracker.stop()
    """
    def __init__(self, device: str):
        self._use_cuda = "cuda" in str(device) and torch.cuda.is_available()

    def start(self):
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats()
        else:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            tracemalloc.start()

    def stop(self) -> float:
        if self._use_cuda:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak = peak_bytes / 1024 ** 2
        return round(peak, 1)


def to_device(adj, feats, labels, idx_train, idx_val, idx_test, sens, device):
    return (
        adj.to(device), feats.to(device), labels.to(device),
        idx_train.to(device), idx_val.to(device),
        idx_test.to(device), sens.to(device),
    )


def pack_result(values) -> dict:
    """모델 predict()의 11개 반환값 → 컬럼명 dict"""
    return {k: float(v) for k, v in zip(METRIC_NAMES, values)}


def pack_scale(graph_stats: dict, model, epochs_run: int, fit_sec: float) -> dict:
    """
    Scalability 지표를 하나의 dict로 묶어 반환.

    Args:
        graph_stats : get_graph_stats_adj / get_graph_stats_pyg 반환값
        model       : fit() 완료된 모델 객체 (파라미터 수 계산용)
        epochs_run  : 실제 수행된 에폭 수
        fit_sec     : fit() 소요 시간(초, time_sec 전체가 아닌 fit만)
    """
    n_params = count_params(model)
    time_per_epoch_ms = round(fit_sec * 1000 / epochs_run, 2) if epochs_run > 0 else float("nan")
    return {
        **graph_stats,
        "n_params":           n_params,
        "epochs_run":         epochs_run,
        "time_per_epoch_ms":  time_per_epoch_ms,
        # peak_mem_mb는 MemTracker가 외부에서 채워 넣음
    }


def _resolve_save_path(args) -> str:
    """
    train.py와 동일한 파일명 결정 로직.
    우선순위: --output_file > --run_name > 실행 시각(YYYYMMDD_HHMM)
    """
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.output_file:
        path = args.output_file
        if not path.endswith(".csv"):
            path += ".csv"
        if not os.path.isabs(path):
            path = os.path.join(save_dir, path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return path

    stem = args.run_name if args.run_name else datetime.now().strftime("%Y%m%d_%H%M")
    return os.path.join(save_dir, f"{stem}.csv")


def save_summary(all_results: list, args: argparse.Namespace):
    """
    train.py와 동일한 구조로 mean/std 요약 후 CSV 누적 저장.
    - dataset 열이 항상 포함
    - 동일 (dataset, model) 조합은 기존 행을 교체
    - 파일명: --output_file > --run_name > 시각
    """
    save_path = _resolve_save_path(args)

    df_runs  = pd.DataFrame(all_results)
    num_cols = [c for c in df_runs.select_dtypes("number").columns
                if c != "run"]

    row = {
        "dataset": args.dataset,
        "task":    "classification",
        "model":   args.model,
        # 학습 설정 (train.py와 동일 열 구조)
        "hidden_dim":       args.hidden_dim,
        "proj_hidden_dim":  args.proj_hidden_dim,
        "runs":             args.runs,
        "seed":         args.seed,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "epochs":       args.epochs,
        "patience":     args.patience,
    }
    for col in num_cols:
        row[f"{col}_mean"] = round(float(df_runs[col].mean()), 4)
        row[f"{col}_std"]  = round(float(df_runs[col].std()),  4) if len(df_runs) > 1 else 0.0

    summary = pd.DataFrame([row])

    # 열 순서: train.py와 동일 + scalability 항목
    priority = [
        "dataset", "task", "model",
        "acc_mean", "acc_std",
        "roc_auc_mean", "roc_auc_std",
        "f1_mean", "f1_std",
        "dp_mean", "dp_std",
        "eo_mean", "eo_std",
        "time_sec_mean", "time_sec_std",
        # ── Scalability 지표 ─────────────────────────
        "n_nodes_mean",
        "n_edges_mean",
        "avg_degree_mean",
        "n_params_mean",
        "peak_mem_mb_mean", "peak_mem_mb_std",
        "epochs_run_mean",  "epochs_run_std",
        "time_per_epoch_ms_mean", "time_per_epoch_ms_std",
    ]

    dedup_keys = ["dataset", "task", "model"]

    if os.path.exists(save_path):
        existing  = pd.read_csv(save_path)
        key_cols  = [c for c in dedup_keys
                     if c in existing.columns and c in summary.columns]
        merged    = existing.merge(summary[key_cols], on=key_cols,
                                   how="left", indicator=True)
        existing  = (existing[merged["_merge"] == "left_only"]
                     .drop(columns=["_merge"], errors="ignore"))
        final = pd.concat([existing, summary], ignore_index=True)
    else:
        final = summary

    ordered = [c for c in priority if c in final.columns]
    rest    = [c for c in final.columns if c not in ordered]
    final   = final[ordered + rest]

    numeric = final.select_dtypes("number").columns
    final[numeric] = final[numeric].round(4)
    final.to_csv(save_path, index=False)

    print(f"\n[Save] {save_path}  "
          f"({len(final)} rows, {final['dataset'].nunique()} dataset(s))")
    print(f"\n  [{args.model}] Summary — {args.dataset}")
    for col in ["acc_mean", "acc_std",
                "roc_auc_mean", "roc_auc_std",
                "f1_mean", "f1_std",
                "dp_mean", "dp_std",
                "eo_mean", "eo_std"]:
        if col in summary.columns:
            print(f"    {col}: {summary[col].values[0]:.4f}")
    if "time_sec_mean" in summary.columns:
        print(f"    time_sec: {summary['time_sec_mean'].values[0]:.1f} ± {summary['time_sec_std'].values[0]:.1f}s")
    # ── Scalability 요약 출력 ─────────────────────────────────────────────
    scale_print = [
        ("n_nodes",           "nodes",         ".0f"),
        ("n_edges",           "edges",         ".0f"),
        ("avg_degree",        "avg_degree",    ".2f"),
        ("n_params",          "n_params",      ".0f"),
        ("peak_mem_mb",       "peak_mem(MB)",  ".1f"),
        ("epochs_run",        "epochs_run",    ".1f"),
        ("time_per_epoch_ms", "ms/epoch",      ".2f"),
    ]
    for key, label, fmt in scale_print:
        mu_col = f"{key}_mean"
        if mu_col in summary.columns and not pd.isna(summary[mu_col].values[0]):
            val = summary[mu_col].values[0]
            print(f"    {label}: {val:{fmt}}")


# ============================================================
# 모델별 단일 run 실행
# ============================================================

def run_model(args, param1, param2, seed: int) -> dict:
    """
    단일 run 실행 → 성능 메트릭 + scalability 메트릭 dict 반환

    반환 dict 추가 키:
        n_nodes, n_edges, avg_degree  : 그래프 규모
        n_params                      : 학습 가능한 파라미터 수
        epochs_run                    : 실제 수행된 에폭 수
        time_per_epoch_ms             : 에폭당 평균 학습 시간(ms)
        (peak_mem_mb 는 외부 MemTracker가 채워 넣음)

    공통 학습 설정 적용 현황:
        GNN       : __init__에서 lr, weight_decay 직접 전달 / fit(epochs)
        FairGNN   : __init__ argparse에서 lr, weight_decay, epochs 설정
        FairVGNN  : fit()에 c_lr, e_lr(=lr), c_wd, e_wd(=weight_decay), epochs 전달
        NIFTY     : __init__에서 lr, weight_decay 직접 전달 / fit(epochs)
        FairEdit  : fit()에 lr, weight_decay, epochs 직접 전달
        EDITS     : fit()에 lr, epochs / predict()에 lr, weight_decay 전달
        FairGB    : fit()에 c_lr, e_lr(=lr), c_wd, e_wd(=weight_decay), epochs 전달
        CrossWalk : 내부 embedding optimizer만 사용, lr 미지원 (walk 기반 알고리즘)
        FairWalk  : 내부 embedding optimizer만 사용, lr 미지원 (walk 기반 알고리즘)
    """
    setup_seed(seed)
    model_name = args.model
    dataset    = args.dataset
    device     = args.device
    lr         = args.lr
    wd         = args.weight_decay
    epochs     = args.epochs

    if model_name == "CrossWalk":
        # walk 기반 — 내부 embedding lr 고정, 외부 설정 미지원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=(dataset in ("nba", "german")))
        graph_stats = get_graph_stats_adj(adj)
        model = CrossWalk()
        t_fit = time.time()
        model.fit(adj, feats, labels, idx_train, sens, device=device,
                  number_walks=param1, walk_length=param2, window_size=5)
        fit_sec = time.time() - t_fit
        # walk 기반: epoch 개념 없음 → param1(num_walks) 근사치 사용
        epochs_run = _infer_epochs_run(model, fallback=int(param1))
        return {**pack_result(model.predict(idx_test)),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "FairWalk":
        # walk 기반 — 내부 embedding lr 고정, 외부 설정 미지원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset)
        graph_stats = get_graph_stats_adj(adj)
        model = FairWalk()
        t_fit = time.time()
        model.fit(adj, labels, idx_train, sens, device=device,
                  num_walks=param1, walk_length=param2)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=int(param1))
        return {**pack_result(model.predict(idx_test, idx_val)),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "FairVGNN":
        # fit()에 c_lr, e_lr, c_wd, e_wd, epochs 파라미터 직접 지원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=(dataset == "german"))
        graph_stats = get_graph_stats_adj(adj)
        model = FairVGNN()
        t_fit = time.time()
        if dataset == "recidivism":
            model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                      device=device, runs=1, seed=seed,
                      c_lr=lr, e_lr=lr, c_wd=wd, e_wd=wd,
                      hidden=args.hidden_dim,
                      top_k=param1, alpha=param2, clip_e=1,
                      g_epochs=10, c_epochs=10, ratio=1,
                      epochs=min(300, epochs))
        elif dataset == "german":
            model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                      device=device, runs=1, seed=seed,
                      c_lr=lr, e_lr=lr * 0.1, c_wd=wd, e_wd=wd,
                      hidden=args.hidden_dim,
                      top_k=param1, clip_e=0.1, d_epochs=5, c_epochs=10,
                      ratio=0, alpha=param2, prop="scatter",
                      epochs=min(600, epochs))
        elif dataset == "credit":
            model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                      device=device, runs=1, seed=seed,
                      c_lr=lr, e_lr=lr, c_wd=wd, e_wd=wd,
                      hidden=args.hidden_dim,
                      top_k=param1, alpha=param2, clip_e=1,
                      g_epochs=10, c_epochs=5, ratio=0,
                      epochs=min(200, epochs))
        else:
            model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                      device=device, runs=1, seed=seed,
                      c_lr=lr, e_lr=lr, c_wd=wd, e_wd=wd,
                      hidden=args.hidden_dim,
                      top_k=param1, alpha=param2,
                      epochs=epochs)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=epochs)
        return {**pack_result(model.predict()),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "FairGNN":
        # __init__ 내부 argparse에서 lr, weight_decay, epochs 설정
        temp_accs = {
            "german": 0.66, "recidivism": 0.84, "credit": 0.60,
            "pokec_z": 0.6, "pokec_n": 0.56, "nba": 0.56,
        }
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=(dataset in ("nba", "german")))
        graph_stats = get_graph_stats_adj(adj)
        adj, feats, labels, idx_train, idx_val, idx_test, sens = to_device(
            adj, feats, labels, idx_train, idx_val, idx_test, sens, device)
        temp_acc = temp_accs.get(dataset, 0.6) - 0.3
        model = FairGNN(feats.shape[-1], acc=temp_acc,
                        epoch=epochs,
                        alpha=param1, beta=param2).to(device)
        # FairGNN은 __init__ argparse로 lr/wd/hidden이 설정됨
        # optimizer 재생성으로 lr/wd 덮어쓰고, hidden_dim도 덮어씀
        model.args.num_hidden = args.hidden_dim
        import itertools
        G_params = list(itertools.chain(
            model.GNN.parameters(),
            model.classifier.parameters(),
            model.estimator.parameters(),
        ))
        model.optimizer_G = torch.optim.Adam(G_params, lr=lr, weight_decay=wd)
        model.optimizer_A = torch.optim.Adam(model.adv.parameters(), lr=lr, weight_decay=wd)
        t_fit = time.time()
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, idx_train,
                  device=device)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=epochs)
        return {**pack_result(model.predict(idx_test)),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "FairEdit":
        # fit()에 lr, weight_decay, epochs 직접 지원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=(dataset == "german"))
        graph_stats = get_graph_stats_adj(adj)
        model = FairEdit()
        t_fit = time.time()
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                  lr=lr, weight_decay=wd, epochs=epochs,
                  hidden=args.hidden_dim, dropout=0.2, device=device)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=epochs)
        return {**pack_result(model.predict()),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "EDITS":
        # __init__에서 lr, weight_decay / fit()에 epochs / predict()에 lr, weight_decay
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=False)
        graph_stats = get_graph_stats_adj(adj)
        if dataset in ("credit", "german"):
            feats = feats / feats.norm(dim=0)
        ep = 100 if dataset == "german" else min(500, epochs)
        model = EDITS(feats, dropout=param1, lr=lr, weight_decay=wd)
        t_fit = time.time()
        model.fit(adj, feats, sens, idx_train, idx_val,
                  half=False, device=device, epochs=ep)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=ep)
        result = pack_result(model.predict(
            adj, labels, sens, idx_train, idx_val, idx_test,
            epochs=ep, lr=lr, weight_decay=wd,
            threshold_proportion=param2))
        return {**result, **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "NIFTY":
        # num_hidden=encoder 출력 차원, num_proj_hidden=SSL projection head 차원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=False)
        graph_stats = get_graph_stats_adj(adj)
        model = NIFTY(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                      num_hidden=args.hidden_dim,
                      num_proj_hidden=args.proj_hidden_dim,
                      lr=lr, weight_decay=wd, device=device)
        t_fit = time.time()
        model.fit(epochs=epochs)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=epochs)
        return {**pack_result(model.predict()),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "GNN":
        # num_hidden=encoder 출력 차원, num_proj_hidden=SSL projection head 차원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=False)
        graph_stats = get_graph_stats_adj(adj)
        model = GNN(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                    num_hidden=args.hidden_dim,
                    num_proj_hidden=args.proj_hidden_dim,
                    lr=lr, weight_decay=wd, device=device)
        t_fit = time.time()
        model.fit(epochs=epochs)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=epochs)
        return {**pack_result(model.predict()),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "FairGB":
        # fit()에 c_lr, e_lr, c_wd, e_wd, epochs, hidden 직접 지원
        data, sens_idx, x_min, x_max = get_dataset(dataset)
        graph_stats = get_graph_stats_pyg(data)
        data.sens_idx = sens_idx
        model = FairGB()
        t_fit = time.time()
        model.fit(data, device=device, runs=1, seed=seed,
                  epochs=epochs, hidden=args.hidden_dim,
                  c_lr=lr, c_wd=wd,
                  e_lr=lr, e_wd=wd)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=epochs)
        return {**pack_result(model.predict()),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    elif model_name == "FairGT":
        data, sens_idx, _, _ = get_dataset(dataset)
        graph_stats = get_graph_stats_pyg(data)
        model = FairGT(data, sens_idx, args, lr=lr, weight_decay=wd, device=device)
        t_fit = time.time()
        model.fit(epochs=epochs, patience=args.patience)
        fit_sec = time.time() - t_fit
        epochs_run = _infer_epochs_run(model, fallback=epochs)
        return {**pack_result(model.predict()),
                **pack_scale(graph_stats, model, epochs_run, fit_sec)}

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline fair GNN experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 필수
    g = parser.add_argument_group("Required")
    g.add_argument("--model",   type=str, required=True,
                   choices=["GNN", "FairGNN", "FairVGNN",
                            "FairWalk", "CrossWalk",
                            "EDITS", "FairEdit", "NIFTY", "FairGB", "FairGT"])
    g.add_argument("--dataset", type=str, required=True)

    # ── 학습 설정 (train.py와 완전히 동일한 인자명/기본값)
    g = parser.add_argument_group("Training  [identical to train.py]")
    g.add_argument("--lr",           type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--epochs",   type=int, default=500)  # 수정
    g.add_argument("--patience", type=int, default=501)  # 수정
    g.add_argument("--seed",         type=int,   default=27)
    g.add_argument("--runs",         type=int,   default=5)
    g.add_argument("--device",       type=str,   default="cuda")
    g.add_argument("--hidden_dim",     type=int, default=128,
                   help="hidden layer dimension (train.py와 동일)")
    g.add_argument("--proj_hidden_dim", type=int, default=128,
                   help="projection head dimension for GNN/NIFTY (SSL). "
                        "param.json의 num_proj_hidden 값이 우선 적용됨.")

    # ── 저장 설정 (train.py와 완전히 동일한 인자명/로직)
    g = parser.add_argument_group("Output  [identical to train.py]")
    g.add_argument("--save_dir",    type=str, default="outputs/compare/",
                   help="결과 파일이 저장될 디렉토리")
    g.add_argument("--run_name",    type=str, default=None,
                   help="결과 파일 이름 (확장자 제외). 예: exp_0412_v2")
    g.add_argument("--output_file", type=str, default=None,
                   help="결과 파일 전체 경로. --run_name보다 우선순위 높음.")

    # ── 모델 파라미터
    g = parser.add_argument_group("Model params")
    g.add_argument("--param_path", type=str, default="./utils/param.json",
                   help="모델별 최적 파라미터 JSON 파일")
    # FairGT 전용
    g.add_argument("--fairgt_hops", type=int, default=2)
    g.add_argument("--fairgt_pe_dim", type=int, default=2)
    g.add_argument("--fairgt_n_heads", type=int, default=2)
    g.add_argument("--fairgt_n_layers", type=int, default=1)
    g.add_argument("--fairgt_same_group_k", type=int, default=64,
                help="large graph에서 node당 추가할 same-sens sampled neighbors 수")
    g.add_argument("--fairgt_dense_threshold", type=int, default=5000,
                help="이보다 큰 그래프는 same-sens clique 대신 sampled edges 사용")
    g.add_argument("--fairgt_remove_sens_from_x", action="store_true",
                help="sens feature column을 입력 x에서 제거")
    g.add_argument("--fairgt_select", type=str, default="acc_sp",
                choices=["acc", "acc_sp", "acc_sp_eo", "auc_sp"])

    # 내부 모델들이 자체 argparse를 쓰는 경우 충돌 방지
    args, _ = parser.parse_known_args()
    return args


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    # ── param.json 로드 ─────────────────────────────────────────
    # param.json의 모든 값을 우선 적용.
    # 없는 값만 args 기본값(run.py COMMON_TRAIN) 사용.
    # runs는 항상 args.runs(커맨드라인) 고정.
    #
    # param.json 키 → args 속성 매핑:
    #   num_hidden, num_proj_hidden → args.hidden_dim (GNN/NIFTY)
    #   hidden                     → args.hidden_dim (FairVGNN/FairEdit/FairGB)
    #   lr, weight_decay, epochs, patience, seed, dropout → 동명 args 속성
    #   나머지 키 → param1, param2 (모델 고유 파라미터)

    # args 속성명으로 매핑되는 param.json 키
    PARAM_TO_ARGS = {
        "lr":               "lr",
        "weight_decay":     "weight_decay",
        "epochs":           "epochs",
        "patience":         "patience",
        "seed":             "seed",
        "dropout":          "dropout",
        "hidden_dim":       "hidden_dim",
        "hidden":           "hidden_dim",       # FairVGNN/FairEdit/FairGB
        "num_hidden":       "hidden_dim",       # GNN/NIFTY encoder 출력 차원
        "num_proj_hidden":  "proj_hidden_dim",  # GNN/NIFTY projection head 차원 (별도)
    }

    param1, param2 = 1, 1
    pval = {}

    if os.path.exists(args.param_path):
        optimal = json.load(open(args.param_path))
        pkey    = args.dataset if args.dataset != "bail" else "recidivism"
        raw     = optimal.get(args.model, {}).get(pkey, {})

        if isinstance(raw, dict):
            pval = raw
            model_params = []   # 모델 고유 파라미터 (args에 매핑 안 되는 것들)

            for k, v in pval.items():
                if k in PARAM_TO_ARGS:
                    attr = PARAM_TO_ARGS[k]
                    setattr(args, attr, v)
                    print(f"[param.json] {args.model}/{pkey}: {k} → args.{attr}={v}")
                else:
                    model_params.append(v)

            if len(model_params) >= 2:
                param1, param2 = model_params[0], model_params[1]
            elif len(model_params) == 1:
                param1 = model_params[0]

            if not model_params:
                print(f"[param.json] {args.model}/{pkey}: "
                      f"no model-specific params (all mapped to args)")
        else:
            print(f"[WARN] No valid params for {args.model}/{pkey}, "
                  f"using defaults")
    else:
        print(f"[WARN] param_path not found: {args.param_path}, "
              f"using defaults")

    print(f"\n{'='*65}")
    print(f"  Model  : {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  lr={args.lr}  wd={args.weight_decay}  "
          f"epochs={args.epochs}  patience={args.patience}")
    print(f"  Runs: {args.runs}  |  Seed base: {args.seed}")
    print(f"{'='*65}\n")

    all_results = []
    mem_tracker = MemTracker(args.device)
    for run in range(args.runs):
        print(f"\n{'─'*65}")
        print(f"  Run {run + 1}/{args.runs}")
        print(f"{'─'*65}")

        seed = args.seed + run
        t0   = time.time()

        mem_tracker.start()
        result         = run_model(args, param1, param2, seed)
        peak_mem       = mem_tracker.stop()

        result["run"]        = run + 1
        result["time_sec"]   = round(time.time() - t0, 1)
        result["peak_mem_mb"] = peak_mem  # MemTracker가 외부에서 채워 넣음

        print(f"  acc={result['acc']:.4f} | roc_auc={result['roc_auc']:.4f} | "
              f"f1={result['f1']:.4f} | dp={result['dp']:.4f} | "
              f"eo={result['eo']:.4f}  ({result['time_sec']:.1f}s)")
        print(f"  [scale] nodes={result.get('n_nodes','?')} | "
              f"edges={result.get('n_edges','?')} | "
              f"params={result.get('n_params','?'):,} | "
              f"mem={peak_mem:.1f}MB | "
              f"epochs={result.get('epochs_run','?')} | "
              f"{result.get('time_per_epoch_ms','?'):.2f}ms/ep")
        all_results.append(result)

    save_summary(all_results, args)