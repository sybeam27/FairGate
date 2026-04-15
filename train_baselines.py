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
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

from utils.dataloading import load_data
from utils.data import get_dataset
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
from algorithms.FairGB   import FairGB
from algorithms.FairGT   import FairGT

# ── 메트릭 컬럼명 (train.py와 동일한 순서) ───────────────────────────────
METRIC_NAMES = [
    "acc", "roc_auc", "f1",
    "acc_sens0", "roc_auc_sens0", "f1_sens0",
    "acc_sens1", "roc_auc_sens1", "f1_sens1",
    "dp", "eo",
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


def to_device(adj, feats, labels, idx_train, idx_val, idx_test, sens, device):
    return (
        adj.to(device), feats.to(device), labels.to(device),
        idx_train.to(device), idx_val.to(device),
        idx_test.to(device), sens.to(device),
    )


def pack_result(values) -> dict:
    """모델 predict()의 11개 반환값 → 컬럼명 dict"""
    return {k: float(v) for k, v in zip(METRIC_NAMES, values)}


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

    # 열 순서: train.py와 동일
    priority = [
        "dataset", "task", "model",
        "acc_mean", "acc_std",
        "roc_auc_mean", "roc_auc_std",
        "f1_mean", "f1_std",
        "dp_mean", "dp_std",
        "eo_mean", "eo_std",
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


def _pyg_data_to_scipy_adj(data):
    edge_index = data.edge_index.cpu().numpy()
    n = data.x.size(0)
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32),
         (edge_index[0], edge_index[1])),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj


def _drop_feature_col(x: torch.Tensor, col_idx: int) -> torch.Tensor:
    return torch.cat([x[:, :col_idx], x[:, col_idx + 1:]], dim=1)


def _adjacency_positional_encoding(adj: sp.csr_matrix, pe_dim: int) -> torch.Tensor:
    n = adj.shape[0]
    k = min(pe_dim, max(1, n - 2))
    if pe_dim <= 0:
        return torch.zeros((n, 0), dtype=torch.float32)

    try:
        vals, vecs = eigsh(adj.astype(np.float32), k=k, which="LM")
        order = np.argsort(-np.abs(vals))
        vecs = vecs[:, order]
    except Exception:
        dense = adj.toarray().astype(np.float32)
        vals, vecs = np.linalg.eigh(dense)
        order = np.argsort(-np.abs(vals))[:k]
        vecs = vecs[:, order]

    return torch.from_numpy(vecs.astype(np.float32))


def _augment_same_sens_graph(
    adj: sp.csr_matrix,
    sens: np.ndarray,
    sample_k: int | None = None,
    seed: int = 0,
) -> sp.csr_matrix:
    """
    FairGT의 same-sensitive topology를 baseline runner에서 안전하게 흉내낸 버전.
    작은 그래프는 same-sens clique, 큰 그래프는 node당 sample_k개만 추가.
    """
    rng = np.random.default_rng(seed)
    n = adj.shape[0]

    rows, cols = [], []
    for g in np.unique(sens):
        idx = np.where(sens == g)[0]
        if len(idx) <= 1:
            continue

        if sample_k is None or len(idx) <= sample_k + 1:
            rr = np.repeat(idx, len(idx))
            cc = np.tile(idx, len(idx))
            mask = rr != cc
            rows.append(rr[mask])
            cols.append(cc[mask])
        else:
            for i in idx:
                peers = idx[idx != i]
                k = min(sample_k, len(peers))
                chosen = rng.choice(peers, size=k, replace=False)
                rows.append(np.full(k, i))
                cols.append(chosen)

    if len(rows) == 0:
        aug = adj.copy().tocsr()
        aug.setdiag(1.0)
        return aug

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    same_adj = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()

    aug = (adj + same_adj).astype(np.float32)
    aug.data[:] = 1.0
    aug = aug.maximum(aug.T)
    aug.setdiag(1.0)
    aug.eliminate_zeros()
    return aug


def _row_normalize_sparse(adj: sp.csr_matrix) -> sp.csr_matrix:
    rowsum = np.asarray(adj.sum(1)).flatten()
    rowsum[rowsum == 0] = 1.0
    inv = 1.0 / rowsum
    return sp.diags(inv).dot(adj).tocsr()


def _re_features(adj: sp.csr_matrix, features: torch.Tensor, hops: int) -> torch.Tensor:
    """
    [N, F] -> [N, hops+1, F]
    """
    adj_norm = _row_normalize_sparse(adj)
    x = features.cpu().numpy().astype(np.float32)

    outs = [torch.from_numpy(x)]
    cur = x
    for _ in range(hops):
        cur = adj_norm @ cur
        outs.append(torch.from_numpy(cur.astype(np.float32)))

    return torch.stack(outs, dim=1)


def _safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _group_fairness_binary(y_true, sens, y_pred):
    y_true = np.asarray(y_true)
    sens = np.asarray(sens)
    y_pred = np.asarray(y_pred)

    m0 = sens == 0
    m1 = sens == 1

    dp0 = y_pred[m0].mean() if m0.any() else 0.0
    dp1 = y_pred[m1].mean() if m1.any() else 0.0
    dp = abs(dp0 - dp1)

    e0 = ((y_true == 1) & m0)
    e1 = ((y_true == 1) & m1)
    eo0 = y_pred[e0].mean() if e0.any() else 0.0
    eo1 = y_pred[e1].mean() if e1.any() else 0.0
    eo = abs(eo0 - eo1)

    return float(dp), float(eo)


def _binary_metrics_dict(labels, sens, prob1, pred, idx):
    idx = idx.detach().cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)

    y = labels.detach().cpu().numpy()[idx]
    s = sens.detach().cpu().numpy()[idx]
    p = pred.detach().cpu().numpy()[idx]
    prob = prob1.detach().cpu().numpy()[idx]

    def _subset_metrics(mask):
        if mask.sum() == 0:
            return 0.0, 0.5, 0.0
        yy, pp, pr = y[mask], p[mask], prob[mask]
        acc = float(accuracy_score(yy, pp))
        auc = _safe_auc(yy, pr)
        f1 = float(f1_score(yy, pp, zero_division=0))
        return acc, auc, f1

    acc = float(accuracy_score(y, p))
    auc = _safe_auc(y, prob)
    f1 = float(f1_score(y, p, zero_division=0))

    acc0, auc0, f10 = _subset_metrics(s == 0)
    acc1, auc1, f11 = _subset_metrics(s == 1)

    dp, eo = _group_fairness_binary(y, s, p)

    return {
        "acc": acc,
        "roc_auc": auc,
        "f1": f1,
        "acc_sens0": acc0,
        "roc_auc_sens0": auc0,
        "f1_sens0": f10,
        "acc_sens1": acc1,
        "roc_auc_sens1": auc1,
        "f1_sens1": f11,
        "dp": dp,
        "eo": eo,
    }


# ============================================================
# 모델별 단일 run 실행
# ============================================================

def run_model(args, param1, param2, seed: int) -> dict:
    """
    단일 run 실행 → result dict 반환

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
        model = CrossWalk()
        model.fit(adj, feats, labels, idx_train, sens, device=device,
                  number_walks=param1, walk_length=param2, window_size=5)
        return pack_result(model.predict(idx_test))

    elif model_name == "FairWalk":
        # walk 기반 — 내부 embedding lr 고정, 외부 설정 미지원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset)
        model = FairWalk()
        model.fit(adj, labels, idx_train, sens, device=device,
                  num_walks=param1, walk_length=param2)
        return pack_result(model.predict(idx_test, idx_val))

    elif model_name == "FairVGNN":
        # fit()에 c_lr, e_lr, c_wd, e_wd, epochs 파라미터 직접 지원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=(dataset == "german"))
        model = FairVGNN()
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
        return pack_result(model.predict())

    elif model_name == "FairGNN":
        # __init__ 내부 argparse에서 lr, weight_decay, epochs 설정
        temp_accs = {
            "german": 0.66, "recidivism": 0.84, "credit": 0.60,
            "pokec_z": 0.6, "pokec_n": 0.56, "nba": 0.56,
        }
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=(dataset in ("nba", "german")))
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
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, idx_train,
                  device=device)
        return pack_result(model.predict(idx_test))

    elif model_name == "FairEdit":
        # fit()에 lr, weight_decay, epochs 직접 지원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=(dataset == "german"))
        model = FairEdit()
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                  lr=lr, weight_decay=wd, epochs=epochs,
                  hidden=args.hidden_dim, dropout=0.2, device=device)
        return pack_result(model.predict())

    elif model_name == "EDITS":
        # __init__에서 lr, weight_decay / fit()에 epochs / predict()에 lr, weight_decay
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=False)
        if dataset in ("credit", "german"):
            feats = feats / feats.norm(dim=0)
        ep = 100 if dataset == "german" else min(500, epochs)
        model = EDITS(feats, dropout=param1, lr=lr, weight_decay=wd)
        model.fit(adj, feats, sens, idx_train, idx_val,
                  half=False, device=device, epochs=ep)
        return pack_result(model.predict(
            adj, labels, sens, idx_train, idx_val, idx_test,
            epochs=ep, lr=lr, weight_decay=wd,
            threshold_proportion=param2))

    elif model_name == "NIFTY":
        # num_hidden=encoder 출력 차원, num_proj_hidden=SSL projection head 차원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=False)
        model = NIFTY(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                      num_hidden=args.hidden_dim,
                      num_proj_hidden=args.proj_hidden_dim,
                      lr=lr, weight_decay=wd, device=device)
        model.fit(epochs=epochs)
        return pack_result(model.predict())

    elif model_name == "GNN":
        # num_hidden=encoder 출력 차원, num_proj_hidden=SSL projection head 차원
        adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(
            dataset, feature_normalize=False)
        model = GNN(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                    num_hidden=args.hidden_dim,
                    num_proj_hidden=args.proj_hidden_dim,
                    lr=lr, weight_decay=wd, device=device)
        model.fit(epochs=epochs)
        return pack_result(model.predict())

    elif model_name == "FairGB":
        # fit()에 c_lr, e_lr, c_wd, e_wd, epochs, hidden 직접 지원
        data, sens_idx, x_min, x_max = get_dataset(dataset)
        data.sens_idx = sens_idx
        model = FairGB()
        model.fit(data, device=device, runs=1, seed=seed,
                  epochs=epochs, hidden=args.hidden_dim,
                  c_lr=lr, c_wd=wd,
                  e_lr=lr, e_wd=wd)
        return pack_result(model.predict())
    
    elif model_name == "FairGT":
        setup_seed(seed)

        data, sens_idx, _, _ = get_dataset(dataset)
        x = data.x.float().cpu()
        y = data.y.long().view(-1).cpu()
        sens = data.sens.long().view(-1).cpu()

        if args.fairgt_remove_sens_from_x and sens_idx is not None:
            x = _drop_feature_col(x, sens_idx)

        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_val   = data.val_mask.nonzero(as_tuple=False).view(-1)
        idx_test  = data.test_mask.nonzero(as_tuple=False).view(-1)

        # -------------------------------------------------
        # invalid label 제거: -1 같은 샘플은 FairGT에서 사용하지 않음
        # -------------------------------------------------
        valid_y = (y >= 0)

        idx_train = idx_train[valid_y[idx_train]]
        idx_val   = idx_val[valid_y[idx_val]]
        idx_test  = idx_test[valid_y[idx_test]]

        if idx_train.numel() == 0 or idx_val.numel() == 0 or idx_test.numel() == 0:
            raise ValueError(
                f"After filtering invalid labels, one of train/val/test is empty. "
                f"train={idx_train.numel()}, val={idx_val.numel()}, test={idx_test.numel()}"
            )

        # -------------------------------------------------
        # FairGT는 binary classification만 지원
        # 전체 y가 아니라 실제 train split 기준으로 검사
        # -------------------------------------------------
        train_unique = torch.unique(y[idx_train])
        if len(train_unique) != 2:
            raise ValueError(
                f"FairGT currently expects binary train labels, got {train_unique}"
            )

        # 혹시 라벨이 {1,2} 같이 들어와도 0..C-1로 재매핑
        used_idx = torch.cat([idx_train, idx_val, idx_test]).unique()
        used_labels = torch.unique(y[used_idx])
        label_map = {int(v): i for i, v in enumerate(sorted(used_labels.tolist()))}

        y_mapped = y.clone()
        for old_v, new_v in label_map.items():
            y_mapped[y == old_v] = new_v
        y = y_mapped

        adj = _pyg_data_to_scipy_adj(data)

        lpe = _adjacency_positional_encoding(adj, args.fairgt_pe_dim)
        feat_with_pe = torch.cat([x, lpe], dim=1)

        sample_k = None
        if data.num_nodes > args.fairgt_dense_threshold:
            sample_k = args.fairgt_same_group_k

        fair_adj = _augment_same_sens_graph(
            adj,
            sens.numpy(),
            sample_k=sample_k,
            seed=seed,
        )

        processed_features = _re_features(
            fair_adj,
            feat_with_pe,
            args.fairgt_hops,
        ).to(device)

        y = y.to(device)
        sens = sens.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

        net_params = {
            "hops": args.fairgt_hops,
            "pe_dim": args.fairgt_pe_dim,
            "in_dim": processed_features.shape[-1],
            "hidden_dim": args.hidden_dim,
            "n_heads": args.fairgt_n_heads,
            "n_layers": args.fairgt_n_layers,
            "nclass": 2,
            "dropout": args.dropout,
        }

        model = FairGT(net_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_score = -1e18
        best_result = None
        bad_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            logits = model(processed_features)
            loss = F.cross_entropy(logits[idx_train], y[idx_train])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(processed_features)
                prob1 = torch.softmax(logits, dim=1)[:, 1]
                pred = logits.argmax(dim=1)

                val_result = _binary_metrics_dict(y, sens, prob1, pred, idx_val)
                test_result = _binary_metrics_dict(y, sens, prob1, pred, idx_test)

                if args.fairgt_select == "acc":
                    score = val_result["acc"]
                elif args.fairgt_select == "acc_sp":
                    score = val_result["acc"] - val_result["dp"]
                elif args.fairgt_select == "acc_sp_eo":
                    score = val_result["acc"] - val_result["dp"] - val_result["eo"]
                elif args.fairgt_select == "auc_sp":
                    score = val_result["roc_auc"] - val_result["dp"]
                else:
                    raise ValueError(args.fairgt_select)

            if score > best_score:
                best_score = score
                best_result = test_result
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter >= args.patience:
                break

        return best_result
    
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
    g.add_argument("--epochs",       type=int,   default=1000)
    g.add_argument("--patience",     type=int,   default=100)
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
    g.add_argument("--save_dir",    type=str, default="outputs/",
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
    for run in range(args.runs):
        print(f"\n{'─'*65}")
        print(f"  Run {run + 1}/{args.runs}")
        print(f"{'─'*65}")

        seed = args.seed + run
        t0   = time.time()

        result         = run_model(args, param1, param2, seed)
        result["run"]  = run + 1
        result["time_sec"] = round(time.time() - t0, 1)

        print(f"  acc={result['acc']:.4f} | roc_auc={result['roc_auc']:.4f} | "
              f"f1={result['f1']:.4f} | dp={result['dp']:.4f} | "
              f"eo={result['eo']:.4f}  ({result['time_sec']:.1f}s)")
        all_results.append(result)

    save_summary(all_results, args)