"""
algorithms/FairGT.py

FairGT baseline wrapper — 원본 LuoRenqiang/FairGT의 로직을 그대로 사용.
train_baselines.py의 fit() / predict() 인터페이스에 맞게 감쌈.

Usage (train_baselines.py):
    data, sens_idx, _, _ = get_dataset(dataset)
    model = FairGT(data, sens_idx, args, lr=lr, weight_decay=wd, device=device)
    model.fit(epochs=epochs)
    return pack_result(model.predict())
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from FairGT.model import FairGT as _FairGTModel


# ============================================================
# 원본 utils.py에서 가져온 핵심 함수들 (변경 없이 이식)
# ============================================================

def _adjacency_positional_encoding(adj, pe_dim):
    """원본: adjacency_positional_encoding — scipy sparse adj 입력"""
    eignvalue, eignvector = eigsh(adj, which='LM', k=pe_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector


def _re_features(adj, features, K):
    """
    원본 re_features — dense adj(torch.Tensor)로 matmul.
    adj: [N, N] dense torch.Tensor (행 정규화된 것)
    features: [N, F] torch.Tensor
    반환: [N, K+1, F]
    """
    if K == 0:
        return features.unsqueeze(1)  # [N, 1, F]

    nodes_features = torch.empty(features.shape[0], 1, K + 1, features.shape[1])
    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)
    for i in range(K):
        x = torch.matmul(adj, x)
        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]

    nodes_features = nodes_features.squeeze(1)  # [N, K+1, F]
    return nodes_features


def _feature_normalize(feature):
    """원본 feature_normalize — row sum normalize"""
    feature = np.array(feature)
    rowsum = feature.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return feature / rowsum


def _pyg_data_to_scipy_adj(data):
    edge_index = data.edge_index.cpu().numpy()
    n = data.x.size(0)
    adj = sp.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float32),
         (edge_index[0], edge_index[1])),
        shape=(n, n), dtype=np.float32,
    ).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return adj


def _get_same_sens_complete_graph(adj_scipy, sens_tensor, dataset_name, cache_dir='./adj_files'):
    """
    원본 get_same_sens_complete_graph — 동일 sens끼리 완전 연결 그래프 생성.
    dgl 없이 순수 scipy/torch로 구현. 캐시 파일 사용.
    반환: dense torch.Tensor [N, N]
    """
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f'{dataset_name}_same_sens_complete_adj.pt')

    if os.path.exists(filepath):
        print(f'[FairGT] Loading cached same-sens adj from {filepath}')
        return torch.load(filepath)

    print('[FairGT] Building same-sens complete graph...')
    n = adj_scipy.shape[0]
    rows, cols = [], []

    for key in torch.unique(sens_tensor):
        idx = (sens_tensor == key).nonzero(as_tuple=False).view(-1).numpy()
        if len(idx) <= 1:
            continue
        # 완전 연결 (self-loop 제외)
        rr = np.repeat(idx, len(idx))
        cc = np.tile(idx, len(idx))
        mask = rr != cc
        rows.append(rr[mask])
        cols.append(cc[mask])

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.float32)

    same_adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()

    # dense로 변환
    new_adj = torch.from_numpy(same_adj.toarray())
    torch.save(new_adj, filepath)
    print(f'[FairGT] Saved same-sens adj to {filepath} (nnz={len(rows)})')
    return new_adj


def _row_normalize_dense(adj_dense):
    """dense adj를 row-normalize"""
    rowsum = adj_dense.sum(dim=1, keepdim=True).clamp(min=1e-10)
    return adj_dense / rowsum


# ============================================================
# 메트릭 유틸
# ============================================================

def _safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _group_fairness(y_true, sens, y_pred):
    m0, m1 = sens == 0, sens == 1
    dp = abs((y_pred[m0].mean() if m0.any() else 0.0) -
             (y_pred[m1].mean() if m1.any() else 0.0))
    e0, e1 = (y_true == 1) & m0, (y_true == 1) & m1
    eo = abs((y_pred[e0].mean() if e0.any() else 0.0) -
             (y_pred[e1].mean() if e1.any() else 0.0))
    return float(dp), float(eo)


def _compute_metrics(y_cpu, s_cpu, prob, pred, idx):
    idx = idx.cpu().numpy() if torch.is_tensor(idx) else np.asarray(idx)
    y, s, p, pr = y_cpu[idx], s_cpu[idx], pred[idx], prob[idx]

    def _sub(mask):
        if mask.sum() == 0:
            return 0.0, 0.5, 0.0
        return (float(accuracy_score(y[mask], p[mask])),
                _safe_auc(y[mask], pr[mask]),
                float(f1_score(y[mask], p[mask], zero_division=0)))

    acc = float(accuracy_score(y, p))
    auc = _safe_auc(y, pr)
    f1  = float(f1_score(y, p, zero_division=0))
    a0, u0, f0  = _sub(s == 0)
    a1, u1, f1_ = _sub(s == 1)
    dp, eo = _group_fairness(y, s, p)
    return (acc, auc, f1, a0, u0, f0, a1, u1, f1_, dp, eo)


# ============================================================
# Wrapper
# ============================================================

class FairGT:
    """
    원본 FairGT(LuoRenqiang/FairGT)의 전처리 로직을 그대로 사용.
    fit() / predict() 인터페이스로 감싸서 train_baselines.py와 통합.
    """

    def __init__(self, data, sens_idx, args, lr=1e-3, weight_decay=1e-5, device="cuda"):
        self.device  = device
        self._select = args.fairgt_select

        # ── 피처 / 레이블 / 민감 속성 ───────────────────────────────
        # 원본과 동일하게 feature_normalize 적용
        x_np   = data.x.float().cpu().numpy()
        x_np   = _feature_normalize(x_np)
        x      = torch.FloatTensor(x_np)

        y    = data.y.long().view(-1).cpu()
        sens = data.sens.long().view(-1).cpu()

        # sens 0/1 재매핑 (income: {0,1} 이외 값 대응)
        sens_vals = sorted(sens.unique().tolist())
        if sens_vals != [0, 1]:
            sens_map = {int(v): i for i, v in enumerate(sens_vals)}
            sens_mapped = sens.clone()
            for old_s, new_s in sens_map.items():
                sens_mapped[sens == old_s] = new_s
            sens = sens_mapped
            print(f"[FairGT] sens remapped: {sens_vals} -> {{0, 1}}")

        if getattr(args, 'fairgt_remove_sens_from_x', False) and sens_idx is not None:
            x = torch.cat([x[:, :sens_idx], x[:, sens_idx + 1:]], dim=1)

        # ── 인덱스 — 원본 train_val_test_split 방식 사용 ────────────
        # get_dataset()이 이미 split을 제공하므로 그대로 사용
        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_val   = data.val_mask.nonzero(as_tuple=False).view(-1)
        idx_test  = data.test_mask.nonzero(as_tuple=False).view(-1)

        # invalid label 제거
        valid_y   = y >= 0
        idx_train = idx_train[valid_y[idx_train]]
        idx_val   = idx_val[valid_y[idx_val]]
        idx_test  = idx_test[valid_y[idx_test]]

        # y 재매핑 (혹시 0/1이 아닌 경우)
        used = torch.cat([idx_train, idx_val, idx_test]).unique()
        used_vals = sorted(y[used].unique().tolist())
        if used_vals != [0, 1]:
            label_map = {int(v): i for i, v in enumerate(used_vals)}
            y_mapped = y.clone()
            for old, new in label_map.items():
                y_mapped[y == old] = new
            y = y_mapped

        # ── 그래프 전처리 ────────────────────────────────────────────
        adj_scipy = _pyg_data_to_scipy_adj(data)

        # PE: 원본과 동일하게 adjacency PE
        pe_dim = args.fairgt_pe_dim
        try:
            _, eignvector = _adjacency_positional_encoding(adj_scipy.astype(np.float32), pe_dim)
            lpe = eignvector  # [N, pe_dim]
        except Exception as e:
            print(f"[FairGT] PE failed ({e}), using zeros")
            lpe = torch.zeros(x.shape[0], pe_dim)

        features = torch.cat([x, lpe], dim=1)  # [N, F+pe_dim]

        # same-sens complete graph (원본 로직) → dense adj → row normalize
        dataset_name = args.dataset
        same_sens_adj_dense = _get_same_sens_complete_graph(
            adj_scipy, sens.float(), dataset_name
        )  # [N, N] dense

        adj_norm = _row_normalize_dense(same_sens_adj_dense)  # [N, N]

        # hop aggregation (원본 re_features)
        print(f"[FairGT] Running hop aggregation (hops={args.fairgt_hops})...")
        processed_features = _re_features(adj_norm, features, args.fairgt_hops)
        # [N, hops+1, F+pe_dim]
        print(f"[FairGT] processed_features shape: {processed_features.shape}")

        # ── 모델 & 옵티마이저 ────────────────────────────────────────
        net_params = {
            "hops":       args.fairgt_hops,
            "pe_dim":     pe_dim,
            "in_dim":     processed_features.shape[-1],
            "hidden_dim": args.hidden_dim,
            "n_heads":    args.fairgt_n_heads,
            "n_layers":   args.fairgt_n_layers,
            "nclass":     2,
            "dropout":    getattr(args, 'dropout', 0.0),
        }

        self.model     = _FairGTModel(net_params).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.X         = processed_features.to(device)
        self.y         = y.to(device)
        self.sens      = sens.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_val   = idx_val.to(device)
        self.idx_test  = idx_test.to(device)

        self._y_cpu = y.numpy()
        self._s_cpu = sens.numpy()
        self._best  = None

    # ----------------------------------------------------------
    def fit(self, epochs=1000, patience=100):
        best_score  = -1e18
        bad_counter = 0

        for ep in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(self.X)
            F.cross_entropy(logits[self.idx_train], self.y[self.idx_train]).backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.X)
                prob   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred   = logits.argmax(dim=1).cpu().numpy()

            vi = self.idx_val.cpu().numpy()
            y_v, s_v, p_v, pr_v = self._y_cpu[vi], self._s_cpu[vi], pred[vi], prob[vi]
            val_acc = float(accuracy_score(y_v, p_v))
            val_auc = _safe_auc(y_v, pr_v)
            val_dp, val_eo = _group_fairness(y_v, s_v, p_v)

            # 원본 metric=7 (acc - sp) 기준
            score = {
                "acc":       val_acc,
                "acc_sp":    val_acc - val_dp,
                "acc_sp_eo": val_acc - val_dp - val_eo,
                "auc_sp":    val_auc - val_dp,
            }.get(self._select)
            if score is None:
                raise ValueError(f"Unknown select: {self._select!r}")

            # 원본 조건: val_sp > 0 인 경우에만 best 갱신
            if score > best_score and val_dp > 0:
                best_score  = score
                self._best  = _compute_metrics(
                    self._y_cpu, self._s_cpu, prob, pred, self.idx_test
                )
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter >= patience:
                    break

        # val_dp > 0 조건을 한 번도 못 만족한 경우 마지막 결과 사용
        if self._best is None:
            self._best = _compute_metrics(
                self._y_cpu, self._s_cpu, prob, pred, self.idx_test
            )

    # ----------------------------------------------------------
    def predict(self):
        if self._best is None:
            raise RuntimeError("Call fit() before predict().")
        return self._best