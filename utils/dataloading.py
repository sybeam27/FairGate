"""
dataloading.py — 어댑터 모듈

비교 모델들(FairGNN, NIFTY, EDITS, FairEdit, FairVGNN, FairWalk, CrossWalk)은
아래 형식으로 데이터를 받습니다:

    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset)

우리 data.py의 get_dataset()이 반환하는 PyG Data 객체를 분해해서
위 형식으로 변환해주는 어댑터입니다.

각 알고리즘이 기대하는 adj 형식:
    - FairGNN  : torch sparse tensor → adj.to_dense() 후 edge_index 변환
    - FairVGNN : torch sparse tensor → adj.to_dense() 후 내부에서 재처리
    - FairEdit : torch sparse tensor → adj.to_dense() 후 scipy coo 변환
    - EDITS    : torch sparse tensor → adj.to_dense() 후 scipy coo 변환
    - CrossWalk: torch sparse tensor → adj.to_dense() 후 nx.from_numpy_array
    - FairWalk : torch sparse tensor → adj.to_dense() 후 scipy coo 변환
    → 모두 torch sparse tensor (COO)를 넘기면 됩니다.
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_scipy_sparse_matrix

from utils.data import get_dataset


def load_data(dataset: str, feature_normalize: bool = True):
    """
    반환값:
        adj         : torch sparse COO tensor (정규화 없는 원본 인접행렬)
        feats       : torch FloatTensor [N, F]
        labels      : torch LongTensor  [N]
        idx_train   : torch LongTensor  [train_size]
        idx_val     : torch LongTensor  [val_size]
        idx_test    : torch LongTensor  [test_size]
        sens        : torch LongTensor  [N]
        sens_idx    : int
    """
    data, sens_idx, x_min, x_max = get_dataset(dataset, feature_normalize=feature_normalize)

    # ── features / labels / sens
    feats  = data.x                        # FloatTensor [N, F]
    labels = data.y.long()                 # LongTensor  [N]
    sens   = data.sens                     # LongTensor  [N]

    # ── train / val / test index (mask → index)
    idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
    idx_val   = data.val_mask.nonzero(as_tuple=False).view(-1)
    idx_test  = data.test_mask.nonzero(as_tuple=False).view(-1)

    # ── adj: edge_index → scipy coo → torch sparse COO
    #    비교 모델들은 모두 adj.to_dense() 또는 sp.coo_matrix(adj.to_dense())를 호출하므로
    #    정규화되지 않은 원본 인접행렬을 torch sparse tensor로 제공합니다.
    N = feats.size(0)
    scipy_adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=N)  # scipy coo
    scipy_adj = (scipy_adj + scipy_adj.T)                             # 대칭화
    scipy_adj.data = np.ones_like(scipy_adj.data)                     # 이진화
    scipy_adj = scipy_adj.tocoo()

    indices = torch.from_numpy(
        np.vstack([scipy_adj.row, scipy_adj.col]).astype(np.int64)
    )
    values = torch.FloatTensor(scipy_adj.data)
    adj = torch.sparse_coo_tensor(indices, values, (N, N))

    return adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx
