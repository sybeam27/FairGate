import os
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix



def feature_norm(features: torch.Tensor, preserve_cols=None) -> torch.Tensor:
    """
    Normalize each feature column to [-1, 1].
    preserve_cols: list of feature column indices to keep unchanged.
    """
    if preserve_cols is None:
        preserve_cols = []

    features = features.clone()
    min_values = features.min(dim=0).values
    max_values = features.max(dim=0).values
    denom = max_values - min_values
    denom[denom == 0] = 1.0

    normed = 2 * (features - min_values) / denom - 1
    if preserve_cols:
        normed[:, preserve_cols] = features[:, preserve_cols]
    return normed

def index_to_mask(node_num: int, index: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(node_num, dtype=torch.bool)
    mask[index] = True
    return mask

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1)).flatten()
    row_sum[row_sum == 0] = 1
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def build_relationship(x: pd.DataFrame, thresh=0.25, seed=912):
    sims = 1 / (1 + distance_matrix(x.values, x.values))
    idx_map = []
    rng = random.Random(seed)

    for ind in range(sims.shape[0]):
        max_sim = np.sort(sims[ind, :])[-2]
        neig_id = np.where(sims[ind, :] > thresh * max_sim)[0].tolist()
        rng.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])

    return np.array(idx_map, dtype=int)

def make_adj_from_edges(edges: np.ndarray, num_nodes: int):
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    return adj

def balanced_split(labels: torch.Tensor, label_number: int, seed: int = 20):
    label_idx_0 = np.where(labels.cpu().numpy() == 0)[0].tolist()
    label_idx_1 = np.where(labels.cpu().numpy() == 1)[0].tolist()

    rng = random.Random(seed)
    rng.shuffle(label_idx_0)
    rng.shuffle(label_idx_1)

    idx_train = np.append(
        label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
        label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)],
    )
    idx_val = np.append(
        label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))],
    )
    idx_test = np.append(
        label_idx_0[int(0.75 * len(label_idx_0)):],
        label_idx_1[int(0.75 * len(label_idx_1)):],
    )

    return (
        torch.LongTensor(idx_train),
        torch.LongTensor(idx_val),
        torch.LongTensor(idx_test),
    )

def pokec_split(labels: torch.Tensor, sens: np.ndarray, label_number: int, sens_number: int, seed: int = 20, test_idx: bool = False):
    rng = random.Random(seed)
    label_idx = np.where(labels.cpu().numpy() >= 0)[0].tolist()
    rng.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens_valid_idx = set(np.where(sens >= 0)[0].tolist())
    idx_test = np.asarray(list(sens_valid_idx & set(idx_test)))

    idx_sens_train = list(sens_valid_idx - set(idx_val) - set(idx_test))
    rng.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    return (
        torch.LongTensor(idx_train),
        torch.LongTensor(idx_val),
        torch.LongTensor(idx_test),
        idx_sens_train,
    )

def _finalize_as_data(
    features: torch.Tensor,
    labels: torch.Tensor,
    sens: torch.Tensor,
    adj,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
    feature_normalize: bool = True,
    preserve_sens_in_x: bool = True,
    sens_idx_in_x: Optional[int] = None,
):
    if feature_normalize:
        preserve_cols = [sens_idx_in_x] if (preserve_sens_in_x and sens_idx_in_x is not None) else []
        features = feature_norm(features, preserve_cols=preserve_cols)

    adj_norm = sys_normalized_adjacency(adj)
    adj_norm_sp = sparse_mx_to_torch_sparse_tensor(adj_norm)
    edge_index, _ = from_scipy_sparse_matrix(adj)

    train_mask = index_to_mask(features.shape[0], idx_train)
    val_mask = index_to_mask(features.shape[0], idx_val)
    test_mask = index_to_mask(features.shape[0], idx_test)

    data = Data(
        x=features,
        edge_index=edge_index,
        adj_norm_sp=adj_norm_sp,
        y=labels.float(),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        sens=sens,
    )

    x_min = torch.min(features, dim=0).values
    x_max = torch.max(features, dim=0).values
    return data, x_min, x_max

def _to_binary_numpy(values, name: str) -> np.ndarray:
    """
    Robustly convert a binary column to {0,1}.
    Works for numeric 0/1, 1/2, bool-like strings, or two unique categories.
    """
    arr = np.asarray(values)

    if np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.int64)
        uniq = np.unique(arr)
        if len(uniq) != 2:
            raise ValueError(f"{name} must be binary, but got values {uniq}")
        if set(uniq.tolist()) == {0, 1}:
            return arr
        return (arr == uniq.max()).astype(np.int64)

    s = pd.Series(arr).astype(str).str.strip().str.lower()
    uniq = pd.unique(s)
    if len(uniq) != 2:
        raise ValueError(f"{name} must be binary, but got values {uniq}")

    # common positive tokens first
    positive_tokens = {
        "1", "true", "yes", "y", ">50k", ">50k.", "high", "white", "male"
    }
    if any(u in positive_tokens for u in uniq):
        return s.isin(positive_tokens).astype(np.int64).to_numpy()

    # fallback: deterministically map one category to 0, the other to 1
    uniq_sorted = sorted(uniq.tolist())
    return (s == uniq_sorted[1]).astype(np.int64).to_numpy()




def load_income(dataset="income", path="./data/income/", label_number=6000, feature_normalize=True, ):
    df = pd.read_csv(os.path.join(path, f"{dataset}.csv")).copy()

    sens_attr = "race"
    predict_attr = "income"

    header = list(df.columns)
    header.remove(predict_attr)

    edge_path = os.path.join(path, f"{dataset}_edges.txt")
    edges_unordered = np.genfromtxt(edge_path).astype(int)
    if edges_unordered.ndim == 1:
        edges_unordered = edges_unordered.reshape(-1, 2)

    # Keep sensitive attribute in x for consistency with current loaders
    # (German / Credit / Bail also keep sens inside features).
    features = torch.FloatTensor(
        np.array(sp.csr_matrix(df[header], dtype=np.float32).todense())
    )

    labels = torch.LongTensor(_to_binary_numpy(df[predict_attr].values, predict_attr))
    sens = torch.LongTensor(_to_binary_numpy(df[sens_attr].values, sens_attr))

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())),
        dtype=int
    ).reshape(edges_unordered.shape)

    adj = make_adj_from_edges(edges, labels.shape[0])

    idx_train, idx_val, idx_test = balanced_split(labels, label_number, seed=20)

    sens_idx = header.index(sens_attr)

    data, x_min, x_max = _finalize_as_data(
        features,
        labels,
        sens,
        adj,
        idx_train,
        idx_val,
        idx_test,
        feature_normalize=feature_normalize,
        preserve_sens_in_x=True,
        sens_idx_in_x=sens_idx,
    )
    return data, sens_idx, x_min, x_max

def load_credit(dataset="credit", path="./data/credit/", label_number=6000, feature_normalize=True):
    df = pd.read_csv(os.path.join(path, f"{dataset}.csv")).copy()

    sens_attr = "Age"
    predict_attr = "NoDefaultNextMonth"

    header = list(df.columns)
    header.remove(predict_attr)
    header.remove("Single")

    edge_path = os.path.join(path, f"{dataset}_edges.txt")
    if os.path.exists(edge_path):
        edges_unordered = np.genfromtxt(edge_path).astype(int)
    else:
        edges_unordered = build_relationship(df[header], thresh=0.7)
        np.savetxt(edge_path, edges_unordered, fmt="%d")

    features = torch.FloatTensor(np.array(sp.csr_matrix(df[header], dtype=np.float32).todense()))
    labels = torch.LongTensor(df[predict_attr].values)
    sens = torch.LongTensor(df[sens_attr].values.astype(int))

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = make_adj_from_edges(edges, labels.shape[0])

    idx_train, idx_val, idx_test = balanced_split(labels, label_number, seed=20)
    sens_idx = 1

    data, x_min, x_max = _finalize_as_data(
        features, labels, sens, adj, idx_train, idx_val, idx_test,
        feature_normalize=feature_normalize,
        preserve_sens_in_x=True,
        sens_idx_in_x=sens_idx,
    )
    return data, sens_idx, x_min, x_max

def load_bail(dataset="bail", path="./data/bail/", label_number=100, feature_normalize=True):
    df = pd.read_csv(os.path.join(path, f"{dataset}.csv")).copy()

    sens_attr = "WHITE"
    predict_attr = "RECID"

    header = list(df.columns)
    header.remove(predict_attr)

    edge_path = os.path.join(path, f"{dataset}_edges.txt")
    if os.path.exists(edge_path):
        edges_unordered = np.genfromtxt(edge_path).astype(int)
    else:
        edges_unordered = build_relationship(df[header], thresh=0.6)
        np.savetxt(edge_path, edges_unordered, fmt="%d")

    features = torch.FloatTensor(np.array(sp.csr_matrix(df[header], dtype=np.float32).todense()))
    labels = torch.LongTensor(df[predict_attr].values)
    sens = torch.LongTensor(df[sens_attr].values.astype(int))

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = make_adj_from_edges(edges, labels.shape[0])

    idx_train, idx_val, idx_test = balanced_split(labels, label_number, seed=20)
    sens_idx = 0

    data, x_min, x_max = _finalize_as_data(
        features, labels, sens, adj, idx_train, idx_val, idx_test,
        feature_normalize=feature_normalize,
        preserve_sens_in_x=True,
        sens_idx_in_x=sens_idx,
    )
    return data, sens_idx, x_min, x_max

def load_german(dataset="german", path="./data/german/", label_number=100, feature_normalize=True):
    df = pd.read_csv(os.path.join(path, f"{dataset}.csv")).copy()

    sens_attr = "Gender"
    predict_attr = "GoodCustomer"

    header = list(df.columns)
    header.remove(predict_attr)
    header.remove("OtherLoansAtStore")
    header.remove("PurposeOfLoan")

    df.loc[df["Gender"] == "Female", "Gender"] = 1
    df.loc[df["Gender"] == "Male", "Gender"] = 0

    edge_path = os.path.join(path, f"{dataset}_edges.txt")
    if os.path.exists(edge_path):
        edges_unordered = np.genfromtxt(edge_path).astype(int)
    else:
        edges_unordered = build_relationship(df[header], thresh=0.8)
        np.savetxt(edge_path, edges_unordered, fmt="%d")

    features = torch.FloatTensor(np.array(sp.csr_matrix(df[header], dtype=np.float32).todense()))
    labels_np = df[predict_attr].values
    labels_np[labels_np == -1] = 0
    labels = torch.LongTensor(labels_np)
    sens = torch.LongTensor(df[sens_attr].values.astype(int))

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = make_adj_from_edges(edges, labels.shape[0])

    idx_train, idx_val, idx_test = balanced_split(labels, label_number, seed=20)
    sens_idx = 0

    data, x_min, x_max = _finalize_as_data(
        features, labels, sens, adj, idx_train, idx_val, idx_test,
        feature_normalize=feature_normalize,
        preserve_sens_in_x=True,
        sens_idx_in_x=sens_idx,
    )
    return data, sens_idx, x_min, x_max

def load_pokec(dataset="region_job", path="./data/pokec/", label_number=500, sens_number=200, seed=20,
               sens_attr="region", predict_attr="I_am_working_in_field", test_idx=False,
               feature_normalize=True):
    df = pd.read_csv(os.path.join(path, f"{dataset}.csv")).copy()

    header = list(df.columns)
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)

    features = torch.FloatTensor(np.array(sp.csr_matrix(df[header], dtype=np.float32).todense()))
    labels = torch.LongTensor(df[predict_attr].values)
    labels[labels > 1] = 1

    idx = np.array(df["user_id"], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges_unordered = np.genfromtxt(
        os.path.join(path, f"{dataset}_relationship.txt"),
        dtype=np.int64
    )
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64).reshape(edges_unordered.shape)
    adj = make_adj_from_edges(edges, labels.shape[0])

    sens_np = df[sens_attr].values
    sens = torch.FloatTensor(sens_np)

    # data_utils 스타일에 맞추되, 기존 dataloading 동작을 유지하기 위해 sens를 마지막 feature로 추가
    features = torch.cat([features, sens.unsqueeze(-1)], dim=-1)
    sens_idx = features.shape[1] - 1

    idx_train, idx_val, idx_test, idx_sens_train = pokec_split(
        labels=labels,
        sens=sens_np,
        label_number=label_number,
        sens_number=sens_number,
        seed=seed,
        test_idx=test_idx,
    )

    data, x_min, x_max = _finalize_as_data(
        features, labels, sens, adj, idx_train, idx_val, idx_test,
        feature_normalize=feature_normalize,
        preserve_sens_in_x=True,
        sens_idx_in_x=sens_idx,
    )
    data.idx_sens_train = idx_sens_train
    return data, sens_idx, x_min, x_max

def load_nba(dataset="nba", path="./data/NBA", label_number=100, sens_number=50, seed=20,
             sens_attr="country", predict_attr="SALARY", feature_normalize=True):
    return load_pokec(
        dataset=dataset,
        path=path,
        label_number=label_number,
        sens_number=sens_number,
        seed=seed,
        sens_attr=sens_attr,
        predict_attr=predict_attr,
        test_idx=True,
        feature_normalize=feature_normalize,
    )



def get_dataset(dataname: str, path: str = "./data", feature_normalize: bool = True) -> Tuple[Data, int, torch.Tensor, torch.Tensor]:
    """
    Return everything in data_utils.py style:
        Data(x, edge_index, adj_norm_sp, y, train_mask, val_mask, test_mask, sens),
        sens_idx,
        x_min,
        x_max
    Supported dataname:
        credit, bail, recidivism, german, pokec_z, pokec_n, nba
    """
    if dataname == "credit":
        return load_credit(
            dataset="credit",
            path=os.path.join(path, "credit"),
            label_number=6000,
            feature_normalize=feature_normalize,
        )

    if dataname in ["bail", "recidivism"]:
        return load_bail(
            dataset="bail",
            path=os.path.join(path, "bail"),
            label_number=100,
            feature_normalize=feature_normalize,
        )

    if dataname == "german":
        return load_german(
            dataset="german",
            path=os.path.join(path, "german"),
            label_number=100,
            feature_normalize=feature_normalize,
        )

    if dataname == "pokec_z":
        return load_pokec(
            dataset="region_job",
            path=os.path.join(path, "pokec"),
            label_number=500,
            sens_number=200,
            seed=20,
            sens_attr="region",
            predict_attr="I_am_working_in_field",
            test_idx=False,
            feature_normalize=feature_normalize,
        )

    if dataname == "pokec_z_g":
        return load_pokec(
            dataset="region_job",
            path=os.path.join(path, "pokec"),
            label_number=500,
            sens_number=200,
            seed=20,
            sens_attr="gender",
            predict_attr="I_am_working_in_field",
            test_idx=False,
            feature_normalize=feature_normalize,
        )

    if dataname == "pokec_n":
        return load_pokec(
            dataset="region_job_2",
            path=os.path.join(path, "pokec"),
            label_number=500,
            sens_number=200,
            seed=20,
            sens_attr="region",
            predict_attr="I_am_working_in_field",
            test_idx=False,
            feature_normalize=feature_normalize,
        )

    if dataname == "pokec_n_g":
        return load_pokec(
            dataset="region_job_2",
            path=os.path.join(path, "pokec"),
            label_number=500,
            sens_number=200,
            seed=20,
            sens_attr="gender",
            predict_attr="I_am_working_in_field",
            test_idx=False,
            feature_normalize=feature_normalize,
        )

    if dataname == "nba":
        return load_nba(
            dataset="nba",
            path=os.path.join(path, "NBA"),
            label_number=100,
            sens_number=50,
            seed=20,
            sens_attr="country",
            predict_attr="SALARY",
            feature_normalize=feature_normalize,
        )

    if dataname == "income":
        return load_income(
            dataset="income",
            path=os.path.join(path, "income"),
            label_number=6000,
            feature_normalize=feature_normalize,
        )
    
    raise NotImplementedError(f"Unsupported dataname: {dataname}")
