import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def classification_metrics(logits, labels):
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    preds = (probs > 0.5).astype(int)
    y_true = labels.detach().cpu().numpy().astype(int)

    acc = (preds == y_true).mean()
    f1 = f1_score(y_true, preds, zero_division=0)

    try:
        roc = roc_auc_score(y_true, probs)
    except ValueError:
        roc = float("nan")

    return {"acc": float(acc),
            "roc_auc": float(roc),
            "f1": float(f1)}

def fairness_metrics(logits, labels, sens, idx):
    """Classification fairness: DP, EO"""
    if torch.is_tensor(idx):
        idx = idx.detach().cpu().numpy()

    y_true = labels.detach().cpu().numpy()[idx].astype(int)
    s = sens.detach().cpu().numpy()[idx].astype(int)
    probs = torch.sigmoid(logits[idx]).detach().cpu().numpy().reshape(-1)
    preds = (probs > 0.5).astype(int)

    mask_0 = (s == 0)
    mask_1 = (s == 1)

    p0 = preds[mask_0].mean() if mask_0.sum() > 0 else 0.0
    p1 = preds[mask_1].mean() if mask_1.sum() > 0 else 0.0
    dp = abs(p0 - p1)

    mask_0_y1 = np.logical_and(mask_0, y_true == 1)
    mask_1_y1 = np.logical_and(mask_1, y_true == 1)
    eo0 = preds[mask_0_y1].mean() if mask_0_y1.sum() > 0 else 0.0
    eo1 = preds[mask_1_y1].mean() if mask_1_y1.sum() > 0 else 0.0
    eo = abs(eo0 - eo1)

    # ── Subgroup accuracy gap (Extra fairness metric) ──────────
    # NOTE: train_baselines.py already computes acc_sens0/acc_sens1 via
    # pack_result(). If those columns already exist in your saved CSVs,
    # you don't need this — just take abs(acc_sens0 - acc_sens1) there.
    # This is included here so FairGate's own evaluate() path (which does
    # NOT currently call fairness_metrics for per-group accuracy) can get
    # the same number without touching model_fairgate.py's core logic.
    acc0 = (preds[mask_0] == y_true[mask_0]).mean() if mask_0.sum() > 0 else float("nan")
    acc1 = (preds[mask_1] == y_true[mask_1]).mean() if mask_1.sum() > 0 else float("nan")
    acc_gap = abs(acc0 - acc1)

    return {
        "dp": float(dp),
        "eo": float(eo),
        "acc_sens0": float(acc0),
        "acc_sens1": float(acc1),
        "acc_gap": float(acc_gap),
    }

def individual_fairness_consistency(probs, features, idx, k=5):
    """
    Individual fairness via k-NN consistency in raw feature space.

    Higher = more individually fair (similar nodes -> similar predictions).
    Only needs per-node prediction probabilities + input features (data.x),
    both of which already exist at evaluation time -- no embeddings needed,
    no retraining needed.

    Args:
        probs:    (N,) array-like of predicted probabilities (sigmoid outputs)
                  for ALL nodes (not just idx) -- so neighbors can be found
                  within the full node set, then restricted to idx for the
                  final score.
        features: (N, d) tensor/array of raw node features (e.g. data.x)
        idx:      indices (train/val/test mask) to compute the score over
        k:        number of nearest neighbors (excluding self)

    Returns:
        dict with "ifair_consistency" (higher = better) and
        "ifair_inconsistency" (raw avg pairwise diff, lower = better,
        kept for readers used to "lower is fairer" conventions elsewhere
        in this codebase, e.g. dp/eo).
    """
    if torch.is_tensor(probs):
        probs = probs.detach().cpu().numpy().reshape(-1)
    else:
        probs = np.asarray(probs).reshape(-1)

    if torch.is_tensor(features):
        feats_np = features.detach().cpu().numpy()
    else:
        feats_np = np.asarray(features)

    if torch.is_tensor(idx):
        idx = idx.detach().cpu().numpy()
    idx = np.asarray(idx)

    n_neighbors = min(k + 1, feats_np.shape[0])  # +1: self is included by kNN
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(feats_np)
    _, neighbor_idx = nbrs.kneighbors(feats_np[idx])

    diffs = []
    for row_pos, center_node in enumerate(idx):
        neighbors = neighbor_idx[row_pos]
        neighbors = neighbors[neighbors != center_node]
        if len(neighbors) == 0:
            continue
        diffs.append(np.abs(probs[center_node] - probs[neighbors]).mean())

    inconsistency = float(np.mean(diffs)) if diffs else float("nan")
    return {
        "ifair_inconsistency": inconsistency,   # lower = fairer (matches dp/eo convention)
        "ifair_consistency": 1.0 - inconsistency if not np.isnan(inconsistency) else float("nan"),
    }

def sensitive_leakage_probe(embeddings, sens, train_idx, test_idx, seed=0, max_iter=1000):
    """
    How much sensitive-attribute information is recoverable from FROZEN
    learned embeddings, via a simple linear probe trained AFTER the main
    model is done training. Lower probe AUC = less leakage = fairer
    representation. This is a different axis than DP/EO (which look at
    prediction rates, not representation content), so it is NOT something
    Lout directly optimizes.

    IMPORTANT: this needs node embeddings (pre-classification-head hidden
    representations), not just predictions. Whether these are available
    without retraining depends on whether the trained model/checkpoint
    exposes an embedding-extraction method (e.g. model.encode(data)).
    See integration notes.

    Args:
        embeddings: (N, hidden_dim) tensor/array of frozen node embeddings
        sens:       (N,) tensor/array of sensitive attribute labels
        train_idx, test_idx: indices for probe training / evaluation
                    (reuse the model's own train/test split, or split idx
                    further -- using a split disjoint from the main model's
                    test set is more conservative but not required, since
                    the probe is diagnostic, not part of the main claim)
        seed:       random seed for the probe classifier

    Returns:
        dict with "leakage_probe_auc" (0.5 = ideal/no leakage, 1.0 = fully
        recoverable) and "leakage_probe_acc".
    """
    if torch.is_tensor(embeddings):
        emb_np = embeddings.detach().cpu().numpy()
    else:
        emb_np = np.asarray(embeddings)

    if torch.is_tensor(sens):
        sens_np = sens.detach().cpu().numpy().astype(int)
    else:
        sens_np = np.asarray(sens).astype(int)

    if torch.is_tensor(train_idx):
        train_idx = train_idx.detach().cpu().numpy()
    if torch.is_tensor(test_idx):
        test_idx = test_idx.detach().cpu().numpy()

    probe = LogisticRegression(max_iter=max_iter, random_state=seed)
    probe.fit(emb_np[train_idx], sens_np[train_idx])

    probe_probs = probe.predict_proba(emb_np[test_idx])[:, 1]
    probe_preds = (probe_probs > 0.5).astype(int)

    try:
        auc = roc_auc_score(sens_np[test_idx], probe_probs)
    except ValueError:
        auc = float("nan")
    acc = (probe_preds == sens_np[test_idx]).mean()

    return {
        "leakage_probe_auc": float(auc),
        "leakage_probe_acc": float(acc),
    }

def evaluate_pyg_model(model, data, split="val", task_type="classification",
                        compute_extra_fairness=False, embeddings=None):
    # [Fix] task_type 파라미터 추가 (model.py에서 전달하므로 시그니처 맞춤)
    model.eval()

    if split == "train":
        idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    elif split == "val":
        idx = data.val_mask.nonzero(as_tuple=False).view(-1)
    elif split == "test":
        idx = data.test_mask.nonzero(as_tuple=False).view(-1)
    else:
        raise ValueError("split must be one of ['train', 'val', 'test'].")

    with torch.no_grad():
        out = model(data)
        if isinstance(out, tuple):
            out = out[0]
        out = out.view(-1)

    y = data.y
    # [Fix] data.sensitive_attr → data.sens (data.py가 sens로 저장)
    s = data.sens

    perf = classification_metrics(out[idx], y[idx])
    fair = fairness_metrics(out, y, s, idx)  # now also returns acc_sens0/1, acc_gap
    result = {**perf, **fair}

    # ── Extra fairness metrics (opt-in, backward compatible) ──────
    # compute_extra_fairness=False by default so existing callers
    # (e.g. train_fairgate.py / train_baselines.py, if they call this
    # function at all) are unaffected unless explicitly opted in.
    if compute_extra_fairness:
        probs_full = torch.sigmoid(out).detach().cpu().numpy().reshape(-1)
        ifair = individual_fairness_consistency(probs_full, data.x, idx, k=5)
        result.update(ifair)

        if embeddings is not None:
            train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
            leak = sensitive_leakage_probe(embeddings, s, train_idx, idx)
            result.update(leak)
        # If embeddings is None, leakage_probe_* keys are simply omitted
        # rather than filled with NaN, so the caller can tell "not computed"
        # apart from "computed but undefined" at a glance.

    result = {
        k: (round(v, 4) if isinstance(v, (float, int)) else v)
        for k, v in result.items()
    }

    return result