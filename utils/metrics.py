import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


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

    return {
        "dp": float(dp),
        "eo": float(eo),
    }

def evaluate_pyg_model(model, data, split="val", task_type="classification"):
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
    fair = fairness_metrics(out, y, s, idx)
    result = {**perf, **fair}

    result = {
        k: (round(v, 4) if isinstance(v, (float, int)) else v)
        for k, v in result.items()
    }

    return result