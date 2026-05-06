"""
FairGate — Fair Graph Neural Network with Hierarchical Fairness Intervention

Design (v3 — dual-head learned uncertainty):
    Module 1: Hierarchical Fairness Intervention Weight (FIW)
              - topology-adaptive gating signal
                  (boundary / degree / loss-blend by graph regime)
              - variance-weighted structural priority score
              - learned uncertainty σ(v) from dual-head UQ for within-gate
                modulation (replaces entropy proxy)
    Module 2: 3-Level Fairness Loss
              L = L_task + λ_fair · (L_struct + L_rep + L_out)
            + λ_uq · L_uq   (dual-head coverage + width penalty)
    Module 3: Scale-calibrated Training Loop
              warm-up → FIW update → auto scale calibration → early stopping

Key design decisions:
    1) w_boundary and w_lhd are not independent FIW inputs.
       w_boundary is the primary structural signal; w_lhd is diagnostics only.
    2) FIW is hierarchical:
         (a) Gate nodes by topology-adaptive structural signal.
         (b) Rank gated nodes via variance-weighted boundary-degree score.
         (c) Modulate with learned σ(v) from the dual-head UQ head.
    3) Learned uncertainty σ(v) replaces entropy as the within-gate
       modulation signal.  Unlike entropy (a deterministic function of p),
       σ(v) is trained via a coverage+width loss and captures node-level
       aleatoric uncertainty independently of predicted probability.
       Nodes with higher σ(v) receive stronger FIW, regardless of graph
       topology—avoiding the direction-switching problem of entropy.
    4) CF-FV direction switching is intentionally removed:
       it was needed only when using entropy because entropy correlates
       with fairness risk in opposite directions for Pattern A vs Pattern B.
       Learned σ(v) does not have this ambiguity.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, SGConv

from utils.metrics import evaluate_pyg_model


# ── Internal constants ────────────────────────────────────────────────────────
_WEIGHT_MIN        = 0.5
_WEIGHT_MAX        = 2.0
_INTRA_DROP_RATIO  = 0.1
_MC_SAMPLES        = 10
_REFRESH_INTERVAL  = 100
_RECAL_INTERVAL    = 200   # periodic scale recalibration every N Phase-2 epochs


# ============================================================
# Backbone GNN Models
# ============================================================

class GCN(nn.Module):
    """GCN backbone with a decoupled prediction head and uncertainty-width head.

    The default call remains backward-compatible: model(data) returns logits.
    Use return_uq=True to additionally obtain a positive logit-width sigma(v).
    """
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, h_feats)
        self.conv2   = GCNConv(h_feats, 1)          # prediction head
        self.uq_head = nn.Linear(h_feats, 1)        # uncertainty-width head
        self.dropout = dropout

    def forward(self, data, edge_index=None, edge_weight=None,
                return_hidden=False, return_uq=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h   = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index, edge_weight=edge_weight).view(-1)
        if return_uq:
            sigma = F.softplus(self.uq_head(h).view(-1)) + 1e-6
            if return_hidden:
                return out, h, sigma
            return out, sigma
        return (out, h) if return_hidden else out


class GraphSAGE(nn.Module):
    """GraphSAGE backbone with a decoupled uncertainty-width head."""
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = SAGEConv(in_feats, h_feats)
        self.conv2   = SAGEConv(h_feats, 1)
        self.uq_head = nn.Linear(h_feats, 1)
        self.dropout = dropout

    def forward(self, data, edge_index=None, edge_weight=None,
                return_hidden=False, return_uq=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        # SAGEConv does not accept edge_weight → ignore silently
        h   = F.relu(self.conv1(x, edge_index))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)
        if return_uq:
            sigma = F.softplus(self.uq_head(h).view(-1)) + 1e-6
            if return_hidden:
                return out, h, sigma
            return out, sigma
        return (out, h) if return_hidden else out


class SGC(nn.Module):
    """SGC backbone with a simple uncertainty-width head on propagated features."""
    def __init__(self, in_feats, sgc_k=2):
        super().__init__()
        self.conv = SGConv(in_feats, 1, K=sgc_k)
        self.uq_head = nn.Linear(in_feats, 1)

    def forward(self, data, edge_index=None, edge_weight=None,
                return_hidden=False, return_uq=False):
        # SGConv does not support edge_weight; scale mode falls back to drop in
        # compute_structural_loss automatically via the TypeError catch.
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        out = self.conv(x, edge_index).view(-1)
        if return_uq:
            sigma = F.softplus(self.uq_head(x).view(-1)) + 1e-6
            if return_hidden:
                return out, x, sigma
            return out, sigma
        return (out, x) if return_hidden else out


def _build_backbone(name, in_feats, h_feats, dropout=0.5, sgc_k=2):
    if name == "GCN":
        return GCN(in_feats, h_feats, dropout=dropout)
    elif name == "GraphSAGE":
        return GraphSAGE(in_feats, h_feats, dropout=dropout)
    elif name == "SGC":
        return SGC(in_feats, sgc_k=sgc_k)
    else:
        raise ValueError(f"Unsupported backbone: {name}")


# ============================================================
# Module 1: Hierarchical Fairness Intervention Weight (FIW)
# ============================================================

def _minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _top_quantile_gate(score, q: float, min_gate: int = 1):
    """Select top-(1-q) nodes without tie-induced full gating."""
    N = score.numel()
    q = float(max(0.0, min(0.999999, q)))
    k = max(min_gate, int(round((1.0 - q) * N)))
    k = min(max(k, min_gate), N)
    idx = torch.topk(score, k=k, largest=True, sorted=False).indices
    gate = torch.zeros(N, dtype=torch.bool, device=score.device)
    gate[idx] = True
    thr = float(score[idx].min().item()) if idx.numel() > 0 else float(score.max().item())
    return gate, thr


def _compute_structural_signals(data):
    """
    Compute node-level structural signals.

    Returns:
        dict with:
            w_degree, w_boundary, w_lhd
    """
    edge_index = data.edge_index
    sens       = data.sens
    N          = data.x.size(0)
    device     = data.x.device

    src, dst = edge_index
    ones     = torch.ones(edge_index.size(1), device=device)

    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, src, ones)

    # Degree
    log_deg  = torch.log1p(deg)
    w_degree = _minmax(log_deg)

    # Boundary risk: fraction of cross-group neighbors
    cross       = (sens[src] != sens[dst]).float()
    cross_count = torch.zeros(N, device=device)
    cross_count.scatter_add_(0, src, cross)
    boundary    = cross_count / (deg + 1e-8)

    # Light log scaling helps avoid score collapse on high-homophily graphs
    boundary_log = torch.log1p(boundary * 10.0)
    w_boundary   = _minmax(boundary_log)

    # Local homophily deviation (diagnostic only in the revised FIW)
    same       = (sens[src] == sens[dst]).float()
    same_count = torch.zeros(N, device=device)
    same_count.scatter_add_(0, src, same)
    local_h    = same_count / (deg + 1e-8)
    lhd        = torch.abs(local_h - local_h.mean())
    w_lhd      = _minmax(lhd)

    return {
        "w_degree": w_degree.detach(),
        "w_boundary": w_boundary.detach(),
        "w_lhd": w_lhd.detach(),
    }


@torch.no_grad()
def _estimate_entropy_uncertainty(model, data):
    """
    Predictive entropy uncertainty from a single deterministic forward pass.
    This is the default uncertainty source because it matches the
    pre-method analysis that motivates FIW.
    """
    was_training = model.training
    model.eval()

    out = model(data)
    if isinstance(out, tuple):
        out = out[0]
    p = torch.sigmoid(out.view(-1)).clamp(1e-6, 1.0 - 1e-6)
    u = -(p * p.log() + (1.0 - p) * (1.0 - p).log())

    if was_training:
        model.train()

    return _minmax(u).detach()


@torch.no_grad()
def _estimate_mc_uncertainty(model, data):
    """
    Optional MC-Dropout uncertainty estimate.
    Kept as an ablation / optional setting, but not the default.
    """
    was_training = model.training
    model.train()

    samples = []
    for _ in range(_MC_SAMPLES):
        out = model(data)
        if isinstance(out, tuple):
            out = out[0]
        samples.append(torch.sigmoid(out.view(-1)).unsqueeze(0))

    preds  = torch.cat(samples, dim=0)  # [T, N]
    mean_p = preds.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
    entropy = -(mean_p * mean_p.log() + (1.0 - mean_p) * (1.0 - mean_p).log())
    u = entropy + preds.var(dim=0)

    if not was_training:
        model.eval()

    return _minmax(u).detach()


def _compute_edge_homophily(data) -> float:
    """
    Edge homophily ratio: fraction of edges connecting same-group nodes.
    h ∈ [0, 1]; high-homophily graphs have h close to 1.
    Used to detect heterophilic graphs where boundary-only gating may mislead.
    """
    src, dst = data.edge_index
    same = (data.sens[src] == data.sens[dst]).float()
    return float(same.mean().item())


@torch.no_grad()
def _compute_loss_based_signal(model, data) -> torch.Tensor:
    """
    Per-node fairness-relevant risk signal.

    BCE is computed only for valid binary labels. Nodes with invalid labels
    such as -1 are assigned zero supervised loss and still receive structural
    risk through boundary/degree signals. This avoids corrupting FIW on
    Pokec/NBA, where unlabeled nodes remain in the graph.
    """
    was_training = model.training
    model.eval()

    out = model(data)
    if isinstance(out, tuple):
        out = out[0]
    out = out.view(-1)

    labels = data.y.float()
    valid = (labels == 0.0) | (labels == 1.0)

    node_loss = torch.zeros_like(out)
    if valid.sum() > 0:
        node_loss[valid] = F.binary_cross_entropy_with_logits(
            out[valid], labels[valid], reduction="none"
        )

    edge_index = data.edge_index
    src, dst   = edge_index
    N          = data.x.size(0)
    device     = data.x.device
    ones       = torch.ones(edge_index.size(1), device=device)
    deg        = torch.zeros(N, device=device)
    deg.scatter_add_(0, src, ones)
    cross      = (data.sens[src] != data.sens[dst]).float()
    cross_cnt  = torch.zeros(N, device=device)
    cross_cnt.scatter_add_(0, src, cross)
    inter_frac = cross_cnt / (deg + 1e-8)

    signal = node_loss * (0.5 + inter_frac)
    signal = _minmax(signal) if valid.sum() > 0 else _minmax(inter_frac)

    if was_training:
        model.train()
    return signal.detach()


@torch.no_grad()
def _estimate_dual_uncertainty(model, data):
    """Uncertainty from the dual-head logit interval width.

    The model predicts logit z and positive width sigma. We convert this into
    a probability interval [sigmoid(z-sigma), sigmoid(z+sigma)] and use its
    width as the node-level uncertainty score. Unlike entropy, this uncertainty
    is learned by a separate head and is not a deterministic function of p.
    """
    was_training = model.training
    model.eval()

    out = model(data, return_uq=True)
    if not isinstance(out, tuple) or len(out) != 2:
        raise RuntimeError("Model must support return_uq=True for dual uncertainty.")
    z, sigma = out
    z = z.view(-1)
    sigma = sigma.view(-1).clamp(min=1e-6, max=20.0)
    p_lo = torch.sigmoid(z - sigma)
    p_hi = torch.sigmoid(z + sigma)
    width = (p_hi - p_lo).clamp(min=0.0)

    if was_training:
        model.train()
    return _minmax(width).detach()


def compute_dual_uq_loss(model, data, idx_train=None, width_penalty=0.05, node_weight=None):
    """QpiGNN-inspired dual-head uncertainty objective for binary classification.

    The uncertainty head predicts a logit interval [z-sigma, z+sigma].
    We penalize labels that fall outside the corresponding probability interval
    while also discouraging trivially wide intervals. This gives the uncertainty
    head its own learning signal, instead of reusing entropy as uncertainty.

    L_uq = coverage_violation + width_penalty * interval_width

    Important: this loss should be computed on the training labels only.
    """
    if idx_train is None:
        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)

    z, sigma = model(data, return_uq=True)
    z = z.view(-1)
    sigma = sigma.view(-1).clamp(min=1e-6, max=20.0)

    labels = data.y.float()
    valid = ((labels == 0.0) | (labels == 1.0))
    mask = torch.zeros_like(valid, dtype=torch.bool)
    mask[idx_train] = True
    idx = (valid & mask).nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        zero = z.sum() * 0.0
        return zero, {"uq_cov": 0.0, "uq_width": 0.0}

    y = labels[idx]
    p_lo = torch.sigmoid(z[idx] - sigma[idx])
    p_hi = torch.sigmoid(z[idx] + sigma[idx])
    interval_width = (p_hi - p_lo).clamp(min=0.0)

    # zero if y lies inside [p_lo, p_hi], positive otherwise
    coverage_violation = F.relu(p_lo - y) + F.relu(y - p_hi)

    if node_weight is not None:
        w = node_weight[idx].detach()
        w = w / (w.mean() + 1e-8)
        cov_loss = (coverage_violation * w).mean()
        width_loss = (interval_width * w).mean()
    else:
        cov_loss = coverage_violation.mean()
        width_loss = interval_width.mean()

    loss = cov_loss + float(width_penalty) * width_loss
    return loss, {
        "uq_cov": float(cov_loss.detach()),
        "uq_width": float(width_loss.detach()),
    }


def _estimate_uncertainty(model, data, uncertainty_type="entropy"):
    if uncertainty_type == "entropy":
        return _estimate_entropy_uncertainty(model, data)
    elif uncertainty_type == "mc":
        return _estimate_mc_uncertainty(model, data)
    elif uncertainty_type == "dual":
        return _estimate_dual_uncertainty(model, data)
    else:
        raise ValueError(f"Unsupported uncertainty_type: {uncertainty_type}")


# ── Homophily threshold below which the adaptive gating is activated ──────────
_LOW_HOMOPHILY_THR = 0.5


def compute_fiw_weights(
    data,
    model=None,
    sbrs_quantile=0.7,
    fips_lam=1.0,
    uncertainty_type="entropy",
    gating_mode_override: str = "adaptive",
    fiw_weight_mode: str = "continuous_uncert",
):
    """
    Compute Hierarchical Fairness Intervention Weights (FIW).

    Revised rule:
        1) Determine gating signal G based on graph regime:
           - boundary   (default):   G = w_boundary
           - heterophilic (h<0.5):   G = mix(w_boundary, loss_signal)
           - saturated (bnd≥0.9):    G = loss_signal  [Phase 2]
                                         w_degree      [warm-up]
           Gate nodes: {v | G(v) >= quantile_q(G)}
        2) Inside the gate, compute structural priority score:
           alpha = Var(w_boundary) / (Var(w_boundary) + Var(w_degree))
           beta  = Var(w_degree)   / (Var(w_boundary) + Var(w_degree))
           s_struct(v) = alpha * w_boundary(v) + beta * w_degree(v)
        3) Within-gate modulation with learned uncertainty σ(v):
           FIW(v) ∝ s_struct(v) * (1 + fips_lam * σ(v))
           where σ(v) = dual-head predicted interval width,
           trained via coverage + width penalty loss (L_uq).
           Unlike entropy, σ(v) is not a deterministic function of p(v);
           it captures node-level aleatoric uncertainty independently,
           so higher σ(v) reliably indicates higher intervention need
           regardless of graph topology.

    Graph regime detection:
        boundary_ratio ≥ 0.9 (saturated): w_boundary 변별력 붕괴
            → loss_signal / w_degree 기반 gating으로 전환
        homophily < 0.5 (heterophilic): w_boundary 단독 신호 부정확
            → w_boundary + loss_signal 혼합
        그 외: w_boundary 기반 gating (기본)
    """
    N      = data.x.size(0)
    device = data.x.device

    assert gating_mode_override in (
        "none", "score", "adaptive", "boundary", "degree",
        "boundary_degree", "loss", "random"
    ), f"Unsupported gating_mode_override: {gating_mode_override}"
    assert fiw_weight_mode in (
        "uniform", "struct_only", "continuous_uncert",
        "binary_mean", "matched_random_perm"
    ), f"Unsupported fiw_weight_mode: {fiw_weight_mode}"

    sig = _compute_structural_signals(data)
    w_degree   = sig["w_degree"]
    w_boundary = sig["w_boundary"]
    w_lhd      = sig["w_lhd"]  # diagnostics only

    # ── Graph-adaptive FIW gating ─────────────────────────────────────
    # The previous fixed rule used boundary-first gating for all graphs.
    # The updated rule makes the *selection signal* adaptive while keeping
    # continuous risk-aware weighting.
    #
    # Override modes for ablation:
    #   boundary        : gate by boundary exposure only
    #   degree          : gate by degree risk only
    #   boundary_degree : gate by variance-mixed boundary/degree score
    #   loss            : gate by loss-based signal when available
    #   random          : random gate with the same budget
    #   none            : uniform weights
    #   score/adaptive  : graph-regime-adaptive default
    homophily      = _compute_edge_homophily(data)
    graph_stats    = _compute_graph_stats(data)
    boundary_ratio = graph_stats["boundary_ratio"]
    deg_gap        = graph_stats["deg_gap"]

    has_model = model is not None
    loss_signal = _compute_loss_based_signal(model, data) if has_model else None

    vars_tmp = torch.stack([w_boundary.var(), w_degree.var()])
    coefs_tmp = vars_tmp / (vars_tmp.sum() + 1e-8)
    bd_score = _minmax(coefs_tmp[0] * w_boundary + coefs_tmp[1] * w_degree)

    requested_gate = "adaptive" if gating_mode_override == "score" else gating_mode_override

    if requested_gate == "boundary":
        gating_signal = w_boundary
        gating_mode = "boundary"
    elif requested_gate == "degree":
        gating_signal = w_degree
        gating_mode = "degree"
    elif requested_gate == "boundary_degree":
        gating_signal = bd_score
        gating_mode = "boundary_degree"
    elif requested_gate == "loss":
        gating_signal = loss_signal if loss_signal is not None else w_degree
        gating_mode = "loss" if loss_signal is not None else "loss_fallback_degree"
    elif requested_gate == "random":
        gating_signal = bd_score
        gating_mode = "random"
    else:
        # Adaptive default.
        if boundary_ratio >= _BOUNDARY_SAT_THR:
            gating_signal = loss_signal if loss_signal is not None else w_degree
            gating_mode = "adaptive_saturated_loss" if loss_signal is not None else "adaptive_saturated_degree"
        elif deg_gap > _DEG_GAP_THR:
            gating_signal = w_degree
            gating_mode = "adaptive_degree"
        elif homophily < _LOW_HOMOPHILY_THR and loss_signal is not None:
            mix = 1.0 - homophily / _LOW_HOMOPHILY_THR
            gating_signal = _minmax((1.0 - mix) * w_boundary + mix * loss_signal)
            gating_mode = "adaptive_heterophilic"
        elif boundary_ratio < _BOUNDARY_RATIO_THR:
            gating_signal = w_boundary
            gating_mode = "adaptive_boundary"
        else:
            gating_signal = bd_score
            gating_mode = "adaptive_boundary_degree"

    # Exact top-quantile gate avoids full gating due to tied scores.
    score_gate, boundary_threshold = _top_quantile_gate(gating_signal, sbrs_quantile)
    gate = score_gate.clone()

    # FIW ablation: override the node-selection mechanism.
    # score : original score-based FIW gate.
    # random: uniformly random gate with the same number of gated nodes.
    # none  : disable FIW and return uniform weights.
    if gating_mode_override == "none" or fiw_weight_mode == "uniform":
        weight = torch.ones(N, device=device)
        meta = dict(
            phase="uniform",
            alpha=0.0,
            beta=0.0,
            boundary_threshold=boundary_threshold,
            gated=0,
            n_total=N,
            uncertainty_type="none",
            homophily=round(homophily, 4),
            gating_mode="none",
            fiw_weight_mode="uniform",
            w_boundary_mean=float(w_boundary.mean().item()),
            w_degree_mean=float(w_degree.mean().item()),
            w_lhd_mean=float(w_lhd.mean().item()),
        )
        return weight.detach(), meta

    if gating_mode_override == "random":
        n_gate = max(1, int(score_gate.sum().item()))
        perm = torch.randperm(N, device=device)
        gate = torch.zeros(N, dtype=torch.bool, device=device)
        gate[perm[:n_gate]] = True
        gating_mode = "random"

    # ── 3순위 개선: α, β 계산 방식 ablation ──────────────────────────────
    # variance (기본): 분산이 클수록 변별력이 크다는 휴리스틱
    # mutual_info    : 민감속성과의 상호정보량 → 공정성과 직접 연관
    # uniform        : α=β=0.5 기준선 (hand-tuning 없는 하한선)
    alpha_beta_mode = getattr(data, "alpha_beta_mode", "variance")

    if alpha_beta_mode == "uniform":
        alpha, beta = 0.5, 0.5

    elif alpha_beta_mode == "mutual_info":
        # 이산화된 두 신호와 이진 sens 사이의 MI 근사
        # MI(X; S) ≈ H(S) - H(S|X) (discretize X into 10 bins)
        def _approx_mi(signal: torch.Tensor, sens: torch.Tensor) -> float:
            bins  = torch.linspace(0.0, 1.0, 11, device=signal.device)
            bin_idx = torch.bucketize(signal.clamp(0.0, 1.0), bins[1:-1])
            n     = signal.size(0)
            p_s   = (sens == 1).float().mean()
            h_s   = -(p_s * (p_s + 1e-8).log() +
                      (1 - p_s) * (1 - p_s + 1e-8).log()).item()
            h_s_x = 0.0
            for b in bin_idx.unique():
                mask   = (bin_idx == b)
                p_b    = mask.float().mean().item()
                p_s1_b = sens[mask].float().mean().item()
                p_s0_b = 1.0 - p_s1_b
                h_b    = -(p_s1_b * (p_s1_b + 1e-8) +
                           p_s0_b * (p_s0_b + 1e-8)) if p_b > 0 else 0.0
                h_s_x += p_b * float(h_b)
            return max(h_s - h_s_x, 0.0)

        mi_boundary = _approx_mi(w_boundary, data.sens)
        mi_degree   = _approx_mi(w_degree,   data.sens)
        mi_sum      = mi_boundary + mi_degree + 1e-8
        alpha, beta = mi_boundary / mi_sum, mi_degree / mi_sum

    elif alpha_beta_mode == "bnd_only":
        # FIW ablation: boundary 신호만 사용 (beta=0)
        alpha, beta = 1.0, 0.0

    elif alpha_beta_mode == "deg_only":
        # FIW ablation: degree 신호만 사용 (alpha=0)
        alpha, beta = 0.0, 1.0

    elif alpha_beta_mode == "random":
        # Backward-compatible alias. Prefer gating_mode_override="random".
        alpha, beta = 0.5, 0.5
        n_gate = max(1, int(score_gate.sum().item()))
        perm = torch.randperm(N, device=device)
        gate = torch.zeros(N, dtype=torch.bool, device=device)
        gate[perm[:n_gate]] = True
        gating_mode = "random"

    else:   # "variance" — original behaviour
        vars_   = torch.stack([w_boundary.var(), w_degree.var()])
        coefs   = vars_ / (vars_.sum() + 1e-8)
        alpha, beta = coefs[0].item(), coefs[1].item()
    # ─────────────────────────────────────────────────────────────────────

    struct_score = alpha * w_boundary + beta * w_degree
    struct_score = _minmax(struct_score)

    weight = torch.full((N,), _WEIGHT_MIN, device=device)

    if model is None:
        # Warm-up: uncertainty is unavailable, so use structure-only weights.
        if gate.sum() > 0:
            gated_struct = _minmax(struct_score[gate])
            if fiw_weight_mode == "binary_mean":
                tmp = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * gated_struct
                weight[gate] = tmp.mean().detach()
            else:
                weight[gate] = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * gated_struct

        meta = dict(
            phase="structure_only",
            alpha=alpha,
            beta=beta,
            boundary_threshold=boundary_threshold,
            gated=int(gate.sum().item()),
            n_total=N,
            uncertainty_type="none",
            homophily=round(homophily, 4),
            gating_mode=gating_mode,
            fiw_weight_mode=fiw_weight_mode,
            w_boundary_mean=float(w_boundary.mean().item()),
            w_degree_mean=float(w_degree.mean().item()),
            w_lhd_mean=float(w_lhd.mean().item()),
        )
        return weight.detach(), meta

    u_n = _estimate_uncertainty(model, data, uncertainty_type=uncertainty_type)

    # ── Within-gate modulation with learned uncertainty σ(v) ─────────────────
    # When uncertainty_type="dual", u_n = σ(v) from the dual-head UQ head.
    # Unlike entropy (which is a deterministic function of p(v) and may
    # correlate with fairness risk in opposite directions depending on graph
    # topology), σ(v) is trained via a coverage+width loss and captures
    # aleatoric uncertainty independently.  Higher σ(v) → stronger FIW,
    # regardless of Pattern A / Pattern B topology.
    #
    # When uncertainty_type="entropy" or "mc", u_n is a proxy uncertainty.
    # The same formula applies; no direction switching is performed.
    # CF-FV-based direction switching has been intentionally removed because
    # it is unnecessary for learned uncertainty and adds complexity.
    print(
        f"[FIW] unc_type={uncertainty_type}  "
        f"u_mean={float(u_n.mean()):.4f}  "
        f"homophily={homophily:.4f}"
    )

    if fiw_weight_mode == "matched_random_perm":
        # F7: preserve score-based Full-FIW weight distribution on random nodes.
        n_gate = max(1, int(score_gate.sum().item()))
        score_gate_for_weights = score_gate
        if score_gate_for_weights.sum() == 0:
            score_gate_for_weights = gate
        combined_score = struct_score[score_gate_for_weights] * (
            1.0 + fips_lam * u_n[score_gate_for_weights])
        combined_score = _minmax(combined_score)
        full_gate_weight = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * combined_score

        perm_nodes = torch.randperm(N, device=device)[:n_gate]
        perm_weights = torch.randperm(full_gate_weight.numel(), device=device)[:n_gate]
        gate = torch.zeros(N, dtype=torch.bool, device=device)
        gate[perm_nodes] = True
        weight = torch.full((N,), _WEIGHT_MIN, device=device)
        weight[perm_nodes] = full_gate_weight[perm_weights].detach()
        gating_mode = "matched_random_assignment"

    elif gate.sum() > 0:
        if fiw_weight_mode == "struct_only":
            # F4: selection is preserved, uncertainty modulation is removed.
            gated_struct = _minmax(struct_score[gate])
            weight[gate] = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * gated_struct

        elif fiw_weight_mode == "binary_mean":
            # F6: constant weight equal to mean Full-FIW gated weight.
            combined = struct_score[gate] * (1.0 + fips_lam * u_n[gate])
            combined = _minmax(combined)
            full_gate_weight = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * combined
            weight[gate] = full_gate_weight.mean().detach()

        elif fiw_weight_mode == "continuous_uncert":
            # Full FIW: continuous structural score modulated by learned σ(v).
            # u_n = σ(v) when uncertainty_type="dual" (dual-head UQ head).
            # Higher σ(v) → stronger intervention weight, regardless of topology.
            combined = struct_score[gate] * (1.0 + fips_lam * u_n[gate])
            combined = _minmax(combined)
            weight[gate] = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * combined

        else:
            raise ValueError(f"Unsupported fiw_weight_mode: {fiw_weight_mode}")

    meta = dict(
        phase="fiw",
        alpha=alpha,
        beta=beta,
        boundary_threshold=boundary_threshold,
        gated=int(gate.sum().item()),
        n_total=N,
        uncertainty_type=uncertainty_type if fiw_weight_mode != "struct_only" else "none",
        u_mean=round(float(u_n.mean().item()), 4),
        homophily=round(homophily, 4),
        gating_mode=gating_mode,
        fiw_weight_mode=fiw_weight_mode,
        w_boundary_mean=float(w_boundary.mean().item()),
        w_degree_mean=float(w_degree.mean().item()),
        w_lhd_mean=float(w_lhd.mean().item()),
    )
    return weight.detach(), meta


# ============================================================
# Module 2: 3-Level Fairness Loss Components
# ============================================================

def _sensitive_aware_perturb(edge_index, sensitive_attr, struct_drop,
                              mode: str = "drop"):
    """
    Sensitive-Aware Edge Perturbation.

    mode="drop"  (기존): inter-group 엣지를 struct_drop 확률로 제거.
                         bridge 역할 엣지까지 끊을 위험이 있음.
    mode="scale" (4순위 개선): 엣지를 제거하지 않고, inter-group 엣지의
                         어텐션 가중치를 (1 - struct_drop)으로 낮춤.
                         bridge 역할은 유지하면서 불공정 전파는 억제.
                         반환값이 (edge_index, edge_weight) 튜플임에 주의.
    """
    src, dst = edge_index
    device   = edge_index.device
    n_edges  = edge_index.size(1)
    is_inter = (sensitive_attr[src] != sensitive_attr[dst])

    if mode == "scale":
        # 연속 가중치 방식: 엣지를 유지하되 inter-group 엣지 가중치만 감쇠
        edge_weight = torch.ones(n_edges, device=device)
        edge_weight[is_inter] = 1.0 - struct_drop
        return edge_index, edge_weight

    # mode == "drop" — 기존 동작
    intra_drop = struct_drop * _INTRA_DROP_RATIO
    keep = torch.ones(n_edges, dtype=torch.bool, device=device)

    inter_idx = is_inter.nonzero(as_tuple=True)[0]
    if inter_idx.numel() > 0:
        mask = torch.rand(inter_idx.numel(), device=device) < struct_drop
        keep[inter_idx[mask]] = False

    intra_idx = (~is_inter).nonzero(as_tuple=True)[0]
    if intra_idx.numel() > 0 and intra_drop > 0.0:
        mask = torch.rand(intra_idx.numel(), device=device) < intra_drop
        keep[intra_idx[mask]] = False

    if keep.sum() == 0:
        keep[torch.randint(0, n_edges, (1,), device=device)] = True

    return edge_index[:, keep], None


def compute_structural_loss(model, data, node_weight, struct_drop,
                             edge_intervention: str = "drop"):
    """
    edge_intervention="drop"  : 기존 inter-group 엣지 무작위 제거
    edge_intervention="scale" : inter-group 엣지 가중치 감쇠 (bridge 보존)
    """
    idx_fair = _get_fair_idx(data)
    pert_out = _sensitive_aware_perturb(
        data.edge_index, data.sens, struct_drop, mode=edge_intervention
    )
    edge_pert, edge_weight = pert_out

    _, h_orig = model(data, return_hidden=True)

    if edge_intervention == "scale" and edge_weight is not None:
        # edge_weight를 지원하는 backbone은 GCNConv/SAGEConv이다.
        # SGConv는 지원하지 않으므로 drop으로 fallback.
        try:
            _, h_pert = model(data, edge_index=edge_pert,
                              edge_weight=edge_weight, return_hidden=True)
        except TypeError:
            _, h_pert = model(data, edge_index=edge_pert, return_hidden=True)
    else:
        _, h_pert = model(data, edge_index=edge_pert, return_hidden=True)

    w   = node_weight[idx_fair]
    w   = w / (w.mean() + 1e-8)
    mse = ((h_orig[idx_fair] - h_pert[idx_fair]) ** 2).mean(dim=-1)
    return (mse * w).mean() / h_orig.size(-1)


class _MomentNorm(nn.Module):
    def forward(self, z, sens, weight, idx):
        z, sens, weight = z[idx], sens[idx], weight[idx]
        if z.dim() == 1:
            z = z.unsqueeze(1)
        weight = weight / (weight.mean() + 1e-8)
        w      = weight.unsqueeze(1)
        m0, m1 = (sens == 0), (sens == 1)
        if m0.sum() == 0 or m1.sum() == 0:
            return z.new_tensor(0.0)
        z0, w0 = z[m0], w[m0]
        z1, w1 = z[m1], w[m1]
        mean0 = (z0 * w0).sum(0) / (w0.sum() + 1e-8)
        mean1 = (z1 * w1).sum(0) / (w1.sum() + 1e-8)
        var0  = ((z0 - mean0)**2 * w0).sum(0) / (w0.sum() + 1e-8)
        var1  = ((z1 - mean1)**2 * w1).sum(0) / (w1.sum() + 1e-8)
        return torch.abs(mean0 - mean1).mean() + torch.abs(var0 - var1).mean()


class _MMDNorm(nn.Module):
    _BANDWIDTHS = (0.5, 1.0, 2.0)
    _MAX_SAMPLE = 256

    def _rbf(self, X, Y):
        dist_sq = ((X.unsqueeze(1) - Y.unsqueeze(0))**2).sum(-1)
        return sum(torch.exp(-dist_sq / (2 * bw**2))
                   for bw in self._BANDWIDTHS).mean() / len(self._BANDWIDTHS)

    def forward(self, z, sens, weight, idx):
        z, sens, weight = z[idx], sens[idx], weight[idx]
        if z.dim() == 1:
            z = z.unsqueeze(1)
        weight = weight / (weight.mean() + 1e-8)
        m0, m1 = (sens == 0), (sens == 1)
        if m0.sum() == 0 or m1.sum() == 0:
            return z.new_tensor(0.0)
        z0, w0 = z[m0], weight[m0]
        z1, w1 = z[m1], weight[m1]
        if z0.size(0) > self._MAX_SAMPLE:
            p = (w0 / w0.sum()).detach()
            z0 = z0[torch.multinomial(p, self._MAX_SAMPLE, replacement=False)]
        if z1.size(0) > self._MAX_SAMPLE:
            p = (w1 / w1.sum()).detach()
            z1 = z1[torch.multinomial(p, self._MAX_SAMPLE, replacement=False)]
        mmd_sq = self._rbf(z0, z0) - 2 * self._rbf(z0, z1) + self._rbf(z1, z1)
        return mmd_sq.clamp(min=0.0).sqrt()


class RepresentationLoss(nn.Module):
    def __init__(self, mmd_alpha=0.3):
        super().__init__()
        self.mmd_alpha = mmd_alpha
        self._moment   = _MomentNorm()
        self._mmd      = _MMDNorm()

    def forward(self, h, sens, node_weight, idx_fair):
        m = self._moment(h, sens, node_weight, idx_fair)
        d = self._mmd(h, sens, node_weight, idx_fair)
        return (1.0 - self.mmd_alpha) * m + self.mmd_alpha * d


def compute_output_loss(prob, labels, sens, node_weight, idx_fair, dp_eo_ratio=0.3):
    """
    Weighted output-level fairness surrogate (DP + EO).

    Revised default:
        dp_eo_ratio = 0.3
    so EO receives more emphasis than DP by default, reflecting the
    preliminary finding that fairness amplification is more consistently
    visible in EO than in DP.
    """
    prob, labels, sens, node_weight = (
        prob[idx_fair], labels[idx_fair].float(),
        sens[idx_fair], node_weight[idx_fair]
    )
    node_weight = node_weight / (node_weight.mean() + 1e-8)
    m0, m1 = (sens == 0), (sens == 1)
    if m0.sum() == 0 or m1.sum() == 0:
        return prob.new_tensor(0.0)

    def _wm(mask):
        w = node_weight[mask]
        return (prob[mask] * w).sum() / (w.sum() + 1e-8)

    dp_loss = torch.abs(_wm(m0) - _wm(m1))

    eo_gaps = []
    for lv in (0.0, 1.0):
        lm0 = m0 & (labels == lv)
        lm1 = m1 & (labels == lv)
        if lm0.sum() > 0 and lm1.sum() > 0:
            eo_gaps.append(torch.abs(_wm(lm0) - _wm(lm1)))
    eo_loss = torch.stack(eo_gaps).mean() if eo_gaps else prob.new_tensor(0.0)

    total  = dp_loss.detach() + eo_loss.detach() + 1e-8
    adp_dp = dp_loss.detach() / total
    adp_eo = eo_loss.detach() / total
    w_dp   = 0.5 * prob.new_tensor(float(dp_eo_ratio))       + 0.5 * adp_dp
    w_eo   = 0.5 * prob.new_tensor(float(1.0 - dp_eo_ratio)) + 0.5 * adp_eo
    norm   = w_dp + w_eo + 1e-8

    return (w_dp / norm) * dp_loss + (w_eo / norm) * eo_loss


# ============================================================
# Module 3: FairGate — Scale-Calibrated Training
# ============================================================

def _get_fair_idx(data):
    if hasattr(data, "idx_sens_train") and data.idx_sens_train is not None:
        return data.idx_sens_train
    return data.train_mask.nonzero(as_tuple=False).view(-1)


# ── Auto-config 임계값 ────────────────────────────────────────────────────────
_BOUNDARY_RATIO_THR  = 0.5   # < 0.5  : clustered      → mutual_info + scale
_BOUNDARY_SAT_THR    = 0.9   # ≥ 0.9  : saturated      → variance + drop
_DEG_GAP_THR         = 0.2   # > 0.2  : degree-skewed  → variance + drop


def _compute_graph_stats(data) -> dict:
    """
    Auto-config 결정에 필요한 그래프 통계를 계산한다.

    boundary_ratio : inter-group 이웃이 하나라도 있는 노드 비율
                     낮으면 집단이 뭉쳐있고 경계 노드가 병목 역할
    deg_gap        : 두 집단 평균 degree의 정규화된 차이
                     크면 degree 불균형이 불공정성의 주요 구조적 원인
    """
    edge_index = data.edge_index
    src, dst   = edge_index
    N          = data.x.size(0)
    device     = data.x.device
    sens       = data.sens

    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, src, torch.ones(edge_index.size(1), device=device))

    # degree gap
    d0    = deg[sens == 0].mean().item()
    d1    = deg[sens == 1].mean().item()
    deg_gap = abs(d0 - d1) / (d0 + d1 + 1e-8)

    # boundary ratio
    is_inter   = (sens[src] != sens[dst])
    has_inter  = torch.zeros(N, dtype=torch.bool, device=device)
    has_inter[src[is_inter]] = True
    boundary_ratio = has_inter.float().mean().item()

    return {"boundary_ratio": boundary_ratio, "deg_gap": deg_gap}


def _auto_config_from_graph_stats(boundary_ratio: float, deg_gap: float) -> dict:
    """
    boundary_ratio + deg_gap → (alpha_beta_mode, edge_intervention) 자동 결정.

    clustered   (boundary_ratio < 0.5):
        집단이 뭉쳐있고 경계 노드가 소수 → 경계 노드가 불공정 전파 병목.
        MI 기반으로 정확히 타겟팅 + bridge 보존을 위해 scale.
        → mutual_info + scale   (예: Pokec-z/n, Income)

    saturated   (boundary_ratio ≥ 0.9):
        거의 모든 노드가 경계에 있음 → scale로는 개입 효과가 너무 약함.
        경계 신호 자체가 변별력 없으므로 엣지를 적극적으로 제거.
        → variance + drop       (예: German, Bail, NBA)

    degree-skewed  (0.5 ≤ boundary_ratio < 0.9, deg_gap > 0.2):
        경계 노드가 많지만 degree 불균형이 주원인.
        variance가 degree 차이를 잘 포착 + 엣지 제거 효과적.
        → variance + drop       (예: Credit)

    mixed  (0.5 ≤ boundary_ratio < 0.9, deg_gap ≤ 0.2):
        경계 노드가 중간 수준, degree도 균등 → 엣지 감쇠가 안전.
        → variance + scale      (예: NBA 일부)
    """
    if boundary_ratio < _BOUNDARY_RATIO_THR:
        return {
            "alpha_beta_mode" : "mutual_info",
            "edge_intervention": "scale",
            "regime"          : "clustered",
        }
    elif boundary_ratio >= _BOUNDARY_SAT_THR:
        return {
            "alpha_beta_mode" : "variance",
            "edge_intervention": "drop",
            "regime"          : "saturated",
        }
    elif deg_gap > _DEG_GAP_THR:
        return {
            "alpha_beta_mode" : "variance",
            "edge_intervention": "drop",
            "regime"          : "degree-skewed",
        }
    else:
        return {
            "alpha_beta_mode" : "variance",
            "edge_intervention": "scale",
            "regime"          : "mixed",
        }


class FairGate:
    """
    FairGate: Fair GNN with hierarchical FIW and 3-level fairness regularization.

    핵심 파라미터:
        lambda_fair      : 전체 공정성 손실 계수
        sbrs_quantile    : boundary gating quantile
        fips_lam         : gated 노드 불확실성 증폭 계수
        mmd_alpha        : representation loss 혼합 비율
        struct_drop      : inter-group 엣지 제거율
        warm_up          : task-only warm-up 에포크 수
        dp_eo_ratio      : DP/EO 균형 (기본 0.3 → EO 중심)
        recal_interval   : Phase-2에서 스케일 재보정 주기 (5번 한계 반영)
        alpha_beta_mode  : α,β 계산 방식 — "variance"(기본) / "mutual_info" / "uniform"
                           ablation 실험용. 기본값 variance가 가장 안정적.
        edge_intervention: 엣지 개입 방식 — "drop"(기본) / "scale"
                           "scale"은 bridge 엣지 보존 (4번 한계 대응 옵션)
    """

    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        backbone         = "GCN",
        dropout          = 0.5,
        sgc_k            = 2,
        lambda_fair      = 0.05,
        sbrs_quantile    = 0.7,
        fips_lam         = 1.0,
        mmd_alpha        = 0.3,
        struct_drop      = 0.5,
        warm_up          = 200,
        dp_eo_ratio      = 0.3,
        ramp_epochs      = 0,
        uncertainty_type = "entropy",
        recal_interval   = _RECAL_INTERVAL,
        alpha_beta_mode  : str = "variance",   # ablation용 (3번 한계)
        edge_intervention: str = "drop",       # bridge 보존 옵션 (4번 한계)
        gating_mode_override: str = "adaptive",
        fiw_weight_mode: str = "continuous_uncert",
        fiw_adaptive: bool = False,
        adaptive_probe_epochs: int = 20,
        adaptive_eta: float = 1.0,
        adaptive_auc_tol: float = 0.005,
        ablation_mode: str = "full_loss",
        disable_scale_calibration: bool = False,
        # Dual-head uncertainty learning. Keep lambda_uq=0.0 to reproduce
        # the original model. Set uncertainty_type="dual" and lambda_uq>0
        # to learn an interval-width uncertainty head.
        lambda_uq: float = 0.0,
        uq_width_penalty: float = 0.05,
        use_uq_weighted_loss: bool = False,
    ):
        assert backbone in ("GCN", "GraphSAGE", "SGC"), \
            f"Unsupported backbone: {backbone}"
        assert alpha_beta_mode in (
            "variance", "mutual_info", "uniform",
            "bnd_only", "deg_only", "random",
        ), f"alpha_beta_mode must be variance/mutual_info/uniform/bnd_only/deg_only/random, got: {alpha_beta_mode}"
        assert edge_intervention in ("drop", "scale"), \
            f"edge_intervention must be 'drop' or 'scale', got: {edge_intervention}"
        assert gating_mode_override in (
            "none", "score", "adaptive", "boundary", "degree",
            "boundary_degree", "loss", "random"
        ), f"Unsupported gating_mode_override: {gating_mode_override}"
        assert fiw_weight_mode in (
            "uniform", "struct_only", "continuous_uncert",
            "binary_mean", "matched_random_perm"
        ), f"Unsupported fiw_weight_mode: {fiw_weight_mode}"
        assert ablation_mode in (
            "none", "struct_only", "struct_rep", "struct_out",
            "rep_out", "full_loss"
        ), f"Unsupported ablation_mode: {ablation_mode}"

        self.device            = device
        self.backbone_name     = backbone
        self.lambda_fair       = lambda_fair
        self.sbrs_quantile     = sbrs_quantile
        self.fips_lam          = fips_lam
        self.mmd_alpha         = mmd_alpha
        self.struct_drop       = struct_drop
        self.warm_up           = warm_up
        self.dp_eo_ratio       = dp_eo_ratio
        self.ramp_epochs       = ramp_epochs
        self.uncertainty_type  = uncertainty_type
        self.recal_interval    = recal_interval
        self.alpha_beta_mode   = alpha_beta_mode
        self.edge_intervention = edge_intervention
        self.gating_mode_override = gating_mode_override
        self.fiw_weight_mode = fiw_weight_mode
        self.fiw_adaptive = fiw_adaptive
        self.adaptive_probe_epochs = int(adaptive_probe_epochs)
        self.adaptive_eta = float(adaptive_eta)
        self.adaptive_auc_tol = float(adaptive_auc_tol)
        self.ablation_mode = ablation_mode
        self.disable_scale_calibration = bool(disable_scale_calibration)
        self.lambda_uq = float(lambda_uq)
        self.uq_width_penalty = float(uq_width_penalty)
        self.use_uq_weighted_loss = bool(use_uq_weighted_loss)
        self._adaptive_choice = None

        self.name = f"{backbone}/FairGate"

        self.model      = _build_backbone(backbone, in_feats, h_feats,
                                          dropout=dropout, sgc_k=sgc_k).to(device)
        self._rep_loss  = RepresentationLoss(mmd_alpha=mmd_alpha)
        self._scales    = {"struct": 1.0, "rep": 1.0, "out": 1.0}
        self._node_w    = None

    def _optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.model.parameters(),
                                lr=lr, weight_decay=weight_decay or 0.0)

    def _init_weights_warmup(self, data):
        self._node_w, meta = compute_fiw_weights(
            data,
            model=None,
            sbrs_quantile=self.sbrs_quantile,
            fips_lam=self.fips_lam,
            uncertainty_type=self.uncertainty_type,
            gating_mode_override=self.gating_mode_override,
            fiw_weight_mode=self.fiw_weight_mode,
        )
        _log_fiw(self.name, "Phase 1 (structure-only)", meta, self._node_w)

    def _update_weights_fiw(self, data):
        self._node_w, meta = compute_fiw_weights(
            data,
            model=self.model,
            sbrs_quantile=self.sbrs_quantile,
            fips_lam=self.fips_lam,
            uncertainty_type=self.uncertainty_type,
            gating_mode_override=self.gating_mode_override,
            fiw_weight_mode=self.fiw_weight_mode,
        )
        _log_fiw(self.name, "Phase 2 (hierarchical FIW)", meta, self._node_w)

    def _refresh_weights(self, data, epoch):
        new_w, meta = compute_fiw_weights(
            data,
            model=self.model,
            sbrs_quantile=self.sbrs_quantile,
            fips_lam=self.fips_lam,
            uncertainty_type=self.uncertainty_type,
            gating_mode_override=self.gating_mode_override,
            fiw_weight_mode=self.fiw_weight_mode,
        )
        if torch.isnan(new_w).any():
            print(f"[{self.name}] [Refresh@{epoch}] NaN detected — skipping update")
            return
        self._node_w = new_w
        print(
            f"[{self.name}] [Refresh@{epoch}] "
            f"gated={meta['gated']}/{meta['n_total']} | "
            f"w_mean={self._node_w.mean():.3f}"
        )

    @torch.no_grad()
    def _calibrate_scales(self, data, criterion):
        self.model.eval()
        labels    = data.y.float()
        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_fair  = _get_fair_idx(data)
        sens      = data.sens

        out, h    = self.model(data, return_hidden=True)
        task_val  = criterion(out[idx_train], labels[idx_train]).item()

        def _s(loss_fn):
            v = loss_fn().item()
            if not (v > 1e-6):          # NaN / 0 / inf 모두 처리
                return 1.0
            raw = task_val / (v + 1e-8)
            return float(min(raw, 100.0))   # 상한선 100배

        self._scales["struct"] = _s(lambda: compute_structural_loss(
            self.model, data, self._node_w, self.struct_drop,
            edge_intervention=self.edge_intervention))
        self._scales["rep"] = _s(lambda: self._rep_loss(
            h, sens, self._node_w, idx_fair))
        self._scales["out"] = _s(lambda: compute_output_loss(
            torch.sigmoid(out), labels, sens,
            self._node_w, idx_fair, self.dp_eo_ratio))

        self.model.train()
        print(
            f"[{self.name}] Scale calibrated | "
            f"task={task_val:.4f} | "
            f"struct×{self._scales['struct']:.2f} "
            f"rep×{self._scales['rep']:.2f} "
            f"out×{self._scales['out']:.2f}"
        )

    def _train_step(self, data, optimizer, criterion, lam):
        self.model.train()
        optimizer.zero_grad()

        labels    = data.y.float()
        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_fair  = _get_fair_idx(data)
        sens      = data.sens

        out, h    = self.model(data, return_hidden=True)
        task_loss = criterion(out[idx_train], labels[idx_train])

        # Dual-head UQ is an auxiliary objective, not a direct FIW direction
        # estimator. This keeps uncertainty logically consistent with the
        # analysis: uncertainty is complementary and topology-dependent, so it
        # should be learned/calibrated rather than treated as a universal
        # monotonic fairness-risk multiplier.
        uq_loss = task_loss.new_tensor(0.0)
        uq_info = {"uq_cov": 0.0, "uq_width": 0.0}
        if self.lambda_uq > 0.0:
            uq_node_weight = self._node_w if self.use_uq_weighted_loss else None
            uq_loss, uq_info = compute_dual_uq_loss(
                self.model, data, idx_train=idx_train,
                width_penalty=self.uq_width_penalty,
                node_weight=uq_node_weight,
            )

        struct_loss = rep_loss = out_loss = task_loss.new_tensor(0.0)

        use_struct = self.ablation_mode in ("full_loss", "struct_only", "struct_rep", "struct_out")
        use_rep    = self.ablation_mode in ("full_loss", "struct_rep", "rep_out")
        use_out    = self.ablation_mode in ("full_loss", "struct_out", "rep_out")

        if lam > 0.0 and self.ablation_mode != "none":
            if use_struct:
                struct_loss = compute_structural_loss(
                    self.model, data, self._node_w, self.struct_drop,
                    edge_intervention=self.edge_intervention)
            if use_rep:
                rep_loss = self._rep_loss(h, sens, self._node_w, idx_fair)
            if use_out:
                out_loss = compute_output_loss(
                    torch.sigmoid(out), labels, sens,
                    self._node_w, idx_fair, self.dp_eo_ratio)

        total = task_loss + self.lambda_uq * uq_loss + lam * (
            self._scales["struct"] * struct_loss
            + self._scales["rep"]  * rep_loss
            + self._scales["out"]  * out_loss
        )

        total.backward()
        optimizer.step()

        return dict(
            total=float(total),
            task=float(task_loss),
            struct=float(struct_loss),
            rep=float(rep_loss),
            out=float(out_loss),
            uq=float(uq_loss),
            uq_cov=uq_info["uq_cov"],
            uq_width=uq_info["uq_width"],
        )

    def _metric_auc(self, result):
        for key in ("roc_auc", "auc", "AUC", "roc_auc_mean"):
            if key in result and result[key] is not None:
                return float(result[key])
        return float(result.get("acc", 0.0))

    def _metric_fair(self, result):
        return abs(float(result.get("dp", 0.0))) + abs(float(result.get("eo", 0.0)))

    def _adaptive_candidate_score(self, result, auc_ref):
        auc = self._metric_auc(result)
        fair = self._metric_fair(result)
        auc_penalty = max(0.0, auc_ref - auc - self.adaptive_auc_tol)
        return fair + self.adaptive_eta * auc_penalty

    def _set_fiw_candidate(self, cand):
        self.gating_mode_override = cand["gating"]
        self.alpha_beta_mode = cand["alpha_beta"]
        self.fiw_weight_mode = cand["weight"]

    def _adaptive_candidates(self, data):
        """Candidate FIW policies for validation-based adaptive selection."""
        return [
            {"name": "boundary", "gating": "boundary", "alpha_beta": "bnd_only", "weight": "continuous_uncert"},
            {"name": "degree", "gating": "degree", "alpha_beta": "deg_only", "weight": "continuous_uncert"},
            {"name": "no_uncert", "gating": "adaptive", "alpha_beta": "variance", "weight": "struct_only"},
            {"name": "full", "gating": "adaptive", "alpha_beta": "variance", "weight": "continuous_uncert"},
        ]

    def _select_adaptive_fiw_mode(self, data, lr, weight_decay, criterion, verbose=True):
        """
        Select a FIW policy on the validation split.

        Each candidate starts from the same warm-up checkpoint, is trained for a
        small number of probe epochs, and is scored by validation fairness with
        an AUC-preserving penalty. The warm-up checkpoint is restored before
        main Phase-2 training.
        """
        if self.adaptive_probe_epochs <= 0 or self.lambda_fair <= 0.0:
            return

        warm_state = copy.deepcopy(self.model.state_dict())
        base_result = evaluate_pyg_model(self.model, data, split="val", task_type="classification")
        auc_ref = self._metric_auc(base_result)

        old_gate = self.gating_mode_override
        old_ab = self.alpha_beta_mode
        old_wmode = self.fiw_weight_mode
        old_scales = copy.deepcopy(self._scales)
        old_node_w = self._node_w.clone() if self._node_w is not None else None

        best = None
        logs = []

        for cand in self._adaptive_candidates(data):
            self.model.load_state_dict(warm_state)
            self._set_fiw_candidate(cand)
            data.alpha_beta_mode = self.alpha_beta_mode

            self._node_w, meta = compute_fiw_weights(
                data, model=self.model, sbrs_quantile=self.sbrs_quantile,
                fips_lam=self.fips_lam, uncertainty_type=self.uncertainty_type,
                gating_mode_override=self.gating_mode_override,
                fiw_weight_mode=self.fiw_weight_mode,
            )
            if self.disable_scale_calibration:
                self._scales = {"struct": 1.0, "rep": 1.0, "out": 1.0}
            else:
                self._calibrate_scales(data, criterion)

            opt = self._optimizer(lr, weight_decay)
            for _ in range(self.adaptive_probe_epochs):
                self._train_step(data, opt, criterion, lam=self.lambda_fair)

            val_result = evaluate_pyg_model(self.model, data, split="val", task_type="classification")
            score = self._adaptive_candidate_score(val_result, auc_ref)
            rec = dict(
                cand=cand, score=score,
                auc=self._metric_auc(val_result),
                fair=self._metric_fair(val_result),
                gated=meta.get("gated", 0),
                gating_mode=meta.get("gating_mode", "?"),
            )
            logs.append(rec)
            if best is None or score < best["score"]:
                best = rec

        self.model.load_state_dict(warm_state)
        self._scales = old_scales
        self._node_w = old_node_w

        if best is not None:
            self._set_fiw_candidate(best["cand"])
            self._adaptive_choice = best
        else:
            self.gating_mode_override = old_gate
            self.alpha_beta_mode = old_ab
            self.fiw_weight_mode = old_wmode

        data.alpha_beta_mode = self.alpha_beta_mode

        if verbose:
            print(f"[{self.name}] Adaptive FIW selection | auc_ref={auc_ref:.4f}")
            for r in logs:
                print(
                    f"  - {r['cand']['name']:<10} score={r['score']:.4f} "
                    f"auc={r['auc']:.4f} fair={r['fair']:.4f} "
                    f"gate={r['gating_mode']} gated={r['gated']}"
                )
            if best is not None:
                print(
                    f"[{self.name}] Selected adaptive FIW: {best['cand']['name']} "
                    f"(gating={self.gating_mode_override}, "
                    f"alpha_beta={self.alpha_beta_mode}, weight={self.fiw_weight_mode})"
                )

    def _val_score(self, result):
        acc = float(result.get("acc", 0.0))
        dp  = abs(float(result.get("dp", 0.0)))
        eo  = abs(float(result.get("eo", 0.0)))
        return acc - self.dp_eo_ratio * dp - (1.0 - self.dp_eo_ratio) * eo

    def fit(self, data, epochs=1000, lr=1e-3, weight_decay=0.0,
            patience=100, verbose=True, print_interval=50):

        optimizer = self._optimizer(lr, weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # ── 그래프 통계 진단 로그 (성능에 영향 없음) ─────────────────────────
        homophily      = _compute_edge_homophily(data)
        stats          = _compute_graph_stats(data)
        boundary_ratio = stats["boundary_ratio"]
        deg_gap        = stats["deg_gap"]
        # data에 alpha_beta_mode 부착 → compute_fiw_weights 내부에서 읽힘
        data.alpha_beta_mode = self.alpha_beta_mode
        if verbose:
            print(
                f"[{self.name}] Config | "
                f"homophily={homophily:.4f}  "
                f"boundary_ratio={boundary_ratio:.4f}  "
                f"deg_gap={deg_gap:.4f} | "
                f"alpha_beta={self.alpha_beta_mode}  "
                f"edge={self.edge_intervention}  "
                f"gate_override={self.gating_mode_override}  "
                f"weight_mode={self.fiw_weight_mode}  "
                f"adaptive={self.fiw_adaptive}  "
                f"probe_epochs={self.adaptive_probe_epochs}  "
                f"ablation={self.ablation_mode}  "
                f"disable_scale={self.disable_scale_calibration}  "
                f"recal_interval={self.recal_interval}"
            )
        # ─────────────────────────────────────────────────────────────────────

        # Phase 1: structure-only warm-up
        self._init_weights_warmup(data)
        if verbose:
            print(f"[{self.name}] Phase 1: warm-up {self.warm_up} epochs...")
        for _ in range(self.warm_up):
            self._train_step(data, optimizer, criterion, lam=0.0)

        # Optional validation-based Adaptive FIW policy selection.
        if self.fiw_adaptive and self.lambda_fair > 0.0:
            if verbose:
                print(f"[{self.name}] Selecting adaptive FIW policy...")
            self._select_adaptive_fiw_mode(data, lr, weight_decay, criterion, verbose=verbose)

        # Transition
        if verbose:
            print(f"[{self.name}] Updating FIW weights & calibrating scales...")
        data.alpha_beta_mode = self.alpha_beta_mode
        self._update_weights_fiw(data)
        if self.disable_scale_calibration:
            self._scales = {"struct": 1.0, "rep": 1.0, "out": 1.0}
            if verbose:
                print(f"[{self.name}] Scale calibration disabled; using unit loss scales.")
        else:
            self._calibrate_scales(data, criterion)
        self.model.train()

        # Phase 2
        best_score = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        counter    = 0
        remaining  = epochs - self.warm_up

        if verbose:
            print(f"[{self.name}] Phase 2: main training {remaining} epochs...")

        for epoch in range(remaining):
            if epoch > 0 and epoch % _REFRESH_INTERVAL == 0:
                self._refresh_weights(data, epoch + self.warm_up + 1)
                self.model.train()

            # ── Periodic scale recalibration (1순위 개선) ──────────────────
            # warm-up 직후 단 한 번의 보정이 아니라, Phase 2 전반에 걸쳐
            # recal_interval 마다 손실 간 상대 비율을 재조정한다.
            # FIW 갱신과 인터리빙하여 오버헤드를 분산시킨다.
            if (
                (not self.disable_scale_calibration)
                and self.recal_interval
                and self.recal_interval > 0
                and epoch > 0
                and epoch % self.recal_interval == 0
            ):
                if verbose:
                    print(f"[{self.name}] Periodic recalibration @ "
                          f"epoch {epoch + self.warm_up + 1}...")
                self._calibrate_scales(data, criterion)
                self.model.train()
            # ───────────────────────────────────────────────────────────────

            if self.ramp_epochs > 0 and epoch < self.ramp_epochs:
                lam = self.lambda_fair * (epoch + 1) / self.ramp_epochs
            else:
                lam = self.lambda_fair

            info       = self._train_step(data, optimizer, criterion, lam)
            val_result = evaluate_pyg_model(
                self.model, data, split="val", task_type="classification")
            score      = self._val_score(val_result)

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())
                counter    = 0
            else:
                counter += 1

            if verbose and (epoch == 0 or (epoch + 1) % print_interval == 0):
                tr = evaluate_pyg_model(
                    self.model, data, split="train", task_type="classification")
                print(
                    f"[{self.name}] Ep {epoch + self.warm_up + 1:04d} | "
                    f"Total {info['total']:.4f} "
                    f"Task {info['task']:.4f} "
                    f"Struct {info['struct']:.4f} "
                    f"Rep {info['rep']:.4f} "
                    f"Out {info['out']:.4f} "
                    f"UQ {info.get('uq', 0.0):.4f} "
                    f"(cov {info.get('uq_cov', 0.0):.4f}, width {info.get('uq_width', 0.0):.4f}) | "
                    f"Train {tr} | Val {val_result} | Score {score:.4f}"
                )

            if counter >= patience:
                if verbose:
                    print(f"[{self.name}] Early stopping at "
                          f"epoch {epoch + self.warm_up + 1}.")
                break

        self.model.load_state_dict(best_state)
        if verbose:
            print(f"[{self.name}] Done. Best val score: {best_score:.4f}")

    @torch.no_grad()
    def predict_uncertainty(self, data):
        """Return learned dual-head probability-interval width in [0, 1]."""
        self.model.eval()
        return _estimate_dual_uncertainty(self.model, data)

    @torch.no_grad()
    def evaluate(self, data, split="test"):
        return evaluate_pyg_model(
            self.model, data, split=split, task_type="classification")

    @torch.no_grad()
    def predict_proba(self, data):
        self.model.eval()
        out = self.model(data)
        if isinstance(out, tuple):
            out = out[0]
        return torch.sigmoid(out.view(-1))


def _log_fiw(name, phase, meta, weight):
    gate_pct = 100.0 * meta["gated"] / max(meta["n_total"], 1)
    print(
        f"[{name}] {phase} | "
        f"homophily={meta.get('homophily', '?'):.3f} "
        f"gating={meta.get('gating_mode', '?')} | "
        f"alpha(boundary)={meta['alpha']:.3f} "
        f"beta(degree)={meta['beta']:.3f} | "
        f"thr_boundary={meta['boundary_threshold']:.3f} | "
        f"gated={meta['gated']}/{meta['n_total']} ({gate_pct:.1f}%) | "
        f"unc={meta['uncertainty_type']} | "
        f"wmode={meta.get('fiw_weight_mode', '?')} | "
        f"w_mean={weight.mean():.3f} w_std={weight.std():.3f}"
    )