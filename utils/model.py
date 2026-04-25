"""
FairGate — Fair Graph Neural Network with Hierarchical Fairness Intervention

Revised design (aligned with the updated preliminary analysis):
    Module 1: Hierarchical Fairness Intervention Weight (FIW)
              - boundary-first gating
              - degree as a secondary structural propagation axis
              - predictive entropy as the default uncertainty signal
    Module 2: 3-Level Fairness Loss
              L = L_task + λ_fair · (L_struct + L_rep + L_out)
    Module 3: Scale-calibrated Training Loop
              warm-up → FIW update → auto scale calibration → λ-ramp → early stopping

Key design changes from the previous version:
    1) w_boundary and w_lhd are not treated as independent core FIW inputs.
       On binary, high-homophily graphs they are often empirically redundant.
       We therefore use w_boundary as the representative boundary-type signal
       and keep w_lhd only for diagnostics.
    2) FIW is hierarchical:
         (a) gate nodes by high boundary risk,
         (b) rank gated nodes using a boundary-degree structural score,
         (c) modulate intervention strength using uncertainty.
    3) Predictive entropy is the default uncertainty source, because it matches
       the pre-method analysis used to motivate the method.
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
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, h_feats)
        self.conv2   = GCNConv(h_feats, 1)
        self.dropout = dropout

    def forward(self, data, edge_index=None, edge_weight=None, return_hidden=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h   = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index, edge_weight=edge_weight).view(-1)
        return (out, h) if return_hidden else out


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = SAGEConv(in_feats, h_feats)
        self.conv2   = SAGEConv(h_feats, 1)
        self.dropout = dropout

    def forward(self, data, edge_index=None, edge_weight=None, return_hidden=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        # SAGEConv does not accept edge_weight → ignore silently
        h   = F.relu(self.conv1(x, edge_index))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)
        return (out, h) if return_hidden else out


class SGC(nn.Module):
    def __init__(self, in_feats, sgc_k=2):
        super().__init__()
        self.conv = SGConv(in_feats, 1, K=sgc_k)

    def forward(self, data, edge_index=None, edge_weight=None, return_hidden=False):
        # SGConv does not support edge_weight; scale mode falls back to drop in
        # compute_structural_loss automatically via the TypeError catch.
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        out = self.conv(x, edge_index).view(-1)
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
    Per-node fairness-relevant risk signal: BCELoss(node) weighted by
    cross-group neighbor fraction.

    On heterophilic graphs, nodes whose individual prediction loss is large
    AND whose inter-group exposure is high are the ones where gating matters.
    This replaces pure boundary proximity as the primary discriminator.
    """
    was_training = model.training
    model.eval()

    out = model(data)
    if isinstance(out, tuple):
        out = out[0]

    labels = data.y.float()
    # per-node BCE
    node_loss = F.binary_cross_entropy_with_logits(
        out.view(-1), labels, reduction="none"
    )

    # inter-group neighbor fraction (same as w_boundary numerator)
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

    signal = node_loss * inter_frac
    if was_training:
        model.train()
    return _minmax(signal).detach()


def _estimate_uncertainty(model, data, uncertainty_type="entropy"):
    if uncertainty_type == "entropy":
        return _estimate_entropy_uncertainty(model, data)
    elif uncertainty_type == "mc":
        return _estimate_mc_uncertainty(model, data)
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
        3) Modulate gated nodes with uncertainty:
           FIW(v) ∝ s_struct(v) * (1 + fips_lam * U(v))

    Graph regime detection:
        boundary_ratio ≥ 0.9 (saturated): w_boundary 변별력 붕괴
            → loss_signal / w_degree 기반 gating으로 전환
        homophily < 0.5 (heterophilic): w_boundary 단독 신호 부정확
            → w_boundary + loss_signal 혼합
        그 외: w_boundary 기반 gating (기본)
    """
    N      = data.x.size(0)
    device = data.x.device

    sig = _compute_structural_signals(data)
    w_degree   = sig["w_degree"]
    w_boundary = sig["w_boundary"]
    w_lhd      = sig["w_lhd"]  # diagnostics only

    # ── 2순위 개선: Graph-regime-adaptive gating ──────────────────────────
    # 그래프 구조 특성에 따라 gating 신호를 적응적으로 결정한다.
    #
    # [heterophilic]  homophily < 0.5:
    #     w_boundary가 실제 불공정 위험을 잘못 표현할 수 있음.
    #     per-node loss × inter-group 노출 신호를 혼합하여 보완.
    #
    # [saturated]  boundary_ratio ≥ 0.9:
    #     거의 모든 노드가 경계에 있어 w_boundary의 변별력이 붕괴됨.
    #     → Phase 2: loss_signal 기반으로 전환 (실제 예측 오류가 큰 노드 우선)
    #     → warm-up: w_degree 기반 fallback (degree 불균형이 구조적 대리 신호)
    #
    # [그 외]:  w_boundary 그대로 사용.
    homophily      = _compute_edge_homophily(data)
    boundary_ratio = _compute_graph_stats(data)["boundary_ratio"]

    is_heterophilic = homophily < _LOW_HOMOPHILY_THR
    is_saturated    = boundary_ratio >= _BOUNDARY_SAT_THR

    if is_saturated:
        # w_boundary 변별력 붕괴 → 대체 신호 사용
        if model is not None:
            # Phase 2: per-node 예측 손실 기반 gating
            loss_signal   = _compute_loss_based_signal(model, data)
            gating_signal = _minmax(loss_signal)
            gating_mode   = "saturated_loss"
        else:
            # warm-up: degree 기반 fallback
            gating_signal = w_degree
            gating_mode   = "saturated_degree"
    elif is_heterophilic and model is not None:
        loss_signal = _compute_loss_based_signal(model, data)
        # 혼합 비율: homophily가 낮을수록 loss_signal 비중 증가
        mix = 1.0 - homophily / _LOW_HOMOPHILY_THR   # ∈ (0, 1]
        gating_signal = (1.0 - mix) * w_boundary + mix * loss_signal
        gating_signal = _minmax(gating_signal)
        gating_mode   = "heterophilic"
    else:
        gating_signal = w_boundary
        gating_mode   = "boundary"
    # ─────────────────────────────────────────────────────────────────────

    boundary_threshold = float(torch.quantile(gating_signal, sbrs_quantile).item())
    gate = gating_signal >= boundary_threshold

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
        # FIW ablation: 동일 비율 랜덤 선택
        # alpha/beta는 사용되지 않음 (gate를 아래에서 랜덤으로 덮어씀)
        alpha, beta = 0.5, 0.5
        perm   = torch.randperm(N, device=device)
        n_gate = max(1, int(N * (1.0 - sbrs_quantile)))
        gate   = torch.zeros(N, dtype=torch.bool, device=device)
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
        # Warm-up: structure-only FIW
        if gate.sum() > 0:
            gated_struct = struct_score[gate]
            gated_struct = _minmax(gated_struct)
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
            w_boundary_mean=float(w_boundary.mean().item()),
            w_degree_mean=float(w_degree.mean().item()),
            w_lhd_mean=float(w_lhd.mean().item()),
        )
        return weight.detach(), meta

    u_n = _estimate_uncertainty(model, data, uncertainty_type=uncertainty_type)

    if gate.sum() > 0:
        if fips_lam == 0.0:
            # F4: gating은 수행하되 uncertainty 없이 struct_score만 사용
            # (fips_lam=0 + gating → uncertainty modulation 기여만 제거)
            gated_struct = struct_score[gate]
            gated_struct = _minmax(gated_struct)
            weight[gate] = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * gated_struct
        else:
            # Full FIW: struct_score × (1 + fips_lam × uncertainty)
            combined = struct_score[gate] * (1.0 + fips_lam * u_n[gate])
            combined = _minmax(combined)
            weight[gate] = _WEIGHT_MIN + (_WEIGHT_MAX - _WEIGHT_MIN) * combined

    meta = dict(
        phase="fiw",
        alpha=alpha,
        beta=beta,
        boundary_threshold=boundary_threshold,
        gated=int(gate.sum().item()),
        n_total=N,
        uncertainty_type=uncertainty_type,
        homophily=round(homophily, 4),
        gating_mode=gating_mode,
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
    ):
        assert backbone in ("GCN", "GraphSAGE", "SGC"), \
            f"Unsupported backbone: {backbone}"
        assert alpha_beta_mode in (
            "variance", "mutual_info", "uniform",
            "bnd_only", "deg_only", "random",
        ), f"alpha_beta_mode must be variance/mutual_info/uniform/bnd_only/deg_only/random, got: {alpha_beta_mode}"
        assert edge_intervention in ("drop", "scale"), \
            f"edge_intervention must be 'drop' or 'scale', got: {edge_intervention}"

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
        )
        _log_fiw(self.name, "Phase 1 (structure-only)", meta, self._node_w)

    def _update_weights_fiw(self, data):
        self._node_w, meta = compute_fiw_weights(
            data,
            model=self.model,
            sbrs_quantile=self.sbrs_quantile,
            fips_lam=self.fips_lam,
            uncertainty_type=self.uncertainty_type,
        )
        _log_fiw(self.name, "Phase 2 (hierarchical FIW)", meta, self._node_w)

    def _refresh_weights(self, data, epoch):
        new_w, meta = compute_fiw_weights(
            data,
            model=self.model,
            sbrs_quantile=self.sbrs_quantile,
            fips_lam=self.fips_lam,
            uncertainty_type=self.uncertainty_type,
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

        struct_loss = rep_loss = out_loss = task_loss.new_tensor(0.0)

        if lam > 0.0:
            struct_loss = compute_structural_loss(
                self.model, data, self._node_w, self.struct_drop,
                edge_intervention=self.edge_intervention)
            rep_loss    = self._rep_loss(h, sens, self._node_w, idx_fair)
            out_loss    = compute_output_loss(
                torch.sigmoid(out), labels, sens,
                self._node_w, idx_fair, self.dp_eo_ratio)

        total = task_loss + lam * (
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
                f"recal_interval={self.recal_interval}"
            )
        # ─────────────────────────────────────────────────────────────────────

        # Phase 1: structure-only warm-up
        self._init_weights_warmup(data)
        if verbose:
            print(f"[{self.name}] Phase 1: warm-up {self.warm_up} epochs...")
        for _ in range(self.warm_up):
            self._train_step(data, optimizer, criterion, lam=0.0)

        # Transition
        if verbose:
            print(f"[{self.name}] Updating FIW weights & calibrating scales...")
        self._update_weights_fiw(data)
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
            if epoch > 0 and epoch % self.recal_interval == 0:
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
                    f"Out {info['out']:.4f} | "
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
        f"w_mean={weight.mean():.3f} w_std={weight.std():.3f}"
    )