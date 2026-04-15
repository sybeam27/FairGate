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


# ============================================================
# Backbone GNN Models
# ============================================================

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = GCNConv(in_feats, h_feats)
        self.conv2   = GCNConv(h_feats, 1)
        self.dropout = dropout

    def forward(self, data, edge_index=None, return_hidden=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h   = F.relu(self.conv1(x, edge_index))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)
        return (out, h) if return_hidden else out


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=0.5):
        super().__init__()
        self.conv1   = SAGEConv(in_feats, h_feats)
        self.conv2   = SAGEConv(h_feats, 1)
        self.dropout = dropout

    def forward(self, data, edge_index=None, return_hidden=False):
        x          = data.x
        edge_index = data.edge_index if edge_index is None else edge_index
        h   = F.relu(self.conv1(x, edge_index))
        h   = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, edge_index).view(-1)
        return (out, h) if return_hidden else out


class SGC(nn.Module):
    def __init__(self, in_feats, sgc_k=2):
        super().__init__()
        self.conv = SGConv(in_feats, 1, K=sgc_k)

    def forward(self, data, edge_index=None, return_hidden=False):
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


def _estimate_uncertainty(model, data, uncertainty_type="entropy"):
    if uncertainty_type == "entropy":
        return _estimate_entropy_uncertainty(model, data)
    elif uncertainty_type == "mc":
        return _estimate_mc_uncertainty(model, data)
    else:
        raise ValueError(f"Unsupported uncertainty_type: {uncertainty_type}")


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
        1) Gate nodes by high boundary risk:
           G = {v | w_boundary(v) >= quantile_q}
        2) Inside the gate, compute a structural priority score using
           data-driven (variance-based) coefficients:
           alpha = Var(w_boundary) / (Var(w_boundary) + Var(w_degree))
           beta  = Var(w_degree)   / (Var(w_boundary) + Var(w_degree))
           s_struct(v) = alpha * w_boundary(v) + beta * w_degree(v)
           This replaces the hand-tuned boundary_weight=0.85 with a
           principled, data-adaptive weighting that reflects which signal
           is more informative on the given graph.
        3) Modulate gated nodes with uncertainty:
           FIW(v) ∝ s_struct(v) * (1 + fips_lam * U(v))

    For backward compatibility, the argument name `sbrs_quantile` is retained,
    but it is now used as the boundary-gating quantile.
    """
    N      = data.x.size(0)
    device = data.x.device

    sig = _compute_structural_signals(data)
    w_degree   = sig["w_degree"]
    w_boundary = sig["w_boundary"]
    w_lhd      = sig["w_lhd"]  # diagnostics only

    boundary_threshold = float(torch.quantile(w_boundary, sbrs_quantile).item())
    gate = w_boundary >= boundary_threshold

    # Data-driven (variance-based) structural priority coefficients.
    # alpha reflects how much w_boundary varies across nodes;
    # beta reflects the same for w_degree.
    # Signals with higher variance carry more discriminative information,
    # so they receive proportionally larger weight — no manual tuning required.
    vars_   = torch.stack([w_boundary.var(), w_degree.var()])
    coefs   = vars_ / (vars_.sum() + 1e-8)
    alpha, beta = coefs[0].item(), coefs[1].item()

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
            w_boundary_mean=float(w_boundary.mean().item()),
            w_degree_mean=float(w_degree.mean().item()),
            w_lhd_mean=float(w_lhd.mean().item()),
        )
        return weight.detach(), meta

    u_n = _estimate_uncertainty(model, data, uncertainty_type=uncertainty_type)

    if gate.sum() > 0:
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
        w_boundary_mean=float(w_boundary.mean().item()),
        w_degree_mean=float(w_degree.mean().item()),
        w_lhd_mean=float(w_lhd.mean().item()),
    )
    return weight.detach(), meta


# ============================================================
# Module 2: 3-Level Fairness Loss Components
# ============================================================

def _sensitive_aware_perturb(edge_index, sensitive_attr, struct_drop):
    """
    Sensitive-Aware Edge Perturbation.

    Drops inter-group edges at rate `struct_drop` and intra-group edges at
    rate `struct_drop × _INTRA_DROP_RATIO`.
    """
    src, dst    = edge_index
    device      = edge_index.device
    n_edges     = edge_index.size(1)
    is_inter    = (sensitive_attr[src] != sensitive_attr[dst])
    intra_drop  = struct_drop * _INTRA_DROP_RATIO

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

    return edge_index[:, keep]


def compute_structural_loss(model, data, node_weight, struct_drop):
    idx_fair  = _get_fair_idx(data)
    edge_pert = _sensitive_aware_perturb(data.edge_index, data.sens, struct_drop)

    _, h_orig = model(data, return_hidden=True)
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


class FairGate:
    """
    FairGate: Fair GNN with hierarchical FIW and 3-level fairness regularization.

    Core hyperparameters:
        lambda_fair     : overall fairness loss coefficient
        sbrs_quantile   : boundary gating quantile (kept name for compatibility)
        fips_lam        : uncertainty amplification inside gated nodes
        mmd_alpha       : representation loss mixing ratio
        struct_drop     : inter-group edge drop rate
        warm_up         : task-only warm-up epochs
        dp_eo_ratio     : DP/EO balance (default 0.3 → EO-oriented)
        uncertainty_type: "entropy" (default) or "mc"

    Structural priority coefficients (alpha, beta) are computed automatically
    from signal variances — no manual boundary_weight tuning needed.
    """

    def __init__(
        self,
        in_feats,
        h_feats,
        device,
        backbone        = "GCN",
        dropout         = 0.5,
        sgc_k           = 2,
        lambda_fair     = 0.05,
        sbrs_quantile   = 0.7,
        fips_lam        = 1.0,
        mmd_alpha       = 0.3,
        struct_drop     = 0.5,
        warm_up         = 200,
        dp_eo_ratio     = 0.3,
        ramp_epochs     = 0,
        uncertainty_type= "entropy",
    ):
        assert backbone in ("GCN", "GraphSAGE", "SGC"),             f"Unsupported backbone: {backbone}"

        self.device           = device
        self.backbone_name    = backbone
        self.lambda_fair      = lambda_fair
        self.sbrs_quantile    = sbrs_quantile
        self.fips_lam         = fips_lam
        self.mmd_alpha        = mmd_alpha
        self.struct_drop      = struct_drop
        self.warm_up          = warm_up
        self.dp_eo_ratio      = dp_eo_ratio
        self.ramp_epochs      = ramp_epochs
        self.uncertainty_type = uncertainty_type

        self.name = f"{backbone}/FairGate"

        self.model   = _build_backbone(backbone, in_feats, h_feats,
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
        self._node_w, meta = compute_fiw_weights(
            data,
            model=self.model,
            sbrs_quantile=self.sbrs_quantile,
            fips_lam=self.fips_lam,
            uncertainty_type=self.uncertainty_type,
        )
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
            return task_val / (v + 1e-8)

        self._scales["struct"] = _s(lambda: compute_structural_loss(
            self.model, data, self._node_w, self.struct_drop))
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
                self.model, data, self._node_w, self.struct_drop)
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
        f"alpha(boundary)={meta['alpha']:.3f} "
        f"beta(degree)={meta['beta']:.3f} | "
        f"thr_boundary={meta['boundary_threshold']:.3f} | "
        f"gated={meta['gated']}/{meta['n_total']} ({gate_pct:.1f}%) | "
        f"unc={meta['uncertainty_type']} | "
        f"w_mean={weight.mean():.3f} w_std={weight.std():.3f}"
    )