"""SigLIP and SigLIP2 as targets: an update of the pooling head's last affine.

Their image embedding comes out of a residual attention-pooling head,

    a(x) = AttentionPool(x),   u(x) = act(W_1 LN(a(x)) + b_1),   z(x) = a(x) + W_2 u(x) + b_2,

so an operator on z cannot be folded into any one weight. The last affine still
carries an unconstrained additive channel, z_new(x) - z(x) = dW_2 u(x) + db_2 exactly,
with everything upstream frozen. Writing f = z / ||z|| and stacking the rows
[u(x); 1] / ||z(x)|| into H, the first-order change of the affinities is H B G^T with
B = [dW_2^T; db_2^T], so the descriptor is matched by the same two-sided solve as the
operator with F replaced by H. The map U -> B is linear on a grid, so the head updates of
several terms compose by addition exactly as operators do.
"""
import os

import torch
import torch.nn.functional as F
from . import config as C
from . import models as M


def pooling_head(model):
    head = getattr(M._unwrap(model.vision_model), "head", None)
    if head is None or not hasattr(getattr(head, "mlp", None), "fc2"):
        raise TypeError(f"{type(model).__name__} has no residual pooling head")
    return head


class _Fc2Input:
    """Captures u(x), the input of the head's second affine."""

    def __init__(self, model):
        self.u = None
        self._h = pooling_head(model).mlp.fc2.register_forward_pre_hook(self._grab)

    def _grab(self, module, inputs):
        u = inputs[0].detach()
        self.u = u[:, 0] if u.dim() == 3 else u

    def close(self):
        self._h.remove()


@torch.no_grad()
def head_pool_features(model, processor, pool, device, batch=64):
    """(F, H, r) of a uint8 pool: unit embeddings, rows [u; 1] / ||z||, and ||z||."""
    tap = _Fc2Input(model)
    Fs, Hs, rs = [], [], []
    try:
        for i in range(0, len(pool), batch):
            x = M.prepare(pool[i:i + batch].float() / 255.0, processor, device)
            z = model.get_image_features(pixel_values=x)
            r = z.norm(dim=-1, keepdim=True)
            Fs.append((z / r).cpu())
            Hs.append((torch.cat([tap.u, torch.ones_like(r)], -1) / r).cpu())
            rs.append(r.squeeze(-1).cpu())
    finally:
        tap.close()
    return torch.cat(Fs), torch.cat(Hs), torch.cat(rs)


def head_features(name, bundle, device="cuda", path=None):
    """{"F", "H", "r"} of a head target on the probe bundle, cached on disk."""
    path = path or C.cache_path("features", f"{name}_head_n{bundle['pool'].shape[0]}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    model, proc = M.load(name, device)
    Fb, H, r = head_pool_features(model, proc, bundle["pool"], device)
    out = {"F": Fb, "H": H, "r": r}
    del model
    torch.cuda.empty_cache()
    torch.save(out, path)
    return out


@torch.no_grad()
def fold_head_update(model, B, alpha):
    """W_2 <- W_2 + alpha dW_2 and b_2 <- b_2 + alpha db_2, in place."""
    fc2 = pooling_head(model).mlp.fc2
    d_u = fc2.in_features
    Bt = B.t().to(device=fc2.weight.device, dtype=fc2.weight.dtype)
    fc2.weight.add_(alpha * Bt[:, :d_u])
    fc2.bias.add_(alpha * Bt[:, d_u])
    return model
