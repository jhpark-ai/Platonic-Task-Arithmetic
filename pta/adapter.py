"""The adapter realization: a LoRA inside the target's vision encoder fitted to the shift
the descriptors prescribe.

The objective is the closed form's, on the same per-task grids, with the edit no longer
confined to the last layer and the terms of a composition fitted jointly:

    min_theta  mean_i  mean_{j in prompts of task(i)}
        ( <f_theta(x_i), g(p_j)> - <f_pre(x_i), g(p_j)> - c_l U_l[i, j] )^2,

where U_l is the source's descriptor of task l on that task's probes and prompts and c_l
the term's signed strength (1 for an addition term at full strength, -lambda for
negation). Each probe contributes the mean over its own task's prompts.

Recipe: LoRA rank 16 / alpha 32 on the vision q and v projections, AdamW (lr 3e-4,
weight decay 1e-4), batch 64, cosine schedule, 20 epochs over the edit's probes capped
at 540 optimizer steps, bf16 autocast. The base affinities are read from the same bf16
forward with the adapter disabled, so the untrained adapter induces exactly zero change
and its loss is then the mean squared prescribed shift. Every edit starts from one fixed
adapter initialization and one seeded batch order.
"""
import math
import time

import torch
import torch.nn.functional as F

from . import config as C
from . import models as M

LR, WEIGHT_DECAY, BATCH = 3e-4, 1e-4, 64
EPOCHS, MAX_STEPS = 20, 540
AMP = torch.bfloat16


def attach(model, seed=C.SEED, rank=C.ADAPTER_RANK, alpha=C.ADAPTER_ALPHA):
    """Attach the adapter once and return its initial state, reused by every edit."""
    from peft import get_peft_model_state_dict
    torch.manual_seed(seed)
    M.attach_lora(model, rank, alpha)
    M.freeze_all_but_lora(model)
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model.vision_model).items()}


def reset(model, init_state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model.vision_model, init_state)


def _embed(model, processor, raw_u8, device):
    x = M.prepare(raw_u8.float().div_(255.0), processor, device)
    with torch.autocast("cuda", dtype=AMP):
        f = model.get_image_features(pixel_values=x)
    return F.normalize(f.float(), dim=-1)


@torch.no_grad()
def base_features(model, processor, pool, device, batch=128):
    """bf16 embeddings of every probe with the adapter disabled."""
    model.eval()
    out = []
    with M.lora_disabled(model):
        for i in range(0, pool.shape[0], batch):
            out.append(_embed(model, processor, pool[i:i + batch], device).cpu())
    return torch.cat(out)


def num_steps(n_probes):
    return min(EPOCHS * math.ceil(n_probes / BATCH), MAX_STEPS)


def realize(model, processor, pool, terms, base, device, seed=C.SEED):
    """Fit the attached adapter to a signed sum of per-task terms.

    terms: [{"isl": (start, end) probe slice, "G": target landmarks of the task (M_l, d),
             "U": c_l * U_l (N_l, M_l)}, ...]
    base: bf16 embeddings of the whole pool with the adapter disabled.
    Returns training diagnostics."""
    idx = torch.cat([torch.arange(*t["isl"]) for t in terms])
    owner = torch.cat([torch.full((t["isl"][1] - t["isl"][0],), k) for k, t in enumerate(terms)])
    local = torch.cat([torch.arange(t["isl"][1] - t["isl"][0]) for t in terms])
    G = [t["G"].to(device).float() for t in terms]
    base_aff = [(base[slice(*t["isl"])].to(device) @ g.t()) for t, g in zip(terms, G)]
    U = [t["U"].to(device).float() for t in terms]
    zero_loss = float(torch.cat([(u ** 2).mean(1) for u in U]).mean())

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    N = idx.numel()
    steps = num_steps(N)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    t0, step, hist = time.time(), 0, []
    while step < steps:
        perm = torch.randperm(N, generator=gen)
        for s in range(0, N, BATCH):
            if step >= steps:
                break
            b = perm[s:s + BATCH]
            f = _embed(model, processor, pool[idx[b]], device)
            own, loc = owner[b].to(device), local[b]
            per_img = []
            for k in range(len(terms)):
                sel = (own == k).nonzero().flatten()
                if sel.numel() == 0:
                    continue
                rows = loc[sel.cpu()].to(device)
                per_img.append(((f[sel] @ G[k].t() - base_aff[k][rows] - U[k][rows]) ** 2).mean(1))
            loss = torch.cat(per_img).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            hist.append(float(loss))
            step += 1
    tail = hist[-max(1, steps // 10):]
    model.eval()
    final = sum(tail) / len(tail)
    return {"steps": steps, "n_probes": N, "zero_loss": zero_loss, "final_loss": final,
            "first_loss": hist[0], "rel_residual": final / zero_loss if zero_loss else None,
            "train_s": round(time.time() - t0, 1)}
