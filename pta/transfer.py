"""Addition and negation of tasks on a target model, in either realization.

A term is (source model, task, coefficient c). Its descriptor is read from the source's
fine-tune on the task's own probes and prompts. Addition gives every term c = 1; negation gives
its one term c = -lambda.

  closed form  each term is solved on the task's own grid of the target into an operator
               (for SigLIP and SigLIP2, an update of the pooling head's last affine), and the
               target receives sum_l c_l X_l at strength alpha, folded into its last layer
  adapter      one LoRA in the target's vision encoder is fitted to all terms jointly
"""
from . import adapter as AD
from . import config as C
from . import features as FT
from . import head as HD
from . import models as M
from .solver import ridge_solver


def parse_terms(spec):
    """"openclip:eurosat,metaclip:dtd" -> [("openclip", "eurosat"), ("metaclip", "dtd")]."""
    return [tuple(x.split(":")) for x in spec.split(",")]


def prepare(target, pairs, device="cuda"):
    """Probe bundle of the terms' tasks and the cached features of every model involved.

    pairs: [(source, task), ...]. Each source is encoded with its fine-tunes on the tasks it
    supplies; the target is encoded as pre-trained (and with its own fine-tunes where it is
    also a source)."""
    tasks = [t for t in C.TASKS if t in {t for _, t in pairs}]
    tag = "_".join(tasks)
    bundle = FT.build_probe_bundle(tasks, path=C.cache_path("probes", f"bundle_{tag}.pt"))
    need = {target: set()}
    for m, t in pairs:
        need.setdefault(m, set()).add(t)
    feats = {}
    for m, ts in need.items():
        own = [t for t in tasks if t in ts]
        feats[m] = FT.model_features(
            m, bundle, device, tasks=own,
            path=C.cache_path("features", f"{m}__{tag}__{'-'.join(own) or 'pre'}.pt"))
    return bundle, feats


def closed_form_update(target, terms, bundle, feats, device="cuda"):
    """sum_l c_l X_l over terms [(source, task, c)], each X_l solved on the target's grid of task l.

    For a target with an affine last layer X_l is the operator's transpose (A = X^T); for SigLIP
    and SigLIP2 it is the head update [dW_2^T; db_2^T]."""
    hf = None
    if target in C.HEAD_TARGETS:
        tag = "_".join(m["task"] for m in bundle["meta"])
        hf = HD.head_features(target, bundle, device, path=C.cache_path("features", f"{target}__{tag}__head.pt"))
    total = None
    for m, t, c in terms:
        F, G = FT.task_grid(feats[target], bundle, t, device)
        if hf is not None:
            i0, i1 = FT.image_slice(bundle, t)
            F = hf["H"][i0:i1].to(device)
        X = c * ridge_solver(F, G, C.RIDGE_F, C.RIDGE_G)(FT.descriptor(feats[m], bundle, t, device))
        total = X if total is None else total + X
    return total


def apply_closed_form(model, target, update, alpha):
    """Write the update into the target's weights: W <- (I + alpha A) W, or the head update."""
    if target in C.HEAD_TARGETS:
        return HD.fold_head_update(model, update, alpha)
    return M.fold_operator(model, update, alpha)


def fit_adapter(model, processor, target, terms, bundle, feats, device="cuda"):
    """Attach the adapter to `model` and fit it to sum_l c_l U_l, in place; returns diagnostics."""
    AD.attach(model, C.SEED)
    base = AD.base_features(model, processor, bundle["pool"], device)
    fit = []
    for m, t, c in terms:
        i0, i1 = FT.image_slice(bundle, t)
        p0, p1 = FT.prompt_slice(bundle, t)
        fit.append({"isl": (i0, i1), "G": feats[target]["G"][p0:p1],
                    "U": c * FT.descriptor(feats[m], bundle, t, "cpu")})
    return AD.realize(model, processor, bundle["pool"], fit, base, device)
