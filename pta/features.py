"""The probe bundle, the cached probe embeddings and the descriptors read from them.

Probe bundle. Each task contributes n_cls x 30 unlabeled images drawn uniformly (no
per-class quota) from its TRAINING split with a generator seeded at 42, so no probe
is ever an evaluation image. Images are stored once as 224 x 224 uint8 after CLIP-L's
resize and center crop, and each model reads them resized to its own resolution.

Feature cache of a model. On the shared bundle: the unit image embeddings of the
pre-trained model (`F_pre`), of each of its source fine-tunes (`F_ft[task]`), and the
unit text embeddings of every task's prompts (`G`). A descriptor is then a slice and a
matrix product, and no image is encoded twice.
"""
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from . import config as C
from . import models as M
from .data.tasks import get_task
from .data.transforms import eval_transform

BUNDLE_MODEL = "openai/clip-vit-large-patch14"   # resize / crop statistics of the stored bundle


# --------------------------------------------------------------- bundle -----
def build_probe_bundle(tasks=C.TASKS, per_class=C.PROBES_PER_CLASS, seed=C.SEED, path=None,
                       num_workers=8):
    """The shared unlabeled probe bundle (see the module docstring).

    Returns {"pool": uint8 (N, 3, 224, 224), "meta": [per task: offset, n, n_cls, labels],
    "prompts": [...], "prompt_meta": [per task: n], "split": "train"}. Labels are kept for
    diagnostics only; no descriptor or solve reads them."""
    path = path or C.cache_path("probes", f"bundle_pc{per_class}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    from transformers import CLIPProcessor
    ip = CLIPProcessor.from_pretrained(BUNDLE_MODEL).image_processor
    mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std).view(1, 3, 1, 1)
    size = ip.size.get("height", ip.size.get("shortest_edge", 224))
    tf = eval_transform(ip.image_mean, ip.image_std, size)

    imgs, meta, prompts, off = [], [], [], 0
    for t in tasks:
        task = get_task(t)
        n_cls = task.num_classes()
        ds, _ = task.splits(tf, tf)
        n = min(n_cls * per_class, len(ds))
        g = torch.Generator().manual_seed(seed)
        pick = sorted(torch.randperm(len(ds), generator=g)[:n].tolist())
        buf, lbl = [], []
        for x, y in DataLoader(Subset(ds, pick), batch_size=64, shuffle=False,
                               num_workers=num_workers):
            # stored exactly as the paper's bundle: de-normalized, clamped, truncated
            buf.extend(((x * std + mean).clamp(0, 1) * 255).to(torch.uint8))
            lbl.extend(int(v) for v in y)
        imgs.extend(buf)
        meta.append({"task": t, "offset": off, "n": len(buf), "n_cls": n_cls,
                     "n_per_class": per_class, "labels": torch.tensor(lbl),
                     "split": "train", "n_split": len(ds)})
        prompts.extend(task.prompts())
        off += len(buf)
        print(f"  probes + {t}: {len(buf)} of {len(ds)} training images -> {off}", flush=True)
    bundle = {"pool": torch.stack(imgs), "meta": meta, "prompts": prompts,
              "prompt_meta": [{"task": t, "n": len(get_task(t).prompts())} for t in tasks],
              "split": "train"}
    torch.save(bundle, path)
    return bundle


def image_slice(bundle, task):
    """[start, end) of one task's probes in the pool."""
    for m in bundle["meta"]:
        if m["task"] == task:
            return m["offset"], m["offset"] + m["n"]
    raise KeyError(task)


def prompt_slice(bundle, task):
    """[start, end) of one task's prompt rows in the landmark matrix."""
    o = 0
    for pm in bundle["prompt_meta"]:
        if pm["task"] == task:
            return o, o + pm["n"]
        o += pm["n"]
    raise KeyError(task)


# ------------------------------------------------------ probe embeddings ----
@torch.no_grad()
def embed_pool(model, processor, pool, device, batch=64):
    """Unit image embeddings of a uint8 pool, on CPU."""
    out = []
    for i in range(0, len(pool), batch):
        x = M.prepare(pool[i:i + batch].float() / 255.0, processor, device)
        out.append(F.normalize(model.get_image_features(pixel_values=x), dim=-1).cpu())
    return torch.cat(out)


def model_features(name, bundle, device="cuda", tasks=C.TASKS, sources=None, path=None):
    """{"G", "F_pre", "F_ft": {task: ...}} of one model on the bundle, cached on disk.

    `sources` maps a task to its source checkpoint (default `config.source_path`); a model
    used only as a target passes `tasks=()`."""
    path = path or C.cache_path("features", f"{name}_n{bundle['pool'].shape[0]}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    model, proc = M.load(name, device)
    out = {"G": M.text_landmarks(model, proc, bundle["prompts"], device).cpu(),
           "F_pre": embed_pool(model, proc, bundle["pool"], device), "F_ft": {},
           "split": bundle.get("split")}
    if tasks:
        M.attach_lora(model, C.SOURCE_RANK, C.SOURCE_ALPHA)
        for t in tasks:
            M.load_lora(model, (sources or {}).get(t) or C.source_path(name, t))
            model.eval()
            out["F_ft"][t] = embed_pool(model, proc, bundle["pool"], device)
            print(f"  features {name}/{t}", flush=True)
    del model
    torch.cuda.empty_cache()
    torch.save(out, path)
    return out


def shift_field(feats, task):
    """f_ft(x) - f_pre(x) of the source fine-tuned on `task`, over the whole pool."""
    return feats["F_ft"][task] - feats["F_pre"]


def task_grid(feats, bundle, task, device):
    """(F, G) of one model restricted to one task's probes and prompts."""
    i0, i1 = image_slice(bundle, task)
    p0, p1 = prompt_slice(bundle, task)
    return feats["F_pre"][i0:i1].to(device), feats["G"][p0:p1].to(device)


def descriptor(feats, bundle, task, device):
    """UTD[i, j] = <f_ft(x_i), g(p_j)> - <f_pre(x_i), g(p_j)> of one source on the task's grid.

    Both embeddings and landmarks are the source's own; N_task x M_task."""
    i0, i1 = image_slice(bundle, task)
    p0, p1 = prompt_slice(bundle, task)
    return shift_field(feats, task)[i0:i1].to(device) @ feats["G"][p0:p1].to(device).t()
