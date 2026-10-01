"""Source fine-tunes: a LoRA of rank 4 / alpha 8 on the vision q and v projections,
trained with cross-entropy over the task's class-name prompts.

Two recipes appear among the paper's 48 sources (Appendix, datasets and sources):
  recipe A  learning rate 1e-4, batch 32, weight decay 0.01 (40 sources)
  recipe B  learning rate 3e-4, batch 64, weight decay 1e-4 (the EuroSAT and DTD
            sources of CLIP, OpenCLIP, MetaCLIP and SigLIP)
Three epochs, one for SUN397. The paper's sources were trained without a fixed seed;
pass `seed` to make a run repeatable.
"""
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from . import models as M
from .data.tasks import get_task
from .data.transforms import model_transforms

RECIPES = {"A": dict(lr=1e-4, batch=32, weight_decay=0.01),
           "B": dict(lr=3e-4, batch=64, weight_decay=1e-4)}
RECIPE_B_SOURCES = {(m, t) for m in ("clip", "openclip", "metaclip", "siglip")
                    for t in ("eurosat", "dtd")}


def paper_recipe(model_name, task):
    return "B" if (model_name, task) in RECIPE_B_SOURCES else "A"


def paper_epochs(task):
    return 1 if task == "sun397" else 3


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def train_source(model_name, task, out_path, recipe="A", epochs=None, device="cuda", seed=None,
                 num_workers=4, eval_each_epoch=True):
    """Fine-tune one source and save its LoRA state dict to `out_path`."""
    if seed is not None:
        seed_everything(seed)
    hp = RECIPES[recipe]
    epochs = epochs or paper_epochs(task)
    model, proc = M.load(model_name, device)
    M.attach_lora(model, C.SOURCE_RANK, C.SOURCE_ALPHA)
    M.freeze_all_but_lora(model)
    params = [p for p in model.parameters() if p.requires_grad]

    t = get_task(task)
    text = M.text_landmarks(model, proc, t.prompts(), device)
    train_tf, eval_tf = model_transforms(proc)
    train_ds, test_ds = t.splits(train_tf, eval_tf)
    train = DataLoader(train_ds, hp["batch"], shuffle=True, num_workers=num_workers,
                       pin_memory=True, drop_last=True)
    test = DataLoader(test_ds, hp["batch"], shuffle=False, num_workers=num_workers, pin_memory=True)
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    scale = model.logit_scale.exp().detach()

    for ep in range(epochs):
        model.train()
        t0, seen, total = time.time(), 0, 0.0
        for x, y in train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            img = F.normalize(model.get_image_features(pixel_values=x), dim=-1)
            loss = F.cross_entropy(scale * img @ text.t(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            seen += x.size(0)
        msg = f"  epoch {ep}: loss {total / seen:.4f}"
        if eval_each_epoch or ep == epochs - 1:
            msg += f"  test {100 * accuracy(model, test, text, device):.2f}%"
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
        M.save_lora(model, out_path)
    print(f"saved -> {out_path}", flush=True)
    return out_path


@torch.no_grad()
def accuracy(model, loader, text, device):
    model.eval()
    c = n = 0
    for x, y in loader:
        img = F.normalize(model.get_image_features(pixel_values=x.to(device)), dim=-1)
        c += int(((img @ text.t()).argmax(-1) == y.to(device)).sum())
        n += y.numel()
    return c / n
