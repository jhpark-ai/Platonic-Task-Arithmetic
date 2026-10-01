"""Top-1 accuracy of a model, edited or not, on task test splits and on the controls."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config as C
from . import models as M
from .data.controls import control_sets
from .data.tasks import get_task
from .data.transforms import model_transforms


@torch.no_grad()
def accuracy(model, processor, names, device="cuda", bf16=False, batch=64, num_workers=4):
    """{name: accuracy} over the test splits of the named tasks and the named controls
    ("cifar100", "imagenet"), each image scored against the class-name prompts of its set."""
    _, tf = model_transforms(processor)
    model.eval()
    out = {}
    for name in names:
        if name in C.CONTROLS:
            ds = control_sets(tf, [name])[name]
            prompts = ds.prompts
        else:
            ds = get_task(name).splits(tf, tf)[1]
            prompts = get_task(name).prompts()
        G = M.text_landmarks(model, processor, prompts, device).float()
        hit = n = 0
        for x, y in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers,
                               pin_memory=True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                f = model.get_image_features(pixel_values=x.to(device))
            f = F.normalize(f.float(), dim=-1)
            hit += int(((f @ G.t()).argmax(1).cpu() == y).sum())
            n += len(y)
        out[name] = hit / n
    return out
