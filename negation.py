"""Negation: remove a task from a target with another model's descriptor.

  python negation.py --target clip --source openclip --task eurosat
  python negation.py --target clip --source openclip --task eurosat --method adapter

The source's descriptor of the task is read from its fine-tune in checkpoints/sources and applied
at -lambda (--lam, 1 in the paper): the closed form folds -lambda times the task's operator into
the target's last layer, the adapter fits one LoRA to -lambda times the descriptor. The target is
scored on the task's test split and on the controls before and after the edit.
"""
import argparse

import torch

from pta import models as M
from pta import transfer as TR
from pta.evaluate import accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(M.SPECS))
    ap.add_argument("--source", required=True, choices=list(M.SPECS))
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", choices=["closed_form", "adapter"], default="closed_form")
    ap.add_argument("--lam", type=float, default=1.0, help="negation strength lambda")
    ap.add_argument("--controls", default="cifar100", help="comma-separated, from cifar100,imagenet")
    ap.add_argument("--save", default=None, help="write the edit: the operator, or the adapter's LoRA")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    terms = [(a.source, a.task, -a.lam)]
    names = [a.task] + [c for c in a.controls.split(",") if c]
    bf16 = a.method == "adapter"

    bundle, feats = TR.prepare(a.target, [(a.source, a.task)], a.device)
    model, proc = M.load(a.target, a.device)
    before = accuracy(model, proc, names, a.device, bf16=bf16)
    if a.method == "closed_form":
        update = TR.closed_form_update(a.target, terms, bundle, feats, a.device)
        TR.apply_closed_form(model, a.target, update, 1.0)
        if a.save:
            torch.save({"target": a.target, "update": update.cpu(), "alpha": 1.0, "terms": terms}, a.save)
    else:
        TR.fit_adapter(model, proc, a.target, terms, bundle, feats, a.device)
        if a.save:
            M.save_lora(model, a.save)
    after = accuracy(model, proc, names, a.device, bf16=bf16)

    print(f"\n{a.target} - {a.source}:{a.task} ({a.method}, lambda {a.lam})")
    for n in names:
        print(f"  {n:10s} {100 * before[n]:6.2f} -> {100 * after[n]:6.2f}")


if __name__ == "__main__":
    main()
