"""Addition: compose tasks from other models into a target.

  python addition.py --target clip --terms openclip:eurosat,metaclip:dtd
  python addition.py --target clip --terms openclip:eurosat,metaclip:dtd --method adapter

Each term <source>:<task> reads the source's descriptor of the task from its fine-tune in
checkpoints/sources. The closed form adds the terms' operators and applies the sum at --alpha
(0.5 in the paper); the adapter fits one LoRA to all terms jointly at full strength (--alpha 1).
The target is scored on each task's test split and on the controls before and after the edit.
"""
import argparse

import torch

from pta import models as M
from pta import transfer as TR
from pta.evaluate import accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=list(M.SPECS))
    ap.add_argument("--terms", required=True, help="comma-separated <source>:<task>")
    ap.add_argument("--method", choices=["closed_form", "adapter"], default="closed_form")
    ap.add_argument("--alpha", type=float, default=None,
                    help="strength (default 0.5 for the closed form, 1.0 for the adapter)")
    ap.add_argument("--controls", default="cifar100", help="comma-separated, from cifar100,imagenet")
    ap.add_argument("--save", default=None, help="write the edit: the operator, or the adapter's LoRA")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    alpha = a.alpha if a.alpha is not None else (0.5 if a.method == "closed_form" else 1.0)
    pairs = TR.parse_terms(a.terms)
    names = [t for _, t in pairs] + [c for c in a.controls.split(",") if c]
    bf16 = a.method == "adapter"

    bundle, feats = TR.prepare(a.target, pairs, a.device)
    model, proc = M.load(a.target, a.device)
    before = accuracy(model, proc, names, a.device, bf16=bf16)
    if a.method == "closed_form":
        update = TR.closed_form_update(a.target, [(m, t, 1.0) for m, t in pairs], bundle, feats, a.device)
        TR.apply_closed_form(model, a.target, update, alpha)
        if a.save:
            torch.save({"target": a.target, "update": update.cpu(), "alpha": alpha, "terms": pairs}, a.save)
    else:
        TR.fit_adapter(model, proc, a.target, [(m, t, alpha) for m, t in pairs], bundle, feats, a.device)
        if a.save:
            M.save_lora(model, a.save)
    after = accuracy(model, proc, names, a.device, bf16=bf16)

    print(f"\n{a.target} + {a.terms} ({a.method}, alpha {alpha})")
    for n in names:
        print(f"  {n:10s} {100 * before[n]:6.2f} -> {100 * after[n]:6.2f}")


if __name__ == "__main__":
    main()
