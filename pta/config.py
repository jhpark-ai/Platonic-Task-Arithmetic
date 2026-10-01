"""Paths and the fixed constants of the paper.

    PTA_DATA     datasets (default ./workspace/data)
    PTA_CACHE    probe bundles and cached embeddings (default ./workspace/cache)
    PTA_SOURCES  source fine-tunes (default: the 48 checkpoints shipped in checkpoints/sources)

Hugging Face models and datasets use the standard HF cache (set `HF_HOME` to move it).
"""
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.environ.get("PTA_ROOT", "workspace"))
DATA_DIR = os.path.abspath(os.environ.get("PTA_DATA", os.path.join(ROOT, "data")))
CACHE_DIR = os.path.abspath(os.environ.get("PTA_CACHE", os.path.join(ROOT, "cache")))
SOURCES_DIR = os.path.abspath(os.environ.get("PTA_SOURCES", os.path.join(REPO_DIR, "checkpoints", "sources")))

SEED = 42

# The eight tasks, in the order every probe bundle is laid out.
TASKS = ["eurosat", "dtd", "cars", "gtsrb", "mnist", "resisc45", "svhn", "sun397"]
# Targets whose embedding comes out of a residual pooling head: they take the head update.
HEAD_TARGETS = ["siglip", "siglip2"]
CONTROLS = ["cifar100", "imagenet"]

PROBES_PER_CLASS = 30        # unlabeled probes per class, drawn from the training split
CONTROL_PER_CLASS = 10       # control images per class, from the test split

# Closed form: two-sided ridge, each penalty relative to the mean squared singular value.
RIDGE_F = 1e-2
RIDGE_G = 1e-6

SOURCE_RANK, SOURCE_ALPHA = 4, 8        # the source fine-tunes (LoRA on the vision q, v projections)
ADAPTER_RANK, ADAPTER_ALPHA = 16, 32    # the adapter realization


def source_path(model, task, root=None):
    """Checkpoint of the source fine-tune of `model` on `task` (a LoRA state dict)."""
    return os.path.join(root or SOURCES_DIR, f"{model}_{task}.pt")


def cache_path(*parts):
    p = os.path.join(CACHE_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
