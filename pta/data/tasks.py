"""Task registry: class names, prompt templates and the train / test splits.

Every split is the one the paper uses. EuroSAT has no official split and is cut 80/20 by
a permutation seeded at 42; the other datasets use their own train and test splits. Probes
are always drawn from a training split and every accuracy is measured on the test split.
"""
import os
from dataclasses import dataclass
from typing import Callable, List

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR100, DTD, GTSRB, MNIST, SVHN, EuroSAT

from .. import config as C
from .transforms import with_rgb


def clean_class_name(name):
    return name.replace("_", " ").replace("-", " ").lower().strip()


EUROSAT_CLASSES = [
    "annual crop land", "forest", "herbaceous vegetation", "highway or road",
    "industrial building", "pasture land", "permanent crop land",
    "residential building", "river", "sea or lake",
]
DTD_CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed",
    "cracked", "crosshatched", "crystalline", "dotted", "fibrous", "flecked",
    "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed",
    "interlaced", "knitted", "lacelike", "lined", "marbled", "matted",
    "meshed", "paisley", "perforated", "pitted", "pleated", "polka-dotted",
    "porous", "potholed", "scaly", "smeared", "spiralled", "sprinkled",
    "stained", "stratified", "striped", "studded", "swirly", "veined",
    "waffled", "woven", "wrinkled", "zigzagged",
]
GTSRB_CLASSES = [
    "speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60",
    "speed limit 70", "speed limit 80", "end of speed limit 80",
    "speed limit 100", "speed limit 120", "no passing",
    "no passing for vehicles over 3.5 metric tons", "right-of-way at the next intersection",
    "priority road", "yield", "stop", "no vehicles", "vehicles over 3.5 metric tons prohibited",
    "no entry", "general caution", "dangerous curve to the left", "dangerous curve to the right",
    "double curve", "bumpy road", "slippery road", "road narrows on the right",
    "road work", "traffic signals", "pedestrians", "children crossing", "bicycles crossing",
    "beware of ice or snow", "wild animals crossing", "end of all speed and passing limits",
    "turn right ahead", "turn left ahead", "ahead only", "go straight or right",
    "go straight or left", "keep right", "keep left", "roundabout mandatory",
    "end of no passing", "end of no passing by vehicles over 3.5 metric tons",
]
RESISC45_CLASSES = [
    "airplane", "airport", "baseball diamond", "basketball court", "beach",
    "bridge", "chaparral", "church", "circular farmland", "cloud",
    "commercial area", "dense residential", "desert", "forest", "freeway",
    "golf course", "ground track field", "harbor", "industrial area", "intersection",
    "island", "lake", "meadow", "medium residential", "mobile home park",
    "mountain", "overpass", "palace", "parking lot", "railway",
    "railway station", "rectangular farmland", "river", "roundabout", "runway",
    "sea ice", "ship", "snowberg", "sparse residential", "stadium",
    "storage tank", "tennis court", "terrace", "thermal power station", "wetland",
]
DIGITS = [str(i) for i in range(10)]


class HFImages(Dataset):
    """A Hugging Face split with `image` and `label` columns, as (tensor, int)."""

    def __init__(self, split, transform):
        self.ds, self.transform = split, transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"]
        if not hasattr(img, "convert"):
            from PIL import Image
            img = Image.fromarray(img)
        return self.transform(img.convert("RGB")), int(item["label"])


def _hf(name):
    from datasets import load_dataset
    return load_dataset(name)


def _hf_classes(name):
    from datasets import load_dataset
    feat = load_dataset(name, split="train[:1]").features["label"]
    return [clean_class_name(n) for n in feat.names]


def _seeded_cut(n, frac=0.8, seed=C.SEED):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    cut = int(frac * n)
    return perm[:cut], perm[cut:]


def _root(name):
    p = os.path.join(C.DATA_DIR, name)
    os.makedirs(p, exist_ok=True)
    return p


# --------------------------------------------------------------- splits ----
def _eurosat(train_tf, test_tf):
    root = _root("eurosat")
    full = EuroSAT(root=root, download=True, transform=train_tf)
    tr, te = _seeded_cut(len(full))
    return Subset(full, tr), Subset(EuroSAT(root=root, download=False, transform=test_tf), te)


def _dtd(train_tf, test_tf):
    root = _root("dtd")
    return (DTD(root=root, split="train", download=True, transform=train_tf),
            DTD(root=root, split="test", download=True, transform=test_tf))


def _gtsrb(train_tf, test_tf):
    root = _root("gtsrb")
    return (GTSRB(root=root, split="train", download=True, transform=train_tf),
            GTSRB(root=root, split="test", download=True, transform=test_tf))


def _mnist(train_tf, test_tf):
    root = _root("mnist")
    return (MNIST(root=root, train=True, download=True, transform=with_rgb(train_tf)),
            MNIST(root=root, train=False, download=True, transform=with_rgb(test_tf)))


def _svhn(train_tf, test_tf):
    root = _root("svhn")
    return (SVHN(root=root, split="train", download=True, transform=train_tf),
            SVHN(root=root, split="test", download=True, transform=test_tf))


def _cifar100(train_tf, test_tf):
    root = _root("cifar100")
    return (CIFAR100(root=root, train=True, download=True, transform=train_tf),
            CIFAR100(root=root, train=False, download=True, transform=test_tf))


def _hf_task(name):
    def build(train_tf, test_tf):
        d = _hf(name)
        return HFImages(d["train"], train_tf), HFImages(d["test"], test_tf)
    return build


def _cifar100_classes():
    return [clean_class_name(c) for c in CIFAR100(root=_root("cifar100"), train=False,
                                                  download=True).classes]


@dataclass
class Task:
    name: str
    classes: Callable[[], List[str]]
    template: str
    build: Callable

    def prompts(self):
        return [self.template.format(c) for c in self.classes()]

    def num_classes(self):
        return len(self.classes())

    def splits(self, train_tf, test_tf):
        """(train dataset, test dataset) under the given transforms."""
        return self.build(train_tf, test_tf)


REGISTRY = {
    "eurosat": Task("eurosat", lambda: EUROSAT_CLASSES, "a centered satellite photo of {}.", _eurosat),
    "dtd": Task("dtd", lambda: DTD_CLASSES, "a photo of a {} texture.", _dtd),
    "cars": Task("cars", lambda: _hf_classes("tanganke/stanford_cars"), "a photo of a {}.",
                 _hf_task("tanganke/stanford_cars")),
    "gtsrb": Task("gtsrb", lambda: GTSRB_CLASSES, "a photo of a {} traffic sign.", _gtsrb),
    "mnist": Task("mnist", lambda: DIGITS, "a photo of the number {}.", _mnist),
    "resisc45": Task("resisc45", lambda: RESISC45_CLASSES, "a satellite photo of {}.",
                     _hf_task("timm/resisc45")),
    "svhn": Task("svhn", lambda: DIGITS, "a photo of the number {}.", _svhn),
    "sun397": Task("sun397", lambda: _hf_classes("tanganke/sun397"), "a photo of a {}.",
                   _hf_task("tanganke/sun397")),
    "cifar100": Task("cifar100", _cifar100_classes, "a photo of a {}.", _cifar100),
}


def get_task(name):
    return REGISTRY[name]


def prompts(name):
    return get_task(name).prompts()
