"""Vision-language models, their LoRA adapters and the layer an edit is written into.

Every model is exposed through the Hugging Face CLIP interface
(`get_image_features`, `get_text_features`, `vision_model`, `logit_scale`) together
with a processor whose `image_processor` carries `image_mean`, `image_std` and `size`,
so the rest of the code never branches on the family.

Which layer receives the closed-form edit:
  clip, openclip, metaclip   the bias-free `visual_projection`, W <- (I + alpha A) W
  evaclip                    the affine `visual.trunk.head`, W <- (I + alpha A) W, b <- (I + alpha A) b
  siglip, siglip2            the residual pooling head, whose second affine `head.mlp.fc2`
                             receives the update of `pta.head` instead
"""
import re
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> (kind, identifier). "hf-clip" and "hf-auto" load with transformers,
# "open_clip" with open_clip (architecture, pretrained tag).
SPECS = {
    "clip": ("hf-clip", "openai/clip-vit-large-patch14"),
    "openclip": ("hf-clip", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"),
    "metaclip": ("hf-clip", "facebook/metaclip-l14-400m"),
    "siglip": ("hf-auto", "google/siglip-large-patch16-256"),
    "siglip2": ("hf-siglip2", "google/siglip2-large-patch16-256"),
    "evaclip": ("open_clip", ("EVA02-L-14", "merged2b_s4b_b131k")),
}
DISPLAY = {"clip": "CLIP-L", "openclip": "OpenCLIP-L", "metaclip": "MetaCLIP-L",
           "siglip": "SigLIP-L", "siglip2": "SigLIP2-L", "evaclip": "EVA-CLIP-L"}


# ------------------------------------------------------------ processors ----
class _SiglipLikeProcessor:
    """Image processor + tokenizer behind the HF processor call signature (SigLIP2)."""

    def __init__(self, image_processor, tokenizer):
        self.image_processor, self.tokenizer = image_processor, tokenizer

    def __call__(self, text=None, images=None, return_tensors="pt", padding="max_length",
                 truncation=True, **kw):
        from transformers.tokenization_utils_base import BatchEncoding
        out = {}
        if text is not None:
            max_len = kw.pop("max_length", 64)
            out.update(self.tokenizer(text=text, return_tensors=return_tensors, padding=padding,
                                      truncation=truncation, max_length=max_len, **kw))
        if images is not None:
            out.update(self.image_processor(images=images, return_tensors=return_tensors))
        return BatchEncoding(out)


class _OpenCLIPProcessor:
    """open_clip's preprocess and tokenizer behind the HF processor call signature."""

    class _ImageProcessor:
        def __init__(self, preprocess):
            from torchvision import transforms as T
            mean = std = (0.5, 0.5, 0.5)
            size = 224
            for t in preprocess.transforms:
                if isinstance(t, T.Normalize):
                    mean, std = list(t.mean), list(t.std)
                if isinstance(t, (T.Resize, T.CenterCrop)):
                    s = t.size
                    size = s if isinstance(s, int) else s[0]
            self.image_mean, self.image_std = list(mean), list(std)
            self.size = {"height": int(size), "width": int(size)}

    def __init__(self, preprocess, tokenizer):
        self.image_processor = self._ImageProcessor(preprocess)
        self.tokenizer, self._preprocess = tokenizer, preprocess

    def __call__(self, text=None, images=None, return_tensors="pt", padding="max_length",
                 truncation=True, **kw):
        from transformers.tokenization_utils_base import BatchEncoding
        out = {}
        if text is not None:
            ids = self.tokenizer(text if isinstance(text, list) else [text])
            out["input_ids"], out["attention_mask"] = ids, (ids != 0).long()
        if images is not None:
            images = images if isinstance(images, list) else [images]
            out["pixel_values"] = torch.stack([self._preprocess(im) for im in images])
        return BatchEncoding(out)


class OpenCLIPModel(nn.Module):
    """An open_clip model behind the HF CLIP interface."""

    def __init__(self, model):
        super().__init__()
        self._oc = model
        self.vision_model = model.visual
        self.logit_scale = model.logit_scale

    def get_image_features(self, pixel_values):
        return self._oc.encode_image(pixel_values, normalize=False)

    def get_text_features(self, input_ids=None, attention_mask=None, **kw):
        return self._oc.encode_text(input_ids, normalize=False)


# --------------------------------------------------------------- loading ----
def load(name, device="cuda"):
    """(model, processor) of a registered family, in eval mode on `device`."""
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
    kind, ident = SPECS[name]
    if kind == "hf-clip":
        model = CLIPModel.from_pretrained(ident)
        proc = CLIPProcessor.from_pretrained(ident)
    elif kind == "hf-auto":
        model, proc = AutoModel.from_pretrained(ident), AutoProcessor.from_pretrained(ident)
    elif kind == "hf-siglip2":
        from transformers import AutoImageProcessor, AutoTokenizer
        model = AutoModel.from_pretrained(ident)
        proc = _SiglipLikeProcessor(AutoImageProcessor.from_pretrained(ident),
                                    AutoTokenizer.from_pretrained(ident))
    elif kind == "open_clip":
        import open_clip
        arch, tag = ident
        oc, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
        model, proc = OpenCLIPModel(oc), _OpenCLIPProcessor(preprocess, open_clip.get_tokenizer(arch))
    else:
        raise ValueError(kind)
    return model.to(device).eval(), proc


# ----------------------------------------------------------------- inputs ----
def prepare(raw01, processor, device):
    """Resize images in [0, 1] to the model's resolution and normalize them.

    The probe bundle stores 224 x 224 images; a model at another resolution reads them
    bilinearly resized, which is how every descriptor in the paper was measured."""
    size = processor.image_processor.size
    target = size.get("height", size.get("shortest_edge", 224)) if isinstance(size, dict) else 224
    x = raw01
    if raw01.shape[-1] != target:
        x = F.interpolate(raw01, size=(target, target), mode="bilinear", align_corners=False)
    ip = processor.image_processor
    m = torch.tensor(ip.image_mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(ip.image_std, device=device).view(1, 3, 1, 1)
    return (x.to(device) - m) / s


@torch.no_grad()
def text_landmarks(model, processor, prompts, device):
    """Unit-norm text embeddings of the prompts, (M, d): the grid's landmarks."""
    tok = processor(text=prompts, return_tensors="pt", padding="max_length", truncation=True)
    tok = {k: v.to(device) for k, v in tok.items()}
    return F.normalize(model.get_text_features(**tok), dim=-1)


# ------------------------------------------------------------------- LoRA ----
_LORA_PATTERNS = [
    r"encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)$",            # HF CLIP / SigLIP
    r"(.*\.)?trunk\.blocks\.\d+\.attn\.(q_proj|v_proj)$",            # open_clip EVA (timm)
]


def lora_targets(vision_model):
    return [n for n, m in vision_model.named_modules()
            if isinstance(m, nn.Linear) and any(re.match(p, n) for p in _LORA_PATTERNS)]


def attach_lora(model, rank, alpha):
    """Wrap the vision tower with LoRA on every attention q and v projection."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=lora_targets(model.vision_model),
                     lora_dropout=0.0, bias="none")
    model.vision_model = get_peft_model(model.vision_model, cfg)
    return model


def freeze_all_but_lora(model):
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n


def save_lora(model, path):
    from peft import get_peft_model_state_dict
    torch.save(get_peft_model_state_dict(model.vision_model), path)


def load_lora(model, path):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model.vision_model, torch.load(path, map_location="cpu"))


@contextmanager
def lora_disabled(model):
    with model.vision_model.disable_adapter():
        yield


# ------------------------------------------------------------ last layer ----
def _unwrap(vision_model):
    return vision_model.get_base_model() if hasattr(vision_model, "get_base_model") else vision_model


def last_layer(model):
    """(module, kind) of the layer that writes the image embedding."""
    if hasattr(model, "visual_projection"):
        return model.visual_projection, "projection"
    vm = _unwrap(model.vision_model)
    if hasattr(vm, "trunk") and isinstance(getattr(vm.trunk, "head", None), nn.Linear):
        return vm.trunk.head, "affine"
    head = getattr(vm, "head", None)
    if head is not None and hasattr(getattr(head, "mlp", None), "fc2"):
        return head.mlp.fc2, "head"
    raise TypeError(f"no editable last layer on {type(model).__name__}")


@torch.no_grad()
def fold_operator(model, X, alpha):
    """Write f -> (I + alpha A) f, A = X^T, into the target's affine last layer, in place.

    The edited model is an ordinary checkpoint of the same architecture. For a
    residual pooling head (SigLIP, SigLIP2) use `pta.head.fold_head_update` instead."""
    layer, kind = last_layer(model)
    if kind == "head":
        raise TypeError("a residual pooling head takes the head update of pta.head")
    W = layer.weight
    M = torch.eye(W.shape[0], dtype=W.dtype, device=W.device) + alpha * X.t().to(W)
    W.copy_(M @ W)
    if layer.bias is not None:
        layer.bias.copy_(M @ layer.bias)
    return model
