"""Image transforms shared by fine-tuning, probing and evaluation."""
from torchvision import transforms as T


def input_size(processor):
    """Square input resolution of a model, read from its processor."""
    ip = processor.image_processor
    size = getattr(ip, "size", None)
    if isinstance(size, dict):
        return size.get("height", size.get("shortest_edge", 224))
    return 224


def eval_transform(mean, std, size):
    """Resize to 256/224 of the input size, center crop, normalize."""
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(),
                      T.Normalize(mean=mean, std=std)])


def train_transform(mean, std, size):
    """The fine-tuning augmentation: random crop and horizontal flip."""
    return T.Compose([T.Resize(int(size * 256 / 224)), T.RandomCrop(size),
                      T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=mean, std=std)])


def model_transforms(processor):
    """(train, eval) transforms at a model's own resolution and normalization."""
    ip = processor.image_processor
    size = input_size(processor)
    return (train_transform(ip.image_mean, ip.image_std, size),
            eval_transform(ip.image_mean, ip.image_std, size))


class ToRGB:
    """PIL image -> RGB (grayscale datasets such as MNIST)."""

    def __call__(self, img):
        return img.convert("RGB")


def with_rgb(transform):
    return T.Compose([ToRGB(), transform])
