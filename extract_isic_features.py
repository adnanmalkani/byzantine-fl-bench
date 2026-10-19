"""Fed-ISIC2019 main setting (docs/PROTOCOL.md, decided 2026-10-13): cache frozen
ImageNet EfficientNet-b0 features once per image, so federated learning only
trains the final layer.

- Model: FLamby's backbone (efficientnet-pytorch, ImageNet weights
  efficientnet-b0-355c32eb.pth), eval mode, no gradients. The feature of an
  image is what FLamby's Baseline feeds its final layer: extract_features ->
  global average pool -> flatten (1280 numbers; dropout is off in eval mode).
- Transform: FLamby's test transform (centre crop 200 px, ImageNet
  normalisation) for every image, train and test: no augmentation.
- Split: FLamby's fixed train/test split per centre (train_test_split).

Writes data/flamby/fed_isic2019/features_effb0_centercrop.pt: a dict with
"train"/"test" -> list over the 6 centres of (X float32 [N, 1280], y int64
[N], image names), plus "meta" (weights SHA-256, transform, versions,
device). Also checks, on 64 images, that the MPS features match the CPU
features.

Usage: python extract_isic_features.py   (about 2-5 min on the Mac's GPU)
"""
import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no version check over the network

import torch  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402
from flamby.datasets.fed_isic2019 import FedIsic2019  # noqa: E402

N_CENTERS = 6
WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth"
WEIGHTS_SHA256 = "355c32ebf5c7b2e19ed8f926daea5b6d71c8744a49aab3148fc90775fbaac5a3"
OUT = Path("data/flamby/fed_isic2019/features_effb0_centercrop.pt")


def backbone(device):
    m = EfficientNet.from_pretrained("efficientnet-b0")   # loads the cached weights
    m.set_swish(memory_efficient=False)                   # plain swish: inference only
    return m.eval().to(device)


@torch.no_grad()
def features(model, x):
    return model._avg_pooling(model.extract_features(x)).flatten(start_dim=1)


def dataset(center, train):
    """FLamby's centre/fold selection, always with FLamby's test transform."""
    ds = FedIsic2019(center=center, train=train)
    ds.augmentations = FedIsic2019(center=center, train=False).augmentations
    return ds


def extract(model, ds, device, bs=128, workers=8):
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    xs, ys = [], []
    for x, y in loader:
        xs.append(features(model, x.to(device)).cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def main():
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == WEIGHTS_SHA256, "weights changed"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = backbone(device)
    out = {"train": [], "test": []}
    t0 = time.time()
    for split in ("train", "test"):
        for c in range(N_CENTERS):
            ds = dataset(c, split == "train")
            X, y = extract(model, ds, device)
            assert X.shape == (len(ds), 1280) and torch.isfinite(X).all()
            names = [Path(p).stem for p in ds.image_paths]
            out[split].append((X, y, names))
            print(f"{split} centre {c}: {len(ds):5d} images, "
                  f"classes {torch.bincount(y, minlength=8).tolist()}  ({time.time() - t0:.0f} s)",
                  flush=True)

    # MPS vs CPU on the first 64 test images of centre 0
    ds = dataset(0, False)
    x = torch.stack([ds[i][0] for i in range(64)])
    cpu = features(backbone(torch.device("cpu")), x)
    diff = (out["test"][0][0][:64] - cpu).abs().max().item()
    scale = cpu.abs().max().item()
    print(f"MPS vs CPU, 64 images: max abs difference {diff:.2e} (feature scale {scale:.2f})")
    assert diff < 1e-3 * scale, "MPS features differ from CPU"

    total = sum(len(X) for s in out.values() for X, _, _ in s)
    out["meta"] = dict(weights=WEIGHTS.name, weights_sha256=WEIGHTS_SHA256,
                       transform="FLamby test transform: CenterCrop(200, 200), Normalize (ImageNet)",
                       feature="extract_features -> avg pool -> flatten (1280)",
                       torch=str(torch.__version__), device=str(device), mps_vs_cpu_max_abs_diff=diff,
                       n_images=total)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    torch.save(out, tmp)
    os.replace(tmp, OUT)
    print(f"{total} images in {time.time() - t0:.0f} s; saved {OUT} "
          f"({OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
