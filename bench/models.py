import torch.nn as nn


class SmallCNN(nn.Module):
    """Two conv blocks + MLP head. No BatchNorm, so the full model state
    is just its parameters (keeps update vectors clean for aggregation)."""

    def __init__(self, in_ch: int = 1, num_classes: int = 10, img: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (img // 4) ** 2, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(in_ch: int, img: int, num_classes: int) -> nn.Module:
    return SmallCNN(in_ch=in_ch, num_classes=num_classes, img=img)
