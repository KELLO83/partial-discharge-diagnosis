from __future__ import annotations

from torch import Tensor, nn

from ml.vision.src.schema import DEFAULT_NUM_CLASSES


class SmallPrpdCnn(nn.Module):
    def __init__(self, num_classes: int = DEFAULT_NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            _conv_block(64, 128),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
