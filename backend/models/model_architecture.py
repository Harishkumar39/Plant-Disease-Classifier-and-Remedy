from torch import nn


class custom_block(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.base_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_features),
            nn.GELU(),
            nn.Conv2d(
                in_channels=out_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_features),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.base_block(x)


class custom_cnn(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.block_1 = custom_block(in_features=3, out_features=64)

        self.block_2 = custom_block(in_features=64, out_features=128)

        self.block_3 = custom_block(in_features=128, out_features=256)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        return self.classifier(self.block_3(self.block_2(self.block_1(x))))
