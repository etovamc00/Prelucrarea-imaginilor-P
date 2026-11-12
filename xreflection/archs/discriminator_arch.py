from torch import nn
from xreflection.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Discriminator_VGG(nn.Module):
    def __init__(self, in_channels=3, use_sigmoid=True, **kwargs):
        super(Discriminator_VGG, self).__init__()

        def conv(*args, **kwargs):
            return nn.Conv2d(*args, **kwargs)

        num_groups = 32

        body = [
            conv(in_channels, 64, kernel_size=3, padding=1),  # 224
            nn.LeakyReLU(0.2),

            conv(64, 64, kernel_size=3, stride=2, padding=1),  # 112
            nn.GroupNorm(num_groups, 64),
            nn.LeakyReLU(0.2),

            conv(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2),

            conv(128, 128, kernel_size=3, stride=2, padding=1),  # 56
            nn.GroupNorm(num_groups, 128),
            nn.LeakyReLU(0.2),

            conv(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2),

            conv(256, 256, kernel_size=3, stride=2, padding=1),  # 28
            nn.GroupNorm(num_groups, 256),
            nn.LeakyReLU(0.2),

            conv(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=2, padding=1),  # 14
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),

            conv(512, 512, kernel_size=3, stride=2, padding=1),  # 7
            nn.GroupNorm(num_groups, 512),
            nn.LeakyReLU(0.2),
        ]

        tail = [
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        ]

        if use_sigmoid:
            tail.append(nn.Sigmoid())

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.body(x)
        out = self.tail(x)
        return out