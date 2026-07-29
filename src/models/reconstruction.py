"""CNN reconstruction / upscaling models.

These networks upscale a low-resolution input by 2x (or 4x for the deeper
variant) and are used as the reconstruction backbone in the memory-constrained
super-resolution pipeline.

Architectures are preserved exactly from the original project; only comments
and formatting were cleaned up.
"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block using reflection padding."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock2(nn.Module):
    """Residual block using zero padding."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class TransConv(nn.Module):
    """Single transposed-convolution upscaler (2x)."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        self.resblock = ResidualBlock2(64)

        # ConvTranspose2d upsamples the resolution by 2x
        self.upconv = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # 128 -> 256

        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))   # 128x128
        x = self.resblock(x)            # 128x128
        x = self.relu1(self.upconv(x))  # 256x256
        x = self.relu2(self.conv2(x))   # 256x256
        x = self.conv3(x)               # 256x256
        return x.clamp(0, 1)


class UDUCNN(nn.Module):
    """Up-Down-Up CNN: upsample, refine, downsample, upsample again (2x)."""

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # Base structure: 128 -> 256 -> 128 -> 256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)                     # 128x128
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)   # 256x256

        # Residual block 1: feature refinement at 256x256
        self.res1 = ResidualBlock2(64)

        self.down = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)           # 128x128
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   # 256x256

        # Residual block 2: just before the final output
        self.res2 = ResidualBlock2(32)

        self.output_conv = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))     # 128x128
        x = self.relu(self.up1(x))       # 256x256
        x = self.res1(x)                 # residual block 1
        x = self.relu(self.down(x))      # 128x128
        x = self.relu(self.up2(x))       # 256x256
        x = self.res2(x)                 # residual block 2
        x = self.output_conv(x)          # 256x256
        return x.clamp(0, 1)


class UUDCNN(nn.Module):
    """Up-Up-Down CNN: two transposed-conv upsamples followed by a strided
    downsample (net 4x upscaling)."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  # padding_mode not supported
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # padding_mode not supported
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2, padding_mode='reflect')
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.upsample(x)
        x = self.decoder(x)
        return x.clamp(0, 1)
