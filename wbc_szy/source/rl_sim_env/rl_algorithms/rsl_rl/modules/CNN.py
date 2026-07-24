import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    """
    CNN encoder for PIE estimator net
    """

    def __init__(self, input_channels: int, output_dim:int) -> None:
        super().__init__()
        self.input_channels_dim = int(input_channels)
        self.input_channels = int(input_channels)
        self.output_dim = output_dim
        self.features = nn.ModuleList()
        conv_channels = 16
        # (N, C, H, W) -> (N, conv_channels, H, W)
        # 2×72×128 -> 16×70×126
        self.features.append(nn.Conv2d(self.input_channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))

        # (N, conv_channels, H, W) -> (N, conv_channels, H, W)
        # 16×70×126 -> 16×70×126
        self.features.append(nn.ELU())

        # (N, conv_channels, H, W) -> (N, conv_channels, H/2, W/2)
        # 16×70×126 -> 16×35×63
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.input_channels = conv_channels
        conv_channels = conv_channels * 2

        # (N, C, H, W) -> (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1))
        # 16×35×63 -> 32×35×63
        self.features.append(nn.Conv2d(self.input_channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))

        # (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1)) -> (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1))
        # 16×35×63 -> 32×35×63
        self.features.append(nn.ELU())

        # (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1)) -> (N, conv_channels, H/2^num_layers, W/2^num_layers)
        # 32×35×63 -> 32×17×31
        self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.input_channels = conv_channels
        conv_channels = conv_channels * 2

        # (N, C, H, W) -> (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1))
        # 16×35×63 -> 32×35×63
        self.features.append(nn.Conv2d(self.input_channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))

        # (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1)) -> (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1))
        # 16×35×63 -> 32×35×63
        self.features.append(nn.ELU())

        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))

        # 32×17×31 -> 1×output_dim[0]×output_dim[1]
        self.features.append(nn.Flatten())

        self.features.append(nn.Linear(conv_channels, output_dim))

        self.features.append(nn.ELU())

        self.features_sequential = nn.Sequential(*self.features)

    def forward(self, x: Tensor) -> Tensor:
        # 支持 (N, C, H, W) 或 (N, C*H*W)
        features_output: Tensor = self.features_sequential(x)
        return features_output
