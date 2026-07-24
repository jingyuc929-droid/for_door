import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    """
    CNN encoder for PIE estimator net
    Input: (N, C, H, W)
    Output: (N, output_dim[0]*output_dim[1])
    CNN encoder for PIE estimator net
    Input: (N, C, H, W)
    Output: (N, output_dim[0]*output_dim[1])
    CNN encoder for PIE estimator net
    Input: (N, C, H, W)
    Output: (N, output_dim[0]*output_dim[1])
    CNN encoder for PIE estimator net
    Input: (N, C, H, W)
    Output: (N, output_dim[0]*output_dim[1])
    """

    def __init__(self, input_channels: int, num_layers: int, output_dim: [int, int], height: int, width: int) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("层数必须为正整数。")
        if output_dim[0] <= 0 or output_dim[1] <= 0:
            raise ValueError("输出维度必须为正整数。")
        if output_dim[0] * (2**(num_layers)) > height or output_dim[1] * (2**(num_layers)) > width:
            raise ValueError("卷积层数过大或者输出维度大。")
        self.input_channels_dim = int(input_channels)
        self.input_channels = int(input_channels)
        self.num_layers = int(num_layers)
        self.height = int(height)
        self.width = int(width)
        self.output_dim = output_dim
        self.features = nn.ModuleList()
        conv_channels = 1
        for i in range(num_layers - 1):
            # (N, C, H, W) -> (N, conv_channels, H, W)
            self.features.append(nn.Conv2d(self.input_channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
            # (N, conv_channels, H, W) -> (N, conv_channels, H, W)
            self.features.append(nn.ReLU(inplace=True))
            # (N, conv_channels, H, W) -> (N, conv_channels, H/2, W/2)
            self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.input_channels = conv_channels
        # (N, C, H, W) -> (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1))
        self.features.append(nn.Conv2d(self.input_channels, conv_channels, kernel_size=3, stride=1, padding=1, bias=False))
        # (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1)) -> (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1))
        self.features.append(nn.ReLU(inplace=True))
        # (N, conv_channels, H/2^(num_layers-1), W/2^(num_layers-1)) -> (N, conv_channels, 1, 1)
        self.features.append(nn.AdaptiveAvgPool2d(output_size=(output_dim[0], output_dim[1])))

        self.features_sequential = nn.Sequential(*self.features)
        # num_layers = 4
        # (N, C, H, W) -> (N, 1, H/2, W/2)
        # (N, 1, H/2, W/2) -> (N, 1, H/4, W/4)
        # (N, 1, H/4, W/4) -> (N, 1, H/8, W/8)
        # (N, 1, H/8, W/8) -> (N, 1, H/16, W/16)
        # (N, 1, H/16, W/16) -> (N, 1, H/32, W/32)

        # (N, 1, H/32, W/32) -> (N, 1, H/64, W/64)
        # (N, 1, H/64, W/64) -> (N, 1, H/128, W/128)

    def forward(self, x: Tensor) -> Tensor:
        # 支持 (N, C, H, W) 或 (N, C*H*W)
        if x.dim() == 2:
            expected_flatten = self.input_channels_dim * self.height * self.width
            if x.size(1) != expected_flatten:
                raise ValueError(
                    f"展平输入尺寸不匹配，期望 {expected_flatten} (= C*H*W = {self.input_channels_dim}*{self.height}*{self.width})，收到 {x.size(1)}"
                )
            # (N, C*H*W) -> (N, C, H, W)
            x = x.reshape(-1, self.input_channels_dim, self.height, self.width)
        elif x.dim() == 4:
            if x.size(1) != self.input_channels_dim:
                raise ValueError(f"输入通道数不匹配，期望 {self.input_channels_dim}，收到 {x.size(1)}")
            if x.size(2) != self.height or x.size(3) != self.width:
                raise ValueError(
                    f"输入空间尺寸不匹配，期望 (H, W)=({self.height}, {self.width})，收到 ({x.size(2)}, {x.size(3)})"
                )
        else:
            raise ValueError(f"预期输入维度为 2 或 4，收到形状 {tuple(x.shape)}")

        features_output: Tensor = self.features_sequential(x)  # (N, C, H, W) -> (N,1, output_dim[0], output_dim[1])
        return features_output.flatten(1)  # (N, 1, output_dim[0], output_dim[1]) -> (N, output_dim[0]*output_dim[1])
