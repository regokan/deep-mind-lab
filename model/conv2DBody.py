from torch import nn
from torch.nn import init


class Conv2DBody(nn.Module):
    def __init__(
        self,
        in_channels,
        h_channels,
        out_channels,
        kernel_size,
        a_stride,
        h_stride,
        a_padding=0,
        h_padding=0,
        h_activation: nn.Module = None,
        a_activation: nn.Module = None,
        h_bias=False,
        a_bias=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, h_channels, kernel_size, h_stride, h_padding, bias=h_bias
        )
        self.conv2 = nn.Conv2d(
            h_channels, out_channels, kernel_size, a_stride, a_padding, bias=a_bias
        )
        self.h_activation = h_activation
        self.a_activation = a_activation
        self.h_bias = h_bias
        self.a_bias = a_bias

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        if self.h_activation is not None:
            x = self.h_activation(x)
        x = self.conv2(x)
        if self.a_activation is not None:
            x = self.a_activation(x)
        return x

    def _init_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="relu")
        if self.h_bias:
            self.conv1.bias.data.fill_(0)
        if self.a_bias:
            self.conv2.bias.data.fill_(0)
