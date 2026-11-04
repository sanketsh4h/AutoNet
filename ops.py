"""Candidate operations for the DARTS search space.

The first eight entries of ``OPS`` are the original DARTS primitives.  The rest
are the enhanced nodal operations added for AutoNet: larger receptive fields
(7x7 separable and factorised 7x1/1x7 convolutions) and squeeze-and-excitation
gated separable convolutions, which let an edge re-weight its own channels.
"""
import torch
import torch.nn as nn


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class FacConv(nn.Module):
    """ReLU - Conv(Kx1) - Conv(1xK) - BN."""

    def __init__(self, C_in, C_out, kernel, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, (kernel, 1), stride=(stride, 1), padding=(padding, 0), bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel), stride=(1, stride), padding=(0, padding), bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """ReLU - (dilated) depthwise conv - pointwise conv - BN."""

    def __init__(self, C_in, C_out, kernel, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel, stride, padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Depthwise-separable convolution applied twice, as in DARTS."""

    def __init__(self, C_in, C_out, kernel, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            DilConv(C_in, C_in, kernel, stride, padding, 1, affine),
            DilConv(C_in, C_out, kernel, 1, padding, 1, affine),
        )

    def forward(self, x):
        return self.op(x)


class SqueezeExcite(nn.Module):
    def __init__(self, C, reduction=4):
        super().__init__()
        hidden = max(C // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(C, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, C), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3)))[:, :, None, None]


class SESepConv(nn.Module):
    """Separable conv followed by a squeeze-and-excitation channel gate (enhanced op)."""

    def __init__(self, C_in, C_out, kernel, stride, padding, affine=True):
        super().__init__()
        self.conv = SepConv(C_in, C_out, kernel, stride, padding, affine)
        self.se = SqueezeExcite(C_out)

    def forward(self, x):
        return self.se(self.conv(x))


class Identity(nn.Module):
    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.0
        return x[:, :, ::self.stride, ::self.stride] * 0.0


class PoolBN(nn.Module):
    def __init__(self, pool_type, C, kernel, stride, padding, affine=True):
        super().__init__()
        if pool_type == 'avg':
            pool = nn.AvgPool2d(kernel, stride, padding, count_include_pad=False)
        else:
            pool = nn.MaxPool2d(kernel, stride, padding)
        self.op = nn.Sequential(pool, nn.BatchNorm2d(C, affine=affine))

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    """Halve the spatial size with two offset stride-2 1x1 convs so no pixel is dropped."""

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out - C_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        return self.bn(torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1))


OPS = {
    # --- original DARTS primitives -------------------------------------------
    'none':         lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine),
    # --- enhanced nodal operations (AutoNet) ---------------------------------
    'sep_conv_7x7':    lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine),
    'conv_7x1_1x7':    lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine),
    'se_sep_conv_3x3': lambda C, stride, affine: SESepConv(C, C, 3, stride, 1, affine),
    'se_sep_conv_5x5': lambda C, stride, affine: SESepConv(C, C, 5, stride, 2, affine),
}

DARTS_PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
                    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
AUTONET_PRIMITIVES = DARTS_PRIMITIVES + ['sep_conv_7x7', 'conv_7x1_1x7', 'se_sep_conv_3x3', 'se_sep_conv_5x5']
PRIMITIVE_SETS = {'darts': DARTS_PRIMITIVES, 'autonet': AUTONET_PRIMITIVES}


class MixedOp(nn.Module):
    """Softmax-weighted mixture of every candidate op on one edge."""

    def __init__(self, C, stride, primitives):
        super().__init__()
        self.ops = nn.ModuleList(OPS[p](C, stride, False) for p in primitives)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops))
