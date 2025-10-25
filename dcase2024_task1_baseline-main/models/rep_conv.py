import torch
import torch.nn as nn


class RepConvBlock(nn.Module):
    """Rep-style parallel convolution block for training.

    Training-time: multiple branches (1x1, 3x1, 1x3, identity optional).
    Inference-time: use `fuse_repvgg()` to fuse branches into a single conv.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False,
                 branches=None):
        super().__init__()
        if branches is None:
            branches = {"1x1": True, "3x1": True, "1x3": True, "identity": False}

        self.branches_cfg = branches
        mid_channels = out_channels

        # central 3x3 conv (base)
        self.branch_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=groups, bias=bias)

        # optional 1x1 branch
        if branches.get("1x1", False):
            self.branch_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        stride=stride, padding=0, groups=groups, bias=bias)
        else:
            self.branch_1x1 = None

        # optional vertical/horizontal factorized branches
        if branches.get("3x1", False):
            self.branch_3x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1),
                                        stride=stride, padding=(1, 0), groups=groups, bias=bias)
        else:
            self.branch_3x1 = None

        if branches.get("1x3", False):
            self.branch_1x3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3),
                                        stride=stride, padding=(0, 1), groups=groups, bias=bias)
        else:
            self.branch_1x3 = None

        self.bn_3x3 = nn.BatchNorm2d(out_channels)
        self.bn_1x1 = nn.BatchNorm2d(out_channels) if self.branch_1x1 is not None else None
        self.bn_3x1 = nn.BatchNorm2d(out_channels) if self.branch_3x1 is not None else None
        self.bn_1x3 = nn.BatchNorm2d(out_channels) if self.branch_1x3 is not None else None

        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        out = self.bn_3x3(self.branch_3x3(x))
        if self.branch_1x1 is not None:
            out = out + self.bn_1x1(self.branch_1x1(x))
        if self.branch_3x1 is not None:
            out = out + self.bn_3x1(self.branch_3x1(x))
        if self.branch_1x3 is not None:
            out = out + self.bn_1x3(self.branch_1x3(x))
        return self.nonlinearity(out)


def _fuse_conv_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d into a single Conv2d (weights + bias)."""
    with torch.no_grad():
        w = conv.weight
        if conv.bias is None:
            b = torch.zeros(w.size(0), device=w.device)
        else:
            b = conv.bias

        bn_w = bn.weight
        bn_b = bn.bias
        bn_rm = bn.running_mean
        bn_rv = bn.running_var
        eps = bn.eps

        std = torch.sqrt(bn_rv + eps)
        w_fold = w * (bn_w / std).reshape(-1, 1, 1, 1)
        b_fold = (b - bn_rm) / std * bn_w + bn_b
        fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels,
                               kernel_size=conv.kernel_size, stride=conv.stride,
                               padding=conv.padding, groups=conv.groups, bias=True)
        fused_conv.weight.data = w_fold
        fused_conv.bias.data = b_fold
        return fused_conv


def fuse_repvgg_block(block: RepConvBlock):
    """Fuse all branches of a RepConvBlock into a single Conv2d + bias (in-place return of new conv module).

    Note: after fusion the resulting module no longer contains batchnorms or multiple branches.
    """
    # start with zeros weight for accumulator
    device = next(block.branch_3x3.parameters()).device
    fused_weight = None
    fused_bias = None

    # fuse main 3x3
    base = _fuse_conv_bn(block.branch_3x3, block.bn_3x3)
    fused_weight = base.weight.data
    fused_bias = base.bias.data

    # helper to add other branch contributions (must be reshaped to 3x3)
    def add_branch(conv, bn):
        f = _fuse_conv_bn(conv, bn)
        w = f.weight.data
        b = f.bias.data
        return w, b

    if block.branch_1x1 is not None:
        w1, b1 = add_branch(block.branch_1x1, block.bn_1x1)
        # pad 1x1 to 3x3 (center)
        pad = torch.zeros_like(fused_weight)
        k = w1.shape[-1]
        center = (fused_weight.shape[-1] - k) // 2
        pad[:, :, center:center + k, center:center + k] += w1
        fused_weight += pad
        fused_bias += b1

    if block.branch_3x1 is not None:
        w2, b2 = add_branch(block.branch_3x1, block.bn_3x1)
        # pad (3x1) to 3x3 horizontally center
        pad = torch.zeros_like(fused_weight)
        pad[:, :, :, 1:2] += w2
        fused_weight += pad
        fused_bias += b2

    if block.branch_1x3 is not None:
        w3, b3 = add_branch(block.branch_1x3, block.bn_1x3)
        pad = torch.zeros_like(fused_weight)
        pad[:, :, 1:2, :] += w3
        fused_weight += pad
        fused_bias += b3

    # build fused conv
    fused_conv = nn.Conv2d(block.branch_3x3.in_channels, block.branch_3x3.out_channels,
                           kernel_size=block.branch_3x3.kernel_size, stride=block.branch_3x3.stride,
                           padding=block.branch_3x3.padding, groups=block.branch_3x3.groups, bias=True)
    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias
    return fused_conv
