import torch
from torch import nn
import torchvision.ops


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.modulation = modulation

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        nn.init.constant_(self.p_conv.bias, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            nn.init.constant_(self.m_conv.bias, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

        self.weight = nn.Parameter(torch.empty(outc, inc, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(outc))
        else:
            self.bias = None

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        return tuple(g * 0.1 if g is not None else None for g in grad_input)

    def forward(self, x):
        offset = self.p_conv(x)
        mask = torch.sigmoid(self.m_conv(x)) if self.modulation else None

        return torchvision.ops.deform_conv2d(
            x, offset, self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            mask=mask
        )
