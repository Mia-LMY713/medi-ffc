import torch
import torch.nn as nn
from deform_conv_v2 import DeformConv2d
from ffc import FFC_BN_ACT, ConcatTupleLayer


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1

        ratio_in = 0.5
        self.catLayer = ConcatTupleLayer()

        self.ffc1=nn.Sequential(
            FFC_BN_ACT(image_channels, n_channels, kernel_size=3, padding=1, dilation=1, ratio_gin=0, ratio_gout=ratio_in)
        )
        #self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ffc2=nn.Sequential(
            FFC_BN_ACT(n_channels, n_channels, kernel_size=3, padding=1, dilation=1, ratio_gin=0, ratio_gout=ratio_in)
        )
        #self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)


        #DB：deformable conv+ReLu
        self.db=nn.Sequential(
            DeformConv2d(inc=n_channels, outc=n_channels,
                     kernel_size=kernel_size, padding=padding, bias=False, modulation=True),
            nn.ReLU(inplace=True))
        eb_layers = []
        for _ in range(depth-2):
            if _ == 11:  #
                eb_layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=2, bias=False, dilation=2))
            else:
                eb_layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            eb_layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            eb_layers.append(nn.ReLU(inplace=True))
        self.eb=nn.Sequential(*eb_layers)
        self.rb=nn.Sequential(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                        kernel_size=kernel_size, padding=padding, bias=False))
        self._initialize_weights()

    def forward(self, x):
        y = x
        out=self.ffc1(x)
        out=self.catLayer(out)
        out = self.ffc2(out)
        out = self.catLayer(out)
        # print("after ffc:", out.shape)
        out = self.db(out)
        # print("after db:", out.shape)
        out= self.eb(out)
        # print("after eb:", out.shape)
        out=self.rb(out)
        # print("after rb:", out.shape)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('init weights finished')

    def model_name(self):
        return 'DnCNN_ffc'

class DnCNN_skip(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN_skip, self).__init__()
        kernel_size = 3
        padding = 1
        ratio_in = 0.5
        self.catLayer = ConcatTupleLayer()

        # self.ffc=nn.Sequential(
        #     FFC_BN_ACT(image_channels, n_channels, kernel_size=3, padding=1, dilation=1, ratio_gin=0, ratio_gout=ratio_in)
        # )
        self.ffc1 = nn.Sequential(
            FFC_BN_ACT(image_channels, n_channels, kernel_size=3, padding=1, dilation=1, ratio_gin=0,
                       ratio_gout=ratio_in)
        )
        #self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ffc2 = nn.Sequential(
            FFC_BN_ACT(n_channels, n_channels, kernel_size=3, padding=1, dilation=1, ratio_gin=0, ratio_gout=ratio_in)
        )
        #self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)


        #DB：deformable conv+ReLu
        self.db=nn.Sequential(
            DeformConv2d(inc=n_channels, outc=n_channels,
                     kernel_size=kernel_size, padding=padding, bias=False, modulation=True),
            nn.ReLU(inplace=True))
        eb_layers = []
        for _ in range(depth-2):
            if _ == 11:  #
                eb_layers.append(nn.Conv2d(in_channels=int(n_channels), out_channels=int(n_channels),
                                           kernel_size=kernel_size, padding=2, bias=False, dilation=2))
            else:
                eb_layers.append(nn.Conv2d(in_channels=int(n_channels), out_channels=int(n_channels),
                                           kernel_size=kernel_size, padding=padding, bias=False))
            eb_layers.append(nn.BatchNorm2d(int(n_channels), eps=0.0001, momentum=0.95))
            eb_layers.append(nn.ReLU(inplace=True))
        self.eb=nn.Sequential(*eb_layers)
        self.fuse_eb = nn.Conv2d(n_channels * 2, n_channels, kernel_size=1, bias=False)
        self.fuse_rb = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1, bias=False)
        self.rb=nn.Sequential(nn.Conv2d(in_channels=int(n_channels), out_channels=image_channels,
                                        kernel_size=kernel_size, padding=padding, bias=False))
        self._initialize_weights()

    def forward(self, x):
        y = x
        out1 = self.ffc1(x)
        out1 = self.catLayer(out1)
        out1 = self.ffc2(out1)
        out1 = self.catLayer(out1)
        # print("after ffc:", out.shape)
        out2 = self.db(out1)
        # print("after db:", out.shape)
        out3= self.eb(self.fuse_eb(torch.cat([out1, out2], dim=1)))
        # print("after eb:", out.shape)
        out = self.rb(self.fuse_rb(torch.cat([out1, out2, out3], dim=1)))
        # print("after rb:", out.shape)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('init weights finished')

    def model_name(self):
        return 'DnCNN_ffc_skip'

