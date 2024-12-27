import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from torchvision import models

class DICM(nn.Module):
    def __init__(self):
        super(DICM, self).__init__()
        self.conv_fish = torch.Tensor(np.load("conv/fish_conv.npy")).cuda()
        self.regressor = nn.Sequential(
            nn.Conv2d(40, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        with torch.no_grad():
            x_ = F.conv2d(F.pad(x, ((int(3 / 2)), int((3 - 1) / 2), int(3 / 2), int((3 - 1) / 2))), self.conv_fish)
        x_ = self.regressor(x_)
        return x_


class FCNet(nn.Module):
    def __init__(self, load_weights=False):
        super(FCNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.conv = nn.Conv2d(512, 256, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.cc = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        self.bk = torch.Tensor(np.load("conv/bk_conv.npy")).cuda()       

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dicm1 = DICM()
        self.dicm2 = DICM()
        self.dicm3 = DICM()

        self.backend_feat = [512, 256, 128, 64, 32]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, batch_norm=True, dilation=True)
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x, branch):
        if branch == "train":
            u = self.frontend(x)
            u_ = self.relu(self.conv(u))
            h, w = u.size(2), u.size(3)
            u_s = self.pool(u)
            u_l = self.up(u)
            
            uu_ = self.cc(u)
            uu = self.dicm1(u)
            uu_s = self.dicm2(u_s)
            uu_l = self.dicm3(u_l)
            uu_s = nn.AdaptiveAvgPool2d(output_size=(h, w))(uu_s)
            uu_l = nn.AdaptiveAvgPool2d(output_size=(h, w))(uu_l)
            
            u_3 = torch.cat((uu_, uu), dim=1)
            u_3 = torch.cat((u_3, uu_s), dim=1)
            u_3 = torch.cat((u_3, uu_l), dim=1)
            x_m = torch.cat((u_, u_3), dim=1)    

            x_n = self.backend(x_m)
            out = self.output_layer(x_n)
            return u, self.bk, out
        else:
            u = self.frontend(x)
            u_ = self.relu(self.conv(u))
            h, w = u.size(2), u.size(3)
            u_s = self.pool(u)
            u_l = self.up(u)
            
            uu_ = self.cc(u)
            uu = self.dicm1(u)
            uu_s = self.dicm2(u_s)
            uu_l = self.dicm3(u_l)
            uu_s = nn.AdaptiveAvgPool2d(output_size=(h, w))(uu_s)
            uu_l = nn.AdaptiveAvgPool2d(output_size=(h, w))(uu_l)
            
            u_3 = torch.cat((uu_, uu), dim=1)
            u_3 = torch.cat((u_3, uu_s), dim=1)
            u_3 = torch.cat((u_3, uu_l), dim=1)
            x_m = torch.cat((u_, u_3), dim=1)
            x_n = self.backend(x_m)
            out = self.output_layer(x_n)
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)