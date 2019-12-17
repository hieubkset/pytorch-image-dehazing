import torch.nn as nn
from torchvision import models


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.relu3_1 = nn.Sequential()
        for x in range(12):
            self.relu3_1.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.relu3_1(x)
        return out
