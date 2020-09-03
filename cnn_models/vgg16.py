import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from collections import OrderedDict
import math

__all__ = [
     'vgg16', 'vgg16bn'
]

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):
    def __init__(self, embed_dim):
        super(VGG, self).__init__()
        self.embed_dim = embed_dim
        self.num_local = 64
        self.layer1 = nn.Sequential(
                nn.Conv2d(3,  64, kernel_size=3, padding=1),
                # nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                # nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(64,  128, kernel_size=3, padding=1),
                # nn.BatchNorm2d(128), 
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                # nn.BatchNorm2d(128), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)

            )
        self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                # nn.BatchNorm2d(256), 
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                # nn.BatchNorm2d(256), 
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                # nn.BatchNorm2d(256), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                # nn.BatchNorm2d(512), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )        
        self.encoding = nn.Sequential(
                nn.Linear(512, self.embed_dim),
                nn.ReLU()
            )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.attrs_linear = nn.Linear(self.embed_dim, self.num_local, bias=False)

    def get_parameters(self):
        params = list(self.parameters())

        return params

    def get_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x

    def get_attrs(self, x):
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        x = self.encoding(x)
        attn = self.attrs_linear(x)
        attn = F.softmax(attn, dim=1).unsqueeze(dim=-1)
        x = x.unsqueeze(dim=-2)
        attrs = (x*attn).sum(dim=1)

        return attrs, attn.squeeze(dim=-1) 

    def forward(self, x):
        x = self.get_features(x) 
        feats = self.pooling(x).view(x.size(0), -1)  
        feats = self.encoding(feats) 
        attrs, attns = self.get_attrs(x)

        return feats, attrs, attns


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(**kwargs)
    if pretrained:
        resume = OrderedDict()
        for key, value in model_zoo.load_url(model_urls['vgg16']).items():
            if 'features' in key:
                key = key.lstrip('features.')
                loc, type = key.split('.')
                loc = int(loc)

                if loc > len(model.layer4) + len(model.layer3)  \
                       + len(model.layer2) + len(model.layer1) -1:
                    loc = loc - len(model.layer4) - len(model.layer3)  \
                              - len(model.layer2) - len(model.layer1)
                    key = str(loc) + '.' + str(type)
                    key = 'layer5.' + str(loc) + '.' + str(type)
                elif loc > len(model.layer3) + len(model.layer2)  \
                         + len(model.layer1) - 1:
                    loc = loc - len(model.layer3) - len(model.layer2)  \
                              - len(model.layer1)
                    key = 'layer4.' + str(loc) + '.' + str(type)
                elif loc > len(model.layer2) + len(model.layer1) -1:
                    loc = loc - len(model.layer2) - len(model.layer1)
                    key = 'layer3.' + str(loc) + '.' + str(type)
                elif loc > len(model.layer1) - 1:
                    loc = loc - len(model.layer1)
                    key = 'layer2.' + str(loc) + '.' + str(type)
                else:
                    key = 'layer1.' + str(loc) + '.' + str(type)
                # print (key, ': ', value.size())
                resume[key] = value
        model.load_state_dict(resume, strict=False)
    return model


def vgg16bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(**kwargs)
    if pretrained:
        resume = OrderedDict()
        for key, value in model_zoo.load_url(model_urls['vgg16bn']).items():
            if 'features' in key:
                key = key.lstrip('features.')
                loc, type = key.split('.')
                loc = int(loc)

                if loc > len(model.layer4) + len(model.layer3)  \
                       + len(model.layer2) + len(model.layer1) -1:
                    loc = loc - len(model.layer4) - len(model.layer3)  \
                              - len(model.layer2) - len(model.layer1)
                    key = str(loc) + '.' + str(type)
                    key = 'layer5.' + str(loc) + '.' + str(type)
                elif loc > len(model.layer3) + len(model.layer2)  \
                         + len(model.layer1) - 1:
                    loc = loc - len(model.layer3) - len(model.layer2)  \
                              - len(model.layer1)
                    key = 'layer4.' + str(loc) + '.' + str(type)
                elif loc > len(model.layer2) + len(model.layer1) -1:
                    loc = loc - len(model.layer2) - len(model.layer1)
                    key = 'layer3.' + str(loc) + '.' + str(type)
                elif loc > len(model.layer1) - 1:
                    loc = loc - len(model.layer1)
                    key = 'layer2.' + str(loc) + '.' + str(type)
                else:
                    key = 'layer1.' + str(loc) + '.' + str(type)
                # print (key, ': ', value.size())
                resume[key] = value

        model.load_state_dict(resume, strict=False)
    return model

