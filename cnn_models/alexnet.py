import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, embed_dim):
        super(AlexNet, self).__init__()
        self.embed_dim = embed_dim
        self.num_local = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.encoding = nn.Sequential(
                nn.Linear(256, self.embed_dim),
                nn.ReLU()
            )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.attrs_linear = nn.Linear(self.embed_dim, self.num_local, bias=False)


    def get_features(self, x):
        x = self.features(x)

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



def alexnet(pretrained=False, model_root=None, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root), strict=False)
        
    return model
