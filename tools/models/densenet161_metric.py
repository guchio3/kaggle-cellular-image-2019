import torch
import torch.nn as nn
import torchvision

from ..metrics import ArcMarginProduct
from ..layers import myIdentity


class Network(nn.Module):
    def __init__(self, pretrained, n_classes):
        super(Network, self).__init__()
        self.model = torchvision.models.densenet161(pretrained=pretrained)

        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        new_conv = nn.Conv2d(
            6,
            96,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False)
        with torch.no_grad():
            new_conv.weight[:, :] = torch.stack(
                [torch.mean(self.model.features.conv0.weight, 1)] * 6, dim=1)
        self.model.features.conv0 = new_conv
        self.model.classifier = myIdentity(in_features=self.model.classifier.in_features)
        self.arc = ArcMarginProduct(
            in_features=self.model.classifier.in_features,
            out_features=n_classes,
            easy_margin=True,
        ).to('cuda')

        # weight initialization
        if not pretrained:
            self._init_weight()

    def forward(self, x, labels=None):
        if labels is None:
            with torch.no_grad():
                labels = torch.zeros(x.shape[0], device='cuda')
        features = self.model(x)
        out = self.arc(features, labels)
        return out

    def named_children(self):
        for name, module in self.model.named_children():
            if name == 'classifier':
                name = 'fc'
            yield name, module

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
