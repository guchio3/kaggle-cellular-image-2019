import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class Network(nn.Module):
    def __init__(self, pretrained, n_classes):
        super(Network, self).__init__()
        self.model = EfficientNet.from_pretrained(
            'efficientnet-b5', num_classes=n_classes)

        new_conv = nn.Conv2d(
            6,
            48,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False)
        with torch.no_grad():
            new_conv.weight[:, :] = torch.stack(
                [torch.mean(self.model._conv_stem.weight, 1)] * 6, dim=1)
        self.model._conv_stem = new_conv

        # weight initialization
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        out = self.model(x)
        return out

    def named_children(self):
        for name, module in self.model.named_children():
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
