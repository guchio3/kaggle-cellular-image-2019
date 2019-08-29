import torch.nn as nn
import torchvision


class Network(nn.Module):
    def __init__(self, pretrained, n_classes):
        super(Network, self).__init__()
        # if you are interested in resnet archtecture, check below link.
        # [https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py]
        model = torchvision.models.resnet18(pretrained=pretrained)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # for arbital image size
        self.fc = nn.Linear(model.fc.in_features, n_classes)

        # weight initialization
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        out = self.fc(x.squeeze())

        return out

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
