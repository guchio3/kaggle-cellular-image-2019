import torch.nn as nn


class myIdentity(nn.Identity):
    def __init__(self, in_features, *args, **kwargs):
        super(myIdentity, self).__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = in_features
