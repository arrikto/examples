
# Copyright © 2022 Arrikto Inc.  All Rights Reserved.

"""PyTorch Model Definition.

This script defines a simple PyTorch CNN.
"""

import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2),
            padding=(3, 3), bias=False)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        features_in = self.model.fc.in_features
        self.model.fc = nn.Linear(features_in, 2)

    def forward(self, x):
        return self.model(x)
