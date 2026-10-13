import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]
        prev = 28 * 28
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
