import torch
from torch import nn
from torch.nn import functional as F
import sys

def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn

@export
def Net_local_liner(**kwargs):
    model = Net_local_liner(**kwargs)
    return model

@export
def Net_liner( **kwargs):
    model = Net_liner(**kwargs)
    return model


class Net_local_liner(nn.Module):
    def __init__(self, hidden_size=7000, activation='leaky_relu', orthogonal = True, num_classes=10):
        super(Net_local_liner, self).__init__()
        self.bn = nn.BatchNorm1d(128)
        self.convd1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size = 3)
        self.pool1 = nn.AdaptiveMaxPool1d(50)
        self.activation = getattr(F, activation)

        # if activation in ['relu', 'leaky_relu']:
        #     nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain(activation))
        # else:
        #     torch.nn.init.xavier_uniform(self.fc1.weight, gain=1)
        # if orthogonal:
        #     nn.init.orthogonal_(self.fc1.weight)

        # if activation in ['relu', 'leaky_relu']:
        #     nn.init.xavier_uniform_(self.convd1.weight, gain=nn.init.calculate_gain(activation))
        # else:
        #     torch.nn.init.xavier_uniform(self.convd1.weight, gain=1)
        if orthogonal:
            nn.init.orthogonal_(self.convd1.weight)

        self.fc2 = nn.Linear(50*8, num_classes, bias=False)  # ELM do not use bias in the output layer.

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.convd1(x)
        x = self.pool1(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x

    def forwardToHidden(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.convd1(x)
        x = self.pool1(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        return x


class Net_liner(nn.Module):
    def __init__(self, d = 784, hidden_size=100, activation='leaky_relu', orthogonal=True, num_classes=10):
        super(Net_liner, self).__init__()
        self.bn = nn.BatchNorm1d(d)
        self.fc1 = nn.Linear(d, hidden_size)
        self.activation = getattr(F, activation)

        if activation in ['relu', 'leaky_relu']:
            nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain(activation))
        else:
            torch.nn.init.xavier_uniform(self.fc1.weight, gain=1)
        if orthogonal:
            nn.init.orthogonal_(self.fc1.weight)

        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)  # ELM do not use bias in the output layer.

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def forwardToHidden(self, x):
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.fc1(x)
        x = self.activation(x)
        return x

class CNNNet(nn.Module):
    def __init__(self,num_classes=10):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(10, 210, kernel_size=4, padding=1)
        self.fc2 = nn.Linear(7560,num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc2(x)
        return x

    def forwardToHidden(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

