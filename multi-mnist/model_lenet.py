import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RegressionModel(torch.nn.Module):
    def __init__(self, n_tasks):
        super(RegressionModel, self).__init__()
        self.n_tasks = n_tasks
        self.conv1 = nn.Conv2d(1, 10, 9, 1)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 20, 50)
        self.encoder = nn.Sequential(self.conv1, self.conv2, self.fc1)

        for i in range(self.n_tasks):
            layer = nn.Linear(50, 10)
            # layer.apply(init_weights)
            setattr(self, "task_{}".format(i + 1), layer)
        # self.apply(init_weights)

    def forward(self, x, i=None):
        x = self.encode(x)
        return self.decode(x, i)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 20)
        x = F.relu(self.fc1(x))

        return x

    def decode(self, x, i=None):
        if i is not None:
            layer_i = getattr(self, "task_{}".format(i))
            return layer_i(x)

        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, "task_{}".format(i + 1))
            outs.append(layer(x))

        return outs

    def get_shared_parameters(self):
        params = [
            {"params": self.encoder.parameters(), "lr_mult": 1},
        ]
        return params

    def get_classifier_parameters(self):
        params = []
        for i in range(self.n_tasks):
            layer = getattr(self, "task_{}".format(i + 1))
            params.append({"params": layer.parameters()})
        return params


if __name__ == "__main__":
    net = RegressionModel(n_tasks=2)
    for name, param in net.named_parameters():
        print(name, param.size())
    print(net.get_parameters())
