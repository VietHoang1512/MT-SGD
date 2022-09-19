from torch import nn


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(21, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    net = FCNet()
    for name, param in net.named_parameters():
        print(name, param.shape)
