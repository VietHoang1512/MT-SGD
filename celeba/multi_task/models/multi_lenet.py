import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1, channel_size, 1, 1) * 0.5).cuda())
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x, mask):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        mask = self.dropout2dwithmask(x, mask)
        if self.training:
            x = x * mask
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x, mask


class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        if mask is None:
            mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            x = x * mask
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), mask
