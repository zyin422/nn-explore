from asyncio import runners
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, 10)

        self.act = nn.ReLU()
    def forward(self, x):
        z1 = self.pool(self.act(self.conv1(x)))
        z2 = self.pool(self.act(self.conv2(z1)))
        z3 = self.fc(self.flatten(z2))

        return z3