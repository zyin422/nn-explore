import torch
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.nn as nn
from cnn import SimpleCNN

from train_eval import train_one_epoch, evaluate

import torch.optim as optim

# image processing pipeline; converts to image, converts to float32 and scales to 0.0 to 1.0, and normalizes between 
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

# images, targets = next(iter(train_loader))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    train_loss, train_acc = train_one_epoch(model=model, dataloader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
    test_loss, test_acc = evaluate(model=model, dataloader=test_loader, criterion=criterion, device=device)
    print(f"train_loss: {train_loss}   train_acc: {train_acc}")
    print(f"test_loss: {test_loss}   test_acc: {test_acc}")