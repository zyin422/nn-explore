import torch
import torch.nn as nn



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    running_acc = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        prediction = model(inputs)
        preds = torch.argmax(prediction, dim=1)
        loss = criterion(prediction, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += (preds == targets).sum()

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_acc / len(dataloader.dataset)
    
    return (total_loss, total_acc)
    

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    running_acc = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        prediction = model(inputs)
        preds = torch.argmax(prediction, dim=1)
        loss = criterion(prediction, targets)

        running_loss += loss.item() * inputs.size(0)
        running_acc += (preds == targets).sum().item()

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_acc / len(dataloader.dataset)
    return (total_loss, total_acc)