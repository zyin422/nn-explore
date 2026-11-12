import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(1, 5)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(5, 10)

    def forward(self, x):
        z1 = self.layer1(x)
        a1 = self.act(z1)
        z2 = self.layer2(a1)

        return z2

def training_loop():
    model = MyModel()

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
    y = torch.zeros((4, 10))

    print("--- Before Training ---")
    with torch.no_grad():
        initial_preds = model(x)
        print("Initial Loss:", criterion(initial_preds, y).item())
        print("Sample 0 prediction (first 3 numbers):", initial_preds[0][:3].tolist())
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        print()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.6f}")
    
    print("\n--- After Training ---")
    with torch.no_grad():
        final_preds = model(x)
        print("Final Loss:", criterion(final_preds, y).item())
        print("Sample 0 prediction (first 3 numbers):", final_preds[0][:3].tolist())

if __name__ == "__main__":
    training_loop()