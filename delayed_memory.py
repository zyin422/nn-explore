import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Synthetic Data Generation: Delayed Memory Task
def generate_delayed_memory_data(num_samples=2000, seq_len=100):
    torch.manual_seed(42)
    # Background noise across all steps: [N, seq_len, 1]
    x = torch.randn(num_samples, seq_len, 1) * 0.1
    
    # Random cue at step t=0: either -1.0 or +1.0
    cues = torch.randint(0, 2, (num_samples, 1)).float() * 2.0 - 1.0  # {-1, +1}
    x[:, 0, :] = cues
    
    # Target: 1 if cue > 0 else 0 (binary classification)
    y = (cues > 0).float()  # [N, 1]
    return x, y

# 3. Model Architectures
class VanillaRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)  # out: [B, L, H]
        final_h = out[:, -1, :]  # [B, H]
        logits = self.fc(final_h)  # [B, 1]
        return logits

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

        torch.nn.init.constant_(self.lstm.bias_ih_l0.data[hidden_size:2 * hidden_size], 1.0)
        torch.nn.init.constant_(self.lstm.bias_hh_l0.data[hidden_size:2 * hidden_size], 1.0)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)  # out: [B, L, H]
        final_h = out[:, -1, :]  # [B, H]
        logits = self.fc(final_h)  # [B, 1]
        return logits

# 4. Training & Gradient Measurement
def train_and_measure_model(model_class, name, x_train, y_train, epochs=10, lr=0.005):
    print(f"\n--- Training {name} ---")
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        x1_grad_norm_accum = 0.0
        num_batches = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Enable gradient tracking on the raw inputs to measure backprop to t=0
            inputs.requires_grad_(True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            # Measure gradient magnitude that reached step t=0
            x1_grad = inputs.grad[:, 0, :]  # [B, 1]
            x1_grad_norm_accum += x1_grad.abs().mean().item()

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            num_batches += 1

        avg_loss = total_loss / total
        acc = correct / total
        avg_x1_grad = x1_grad_norm_accum / num_batches

        if epoch in [1, 2, 5, 10]:
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Acc: {acc * 100:.1f}% | Grad at t=0: {avg_x1_grad:.6e}")

    return model

if __name__ == "__main__":
    X, Y = generate_delayed_memory_data(num_samples=2000, seq_len=100)
    train_and_measure_model(VanillaRNNModel, "Vanilla RNN (100 steps)", X, Y, epochs=10)
    train_and_measure_model(LSTMModel, "LSTM (100 steps)", X, Y, epochs=10)
