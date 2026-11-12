import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. MODEL DEFINITION
# =====================================================================
class SimpleNet(nn.Module):
    """
    Inherits from nn.Module, the base class for all neural network modules in PyTorch.
    """
    def __init__(self):
        # Always call the parent class constructor to properly register parameters,
        # submodules, buffers, and hooks.
        super().__init__()
        
        # Layer 1: Linear transformation (y = xW^T + b)
        # in_features = 1, out_features = 5
        self.layer1 = nn.Linear(in_features=1, out_features=5, bias=True)
        
        # Non-linear activation function: ReLU(z) = max(0, z)
        self.act = nn.ReLU()
        
        # Layer 2: Linear transformation
        # in_features = 5 (must match layer1's out_features), out_features = 10
        self.layer2 = nn.Linear(in_features=5, out_features=10, bias=True)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        PyTorch builds the dynamic computation graph here during runtime.
        """
        z1 = self.layer1(x)
        a1 = self.act(z1)
        z2 = self.layer2(a1)
        return z2

def inspect_model_and_shapes():
    print("=" * 60)
    print("STEP 1: INSPECTING LAYERS, WEIGHTS, AND BIASES")
    print("=" * 60)
    
    model = SimpleNet()
    print("Model structure:\n", model)
    print("\nLayer parameter shapes:")
    # Layer 1 weights & bias
    print(f"  layer1.weight shape: {model.layer1.weight.shape}  -> (out_features=5, in_features=1)")
    print(f"  layer1.bias shape:   {model.layer1.bias.shape}    -> (out_features=5,)")
    
    # Layer 2 weights & bias
    print(f"  layer2.weight shape: {model.layer2.weight.shape} -> (out_features=10, in_features=5)")
    print(f"  layer2.bias shape:   {model.layer2.bias.shape}   -> (out_features=10,)")
    print()

    print("=" * 60)
    print("STEP 2: BATCHED VS UNBATCHED INPUTS & SHAPE TRACKING")
    print("=" * 60)

    # 1. Batched Input: batch_size = 4, in_features = 1 -> Shape: (4, 1)
    x_batched = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    print(f"[Batched Input] Shape: {x_batched.shape} (batch_size=4, features=1)")

    # Manual forward step-by-step tracking for Batched Input
    z1_b = model.layer1(x_batched)
    print(f"  -> After layer1: {z1_b.shape} (batch_size=4, features=5)")
    a1_b = model.act(z1_b)
    print(f"  -> After ReLU:   {a1_b.shape} (batch_size=4, features=5)")
    z2_b = model.layer2(a1_b)
    print(f"  -> After layer2: {z2_b.shape} (batch_size=4, features=10)")
    print()

    # 2. Unbatched Input: 1D tensor -> Shape: (1,)
    # PyTorch nn.Linear supports 1D unbatched input (single sample)
    x_unbatched = torch.tensor([1.5])
    print(f"[Unbatched Input] Shape: {x_unbatched.shape} (single sample of 1 feature)")

    # Manual forward step-by-step tracking for Unbatched Input
    z1_u = model.layer1(x_unbatched)
    print(f"  -> After layer1: {z1_u.shape} (features=5)")
    a1_u = model.act(z1_u)
    print(f"  -> After ReLU:   {a1_u.shape} (features=5)")
    z2_u = model.layer2(a1_u)
    print(f"  -> After layer2: {z2_u.shape} (features=10)")
    print()

    # Note on standard convention:
    # While 1D input works for nn.Linear, nearly all modern deep learning
    # architectures (especially Conv2d, BatchNorm, MultiheadAttention)
    # strictly require a batch dimension, e.g., shape (1, 1) instead of (1,).
    # You can add a batch dimension with x.unsqueeze(0) or x[None, :].

def run_training_loop():
    print("=" * 60)
    print("STEP 3: TRAINING LOOP (OPTIMIZER STEP & LOSS.BACKWARD)")
    print("=" * 60)

    model = SimpleNet()
    
    # Target / Ground truth for 4 samples: shape (4, 10)
    x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]]) # (4, 1)
    y_train = torch.zeros((4, 10))                        # (4, 10)

    # Loss function (Mean Squared Error: (1/N) * sum((y_pred - y_true)^2))
    criterion = nn.MSELoss()

    # Optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.01
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 5
    for epoch in range(1, epochs + 1):
        # 1. Zero out previous gradients
        # PyTorch accumulates gradients by default (loss.backward() adds to param.grad).
        # We must clear them before calculating gradients for the new batch.
        optimizer.zero_grad()

        # 2. Forward pass (compute prediction)
        # Note: Call model(x), NEVER model.forward(x). Calling model() executes hooks.
        predictions = model(x_train)

        # 3. Compute loss
        loss = criterion(predictions, y_train)

        # 4. Backward pass (Backpropagation via Autograd)
        # Computes d(loss)/d(param) for every parameter where requires_grad=True,
        # storing the gradient in param.grad.
        loss.backward()

        # Check gradient of layer1.weight before stepping
        grad_norm = model.layer1.weight.grad.norm().item()

        # 5. Optimizer step (Parameter update)
        # Updates parameters using rule: w = w - lr * grad
        optimizer.step()

        print(f"Epoch {epoch:02d} | Loss: {loss.item():.6f} | layer1.weight.grad norm: {grad_norm:.6f}")

if __name__ == "__main__":
    inspect_model_and_shapes()
    run_training_loop()
