import torch
import torch.nn as nn
from test_myself import MyModel

def show_step_by_step():
    model = MyModel()

    print("==================================================")
    print("1. LAYER DEFINITIONS & WEIGHT SHAPES")
    print("==================================================")
    print(f"Layer 1: {model.layer1}")
    print(f"  -> Weight shape: {model.layer1.weight.shape}  ([out_features=5, in_features=1])")
    print(f"  -> Bias shape:   {model.layer1.bias.shape}    ([out_features=5])")
    print()
    print(f"Activation: {model.act}")
    print("  -> (No weights or biases, purely mathematical transformation)")
    print()
    print(f"Layer 2: {model.layer2}")
    print(f"  -> Weight shape: {model.layer2.weight.shape} ([out_features=10, in_features=5])")
    print(f"  -> Bias shape:   {model.layer2.bias.shape}   ([out_features=10])")
    print()

    print("==================================================")
    print("2. TENSOR DATAFLOW STEP-BY-STEP")
    print("==================================================")
    
    # Input batch of 4 samples, 1 feature each
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    print(f"[INPUT x]        Shape: {x.shape}")
    print(x)
    print()

    # Step A: Through Layer 1
    z1 = model.layer1(x)
    print(f"[AFTER layer1]   Shape: {z1.shape}  (x @ W1.T + b1)")
    print(f"  Example row 0: {z1[0].detach().tolist()}")
    print()

    # Step B: Through Activation (ReLU)
    a1 = model.act(z1)
    print(f"[AFTER ReLU]     Shape: {a1.shape}  (max(0, z1))")
    print(f"  Example row 0: {a1[0].detach().tolist()}")
    print()

    # Step C: Through Layer 2
    z2 = model.layer2(a1)
    print(f"[AFTER layer2]   Shape: {z2.shape} (a1 @ W2.T + b2)")
    print(f"  Example row 0 (first 4): {z2[0][:4].detach().tolist()}")
    print()

if __name__ == "__main__":
    show_step_by_step()
