# Deep Learning Mastery & Daily Learning Log

## Student Profile & Objectives
- **Target**: Reach high-speed technical proficiency in modern deep learning research papers, architectural comprehension, and novelty identification.
- **Pedagogical Standard**: Socratic inquiry, tensor dimension tracking at every layer, and freehand implementation from blank files without code dumps.

---

## Log Entries

### Baseline Session (Completed)
* **`nn.Module` Blueprint**:
  - `super().__init__()` initialization.
  - State registration (`__init__`) vs. dynamic computation graph construction (`forward`).
* **Tensor Shapes & Layers**:
  - Linear layers: weight matrix dimensions strictly parameterized as `[out_features, in_features]`.
  - Bias vector alignment: `[out_features]`.
  - Unbatched `[features]` vs. batched `[B, features]` tensor transformations.
  - Nonlinearity: Element-wise ReLU activation mechanics.
* **Autograd Graph Engine**:
  - Leaf nodes vs. intermediate graph nodes.
  - Backward pointer chains (`grad_fn` graph traversal).
  - Gradient accumulation mechanics: how `loss.backward()` traces backward and populates `.grad` attributes on leaf tensors.
* **Optimization & Execution**:
  - Canonical 5-step training loop written freehand from memory (`optimizer.zero_grad()`, forward pass, loss computation, `loss.backward()`, `optimizer.step()`).

---

### Session: Module 1 — Canonical Train & Eval Paradigm (Completed)
* **Execution Modes**:
  - `model.train()` vs. `model.eval()` internal boolean flag (`self.training`).
  - Layer-specific behaviors: Dropout random zeroing vs. deterministic scaling; BatchNorm running tallies vs. frozen global statistics.
* **Graph & Memory Isolation**:
  - `@torch.no_grad()` decorator / context manager disabling the Autograd recording engine.
  - Distinguishing memory optimization (`no_grad`) from layer behavioral switching (`eval`).
* **Loss vs. Accuracy Mechanics**:
  - The differentiability trap: why `torch.argmax(logits, dim=1)` is a non-differentiable step function with zero gradients and must NEVER be passed to `criterion`.
  - Cross-entropy dimensional geometry: Raw float logits `[B, C]` paired directly with integer class index targets `[B]`.
  - Scalar extraction: `loss.item()` and `.sum().item()` severing 0D tensors from the computation graph to prevent memory leaks and GPU synchronization bottlenecks.
* **Unbiased Metric Accumulation**:
  - Solving the "average of averages" trap: dynamically scaling batch loss by `inputs.size(0)` to properly weight uneven terminal mini-batches before dividing by `len(dataloader.dataset)`.
  - Implemented the full training and evaluation template freehand from memory in `train_eval.py`.

---

### Session: Module 2 — Convolutional Geometries & CNNs (Completed)
* **4D Tensor Manifolds**:
  - Image batch geometry: `[B, C, H, W]` ($B$: batch size, $C$: color/feature channels, $H$: height, $W$: width).
  - Spatial locality and translation invariance: why sliding local 3D filters resolves parameter explosion and avoids destroying 2D spatial relationships.
* **`nn.Conv2d` Anatomy**:
  - Weight tensor shape: `[C_out, C_in, K_H, K_W]` (stack of $C_{\text{out}}$ separate 3D filters, each spanning all $C_{\text{in}}$ channels simultaneously).
  - Bias tensor shape: `[C_out]` (one scalar bias per output feature map).
  - Non-square kernels, strides, and paddings using tuples `(K_H, K_W)`.
  - Feature maps as activation heatmaps tracking visual hierarchies from low-level edges to composite parts.
* **Spatial Dimension Arithmetic**:
  - Derived the spatial formula from first principles: $H_{\text{out}} = \lfloor \frac{H_{\text{in}} - K + 2P}{S} \rfloor + 1$.
  - "Same Padding" invariant: $K=3, P=1, S=1$ preserving spatial resolution ($H \rightarrow H$).
* **Downsampling & Receptive Fields**:
  - `nn.MaxPool2d(kernel_size=2, stride=2)`: Halving spatial resolution ($H, W \rightarrow H/2, W/2$) with zero learnable parameters while maintaining strong activations.
  - Global Average Pooling (GAP) mechanics for spatial aggregation.
* **Architecture Transition (`nn.Flatten`)**:
  - Flattening spatial manifolds from 4D `[B, C, H, W]` to 2D `[B, C * H * W]`.
  - Fully connected (`fc`) linear classification heads (`nn.Linear`).
  - Implemented and verified complete `SimpleCNN` freehand from memory in `cnn.py`.

---

### Session: Real-World Vision Pipelines & Optimization Mechanics (CIFAR-10)
* **Convolutional Weight & Bias Topology**:
  - `nn.Conv2d(in_channels, out_channels, kernel_size)` weight shape: strictly `[C_out, C_in, K_H, K_W]` (independent of input image spatial height/width).
  - Bias shape: 1D vector of shape `[C_out]` — exactly ONE scalar bias per output feature map, broadcast across spatial dimensions after summing across all input channels.
* **Input Preprocessing & Normalization Physics**:
  - Scaling $[0, 255] \rightarrow [0.0, 1.0]$ via `v2.ToDtype(torch.float32, scale=True)`.
  - Normalization via `v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))` mapping $[0.0, 1.0] \rightarrow [-1.0, 1.0]$.
  - The All-Positive Zig-Zag Trap: when pixels $x_i \ge 0$, weight gradients $\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot x_i$ share identical signs for all incoming connections, forcing all weights to increase or decrease in unison. Zero-centering enables independent bidirectional parameter updates.
  - Mitigating gradient explosion: unnormalized inputs ($x \approx 200$) scale weight gradients by $200\times$, destabilizing early optimization.
* **Target Geometry & Loss Pairing**:
  - Model outputs: Unbounded continuous logits `[B, C]` of type `torch.float32` required for continuous calculus and fine-grained gradient updates.
  - Ground truth targets: Discrete 1D integer class indices `[B]` of type `torch.long` (`int64`).
  - Why one-hot targets (`[B, C]`) are discarded: avoiding redundant $O(C)$ memory storage and sparse zero-multiplications in favor of $O(1)$ direct logit indexing (`-logits[b, target[b]]`).
* **Autograd Differentiability Invariant**:
  - Step function failure: `torch.argmax` has derivative $\frac{\partial}{\partial z} = 0$ almost everywhere, breaking the computation graph and producing zero gradients if passed to loss.
  - Role separation: `argmax` is strictly an evaluation metric for human accuracy tracking; `CrossEntropyLoss` (via continuous Softmax) drives all backpropagation.
* **Theoretical Loss Baselines & Diagnostics**:
  - Random initialization baseline under Cross-Entropy: $L_{\text{initial}} \approx -\ln(1/C) = \ln(C)$ ($\ln(10) \approx 2.302$, $\ln(100) \approx 4.605$).
  - Divergence between accuracy and loss: overconfidence on misclassified edge cases produces explosive $-\ln(p)$ penalties, causing loss to spike even while accuracy plateaus.
* **Cross-Entropy vs. Mean Squared Error (MSE)**:
  - Gradient saturation: MSE with sigmoid/softmax produces gradients $(p - y) \cdot p(1 - p)$, vanishing to zero when the model is confidently wrong ($p \approx 0$).
  - Linear error signal: Cross-Entropy cancels out the derivative of softmax, yielding a clean gradient $\frac{\partial L}{\partial z} = p - y$, delivering maximum update signal when errors are largest.
* **Empirical Verification on CIFAR-10 (Completed)**:
  - Epoch 0 baseline: Loss $\approx 2.307$ (confirming theoretical $-\ln(0.10) \approx 2.302$).
  - Epoch 5 convergence: Train Loss $0.8966$ / Train Acc $69.00\%$; Test Loss $0.9649$ / Test Acc $66.50\%$.
  - Clean generalization: minimal $\sim 2.5\%$ generalization gap demonstrating learned spatial representations over memorization.
  - End-to-end device integration: hardware-accelerated pipeline on Apple Silicon MPS via `inputs.to(device)` / `targets.to(device)` with Adam optimizer (`lr=0.001`).

---

### Upcoming Modules Roadmap
1. **Module 3: Recurrent Networks (RNNs)**
   - 3D sequence tensors: `[B, L, D]`.
   - Hidden recurrence: $h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$.
   - Temporal unrolling & vanishing gradient dynamics across time horizons.
2. **Module 4: Gated Memory (LSTMs)**
   - Additive error carousel ($C_t$) & gradient highways.
   - Gating vector mechanics: Forget ($f_t$), Input ($i_t$), Output ($o_t$).
   - State tuples `(h_t, c_t)` and sequence routing.
