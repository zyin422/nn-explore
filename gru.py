import torch
import torch.nn as nn


class GRUCellFromScratch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Reset gate projections
        self.w_xr = nn.Linear(input_size, hidden_size)
        self.w_hr = nn.Linear(hidden_size, hidden_size, bias=False)

        # Update gate projections
        self.w_xz = nn.Linear(input_size, hidden_size)
        self.w_hz = nn.Linear(hidden_size, hidden_size, bias=False)

        # Candidate state projections
        self.w_xh = nn.Linear(input_size, hidden_size)
        self.w_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        x_t:    Shape [B, D]
        h_prev: Shape [B, H]
        Returns:
        h_t:    Shape [B, H]
        """
        # TODO 1: Compute reset gate r_t using torch.sigmoid
        # r_t = ...

        # TODO 2: Compute update gate z_t using torch.sigmoid
        # z_t = ...

        # TODO 3: Compute candidate h_tilde using torch.tanh and (r_t * h_prev)
        # h_tilde = ...

        # TODO 4: Compute final state h_t via convex interpolation
        # h_t = ...

        return h_t