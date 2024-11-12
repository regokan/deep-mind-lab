import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class FCBody(nn.Module):
    def __init__(
        self,
        s_size,
        h_size,
        a_size,
        h_activation: nn.Module = None,
        a_activation: nn.Module = None,
    ):
        super().__init__()
        self.s_size = s_size
        self.h_size = h_size
        self.a_size = a_size
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.h_activation = h_activation
        self.a_activation = a_activation

        # Apply weight initialization
        self._init_weights()

    def forward(self, state, device="cpu"):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        state = state.view(-1, self.s_size)
        if self.h_activation is None:
            x = self.fc1(state)
        else:
            x = self.h_activation(self.fc1(state))
        if self.a_activation is None:
            x = self.fc2(x)
        else:
            x = self.a_activation(self.fc2(x))
        return x

    def _init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)
