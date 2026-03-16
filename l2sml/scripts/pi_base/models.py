import torch
import torch.nn as nn
from torch.distributions import Normal

class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.activation(x + residual)

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=None,
        log_std_min=-20.0,
        log_std_max=2.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or []
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        layers = []
        input_dim = state_dim
        for layer_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, layer_dim))
            layers.append(nn.ReLU())
            input_dim = layer_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)

    def forward(self, state):
        features = self.backbone(state)
        mean = self.mean_layer(features)
        log_std = torch.clamp(self.log_std_layer(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def distribution(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, state):
        dist = self.distribution(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or []
        # if input_norm is not None:
        #     self.input_norm = torch.nn.Parameter(input_norm, requires_grad=False) 
        # else:
        #     self.input_norm = torch.nn.Parameter(torch.randn(state_dim), requires_grad=False)
        # if output_norm is not None:
        #     self.output_norm = torch.nn.Parameter(output_norm, requires_grad=False)
        # else:
        #     self.output_norm = torch.nn.Parameter(torch.randn(action_dim), requires_grad=False)

        if self.hidden_dims:
            hidden_dim = self.hidden_dims[0]
            assert all(dim == hidden_dim for dim in self.hidden_dims), "hidden_dims must all be the same"

            layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
            for _ in range(len(self.hidden_dims) - 1):
                layers.append(ResidualMLPBlock(hidden_dim))
            self.backbone = nn.Sequential(*layers)
            output_input_dim = hidden_dim
        else:
            self.backbone = nn.Identity()
            output_input_dim = state_dim

        self.output_layer = nn.Linear(output_input_dim, action_dim)

    def forward(self, state):
        x = self.backbone(state)
        return self.output_layer(x)

    def get_action(self, state, state_mean, state_std, action_mean, action_std):
        state = (state - state_mean) / state_std
        action = self(state)
        action = action * action_std + action_mean
        return action