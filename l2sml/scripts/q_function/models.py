import torch
from torch import nn
from torch.distributions import Normal

class QFunction(nn.Module):
    def __init__(self, state_dim, action_chunk_dim, hidden_dims):
        super().__init__()
        self.state_dim = state_dim
        self.action_chunk_dim = action_chunk_dim
        self.hidden_dims = hidden_dims
        layers = []
        input_dim = state_dim + action_chunk_dim
        for layer_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, layer_dim))
            layers.append(nn.ReLU())
            input_dim = layer_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, state, action_chunk):
        x = self.backbone(torch.cat([state, action_chunk], dim=1))
        return self.output_layer(x)

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        layers = []
        input_dim = state_dim
        for layer_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, layer_dim))
            layers.append(nn.ReLU())
            input_dim = layer_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, state):
        x = self.backbone(state)
        return self.output_layer(x)

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
        self.hidden_dims = hidden_dims
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

class SearchPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None, ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        input_dim = state_dim


        