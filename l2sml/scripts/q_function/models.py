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

class SearchPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None, ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        input_dim = state_dim


        