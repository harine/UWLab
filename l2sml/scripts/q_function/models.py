import torch
from torch import nn

class QFunction(nn.Module):
    def __init__(self, state_dim, action_chunk_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_chunk_dim = action_chunk_dim
        self.fc1 = nn.Linear(state_dim+action_chunk_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state, action_chunk):
        x = torch.relu(self.fc1(torch.cat([state, action_chunk], dim=1)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output_layer(x)
        