import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticNetwork3Layer(nn.Module):
    """Critic Model for DDPG"""

    def __init__(self, state_size, action_size, nh1=64, nh2=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(CriticNetwork3Layer, self).__init__()
        self.FC1 = nn.Linear(state_size + action_size, nh1)
        self.FC2 = nn.Linear(nh1, nh2)
        self.Output = nn.Linear(nh2, 1)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        return self.Output(F.leaky_relu(self.FC2(F.leaky_relu(self.FC1(torch.cat((state, action), dim=1))))))
