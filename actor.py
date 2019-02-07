import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork3Layer(nn.Module):
    """Actor Model for DDPG"""

    def __init__(self, state_size, action_size, nh1=64, nh2=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(ActorNetwork3Layer, self).__init__()
        self.FC1 = nn.Linear(state_size, nh1)
        self.FC2 = nn.Linear(nh1, nh2)
        self.Output = nn.Linear(nh2, action_size)

    def forward(self, state):
        return F.tanh(self.Output(F.leaky_relu(self.FC2(F.leaky_relu(self.FC1(state))))))
