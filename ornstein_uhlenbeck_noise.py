import numpy as np

class OrnsteinUhlenbeckNoise():
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, mu=None, theta=0.15, sigma=0.2):
        self.mu = mu if mu else np.zeros(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()
    
    def reset(self):
        self.state = np.copy(self.mu)
    
    def sample(self):
        dmean = self.state - self.mu
        dx = -self.theta * dmean + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state
    
    def state_dict(self):
        return {
            'size': self.size,
            'mu': self.mu,
            'theta': self.theta,
            'sigma': self.sigma,
            'state': self.state
        }
    
    def load_state_dict(self, state_dict):
        self.size = state_dict['size']
        self.mu = state_dict['mu']
        self.theta = state_dict['theta']
        self.sigma = state_dict['sigma']
        self.state = state_dict['state']