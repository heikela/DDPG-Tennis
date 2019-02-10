import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas
import actor
import critic
from ornstein_uhlenbeck_noise import *
from replaybuffer import *
from utils import soft_update, mutate_param
import numpy as np

class DdpgAgent():
    """An agent that interacts with the environment and does most of the work of the DDPG algorightm,
    manipulating the relevant actor and critic networks and their target copies.
    """
    def __init__(self,
                 buffer_size=5e4,
                 steps_between_updates=1,
                 batch_size=64,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 tau=1e-4,
                 gamma=0.99,
                 action_space_limit=1,
                 make_actor=None,
                 make_critic=None,
                 state_size=24,
                 action_size=2,
                 device=None,
                 name="agent-0"):
        """Initialize the DdpgAgent object

        Params
        ======
            buffer_size (Number): Size of the experience replay buffer to maintain
            steps_between_updates (int): How many time steps to act in the environment between training steps
            batch_size (int): Number of experience samples in each training minibatch
            actor_lr (float): Adam learning rate for training the actor network
            critic_lr (float): Adam learning rate for training the critic network
            tau (float): the rate at which soft updates update target network weights
            gamma (float): discount factor for future rewards in TD-update
            action_space_limit (float): maximum absolute value allowed for each component of the action vector
            make_actor (function): if supplied, used to construct actor neural networks instead of the default implementation.
                                   Will be supplied with state_size and action_size as positional parameters.
            make_critic (function): if supplied, used to construct critic neural networks instead of the default implementation.
                                   Will be supplied with state_size and action_size as positional parameters.
            state_size (int): dimension of state vector
            action_size (int): dimension of action vector
            device (cuda device): cuda device to place computations on. If not supplied, will use cuda:0 if available, otherwise cpu.
            name (string): name for the agent
        """
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor_network_local = (make_actor(state_size, action_size) if make_actor else actor.ActorNetwork3Layer(state_size, action_size)).to(self.device)
        self.critic_network_local = (make_critic(state_size, action_size) if make_critic else critic.CriticNetwork3Layer(state_size, action_size)).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_network_local.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network_local.parameters(), lr=critic_lr)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_network_target = (make_actor(state_size, action_size) if make_actor else actor.ActorNetwork3Layer(state_size, action_size)).to(self.device)
        self.critic_network_target = (make_critic(state_size, action_size) if make_critic else critic.CriticNetwork3Layer(state_size, action_size)).to(self.device)
        self.noise = OrnsteinUhlenbeckNoise(action_size) # TODO treat sigma as a hyperparameter
        self.experience = ReplayBuffer(int(buffer_size))
        self.steps_between_updates = steps_between_updates
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.t_step = 0
        self.action_space_limit = action_space_limit
        self.name = name
        self.history = pandas.DataFrame(columns=['episode', 'return', 'actor_lr', 'critic_lr', 'tau', 'gamma'])
        self.history.set_index('episode')
        self.episode_return = 0
        self.episode = 1
    
    def act(self, state, noise=True):
        """Select action based on state
        
        Determine the action preferred by the actor network and add exploratory noise.
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device)
            self.actor_network_local.eval()
            action = self.actor_network_local(state_tensor).cpu().data.numpy()
            self.actor_network_local.train()
            if noise:
                action += self.noise.sample()
            return np.clip(action, -self.action_space_limit, self.action_space_limit)
    
    def step(self, state, action, reward, next_state, done):
        """Record experience and learn based on observed step"""
        self.experience.add(state, action, reward, next_state, done)
        self.episode_return += reward
        if done:
            self.history = self.history.append([{
                'episode': self.episode,
                'return': self.episode_return,
                'actor_lr': self.actor_lr,
                'critic_lr': self.critic_lr,
                'gamma': self.gamma,
                'tau': self.tau}])
            self.episode += 1
            self.episode_return = 0
        self.t_step += 1
        if self.t_step % self.steps_between_updates == 0:
            self.t_step = 0
            self.learn()
    
    def learn(self):
        """Implement the learning part of the inner loop of DDPG"""
        if len(self.experience) < self.batch_size:
            return
        # sample a batch of experience
        experiences = self.experience.sample(self.batch_size)
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        # update critic
        # calculate targets
        with torch.no_grad():
            next_actions = self.actor_network_target(next_states)
            Q_targets_next = self.critic_network_target(next_states, next_actions)
            Q_targets = self.gamma * Q_targets_next + rewards
        Q_predicted = self.critic_network_local(states, actions)
        critic_loss = F.mse_loss(Q_predicted, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # update actor
        predicted_actions = self.actor_network_local(states)
        actor_loss = -self.critic_network_local(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # soft update targets towards locals
        self.soft_update()
    
    def soft_update(self):
        """Apply a soft update from local actor and critic networks to their target counterparts."""
        soft_update(self.actor_network_local, self.actor_network_target, self.tau)
        soft_update(self.critic_network_local, self.critic_network_target, self.tau)
    
    def forward_state_dict(self):
        """The parts of the state that relate to calculating predicitons from the model and choosing actions"""
        return {
            'actor': self.actor_network_local.state_dict(),
            'critic': self.critic_network_local.state_dict(),
            'actor_target': self.actor_network_target.state_dict(),
            'critic_target': self.critic_network_target.state_dict(),
            'noise': self.noise.state_dict()
        }
    
    def load_forward_state_dict(self, state_dict):
        """Set the relevant parts of the model based on a previously saved forward state dictionary."""
        self.actor_network_local.load_state_dict(state_dict['actor'])
        self.critic_network_local.load_state_dict(state_dict['critic'])
        self.actor_network_target.load_state_dict(state_dict['actor_target'])
        self.critic_network_target.load_state_dict(state_dict['critic_target'])
        self.noise.load_state_dict(state_dict['noise'])
    
    def learning_state_dict(self):
        """A dictionary of the state relating directly to the training process"""
        return {
            'actor_optim_state': self.actor_optimizer.state_dict(),
            'critic_optim_state': self.critic_optimizer.state_dict(),
            'replay_buffer': self.experience.state_dict()
        }
    
    def load_learning_state_dict(self, state_dict):
        """Set optimizer state and replay buffer content based on a state dict returned earlier by learning_state_dict()"""
        self.actor_optimizer.load_state_dict(state_dict['actor_optim_state'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optim_state'])
        self.experience.load_state_dict(state_dict['replay_buffer'])
    
    def state_dict(self):
        """State dict combining the forward calculation weights and learning state"""
        return {
            'forward': self.forward_state_dict(),
            'learning': self.learning_state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Set state from the combined state dict"""
        self.load_forward_state_dict(state_dict['forward'])
        self.load_learning_state_dict(state_dict['learning'])
    
    def hyperparameter_dict(self):
        """A dictionary of hyperparameters used in the current training process for this actor"""
        return {
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'gamma': self.gamma,
            'tau': self.tau
        }
    
    def load_hyperparameter_dict(self, hyperparameter_dict):
        """Set the hyperparameters for the training process for this actor"""
        for p in self.actor_optimizer.param_groups:
            p['lr'] = hyperparameter_dict['actor_lr']
        for p in self.critic_optimizer.param_groups:
            p['lr'] = hyperparameter_dict['critic_lr']
        self.gamma = hyperparameter_dict['gamma']
        self.tau = hyperparameter_dict['tau']
    
    def load_mutated_hyperparameter_dict(self, hyperparameter_dict):
        """Set the hyperparameters for the training process based on a mutated version of a previously saved parameter set"""
        self.load_hyperparameter_dict({
            'actor_lr': mutate_param(hyperparameter_dict['actor_lr']),
            'critic_lr': mutate_param(hyperparameter_dict['critic_lr']),
            'gamma': 1 - mutate_param(1 - hyperparameter_dict['gamma']),
            'tau': mutate_param(hyperparameter_dict['tau']),
        })
    
    def full_save_dict(self):
        """Create a combined dictionary of weights, hyperparameters, and past scores for the agent"""
        return {
            'agent_name': self.name,
            'state': self.state_dict(),
            'hyperparameters': self.hyperparameter_dict(),
            'history': self.history
        }
    
    def load_full_save_dict(self, full_save_dict):
        """Set agent state from a snapshot returned earlier by full_save_dict()"""
        self.load_state_dict(full_save_dict['state'])
        self.load_hyperparameter_dict(full_save_dict['hyperparameters'])
        self.name = full_save_dict['agent_name']
        self.history = full_save_dict['history']
        self.episode_return = 0
        self.episode = len(self.history) + 1

    def get_running_mean_return(self, n):
        return np.mean(self.history.tail(n)['return'])


def ddpg(agent, env, brain_name,
         max_episode=1000, max_t=1100,
         checkpoint_episodes=100):
    """Carry out learning based on the DDPG algorithm."""
    for i_episode in range(len(agent.history) + 1, max_episode + 1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        for t in range(0, max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if i_episode % checkpoint_episodes == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, agent.get_running_mean_return(checkpoint_episodes)))
            torch.save(agent.full_save_dict(),"{}_episode_{}.pth".format(agent.name, i_episode))


def ddpg_collab(agents, env, brain_name,
         max_episode=1000, max_t=1100,
         checkpoint_episodes=100):
    """Carry out learning based on the DDPG algorithm, for two agents collaborating."""
    for i_episode in range(len(agents[0].history) + 1, max_episode + 1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations
        for t in range(0, max_t):
            actions = np.array([agents[0].act(states[0]), agents[1].act(states[1])])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            done = np.any(env_info.local_done)
            for i in range(0, 2):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], done)
            states = next_states
            if done:
                break
        if i_episode % checkpoint_episodes == 0:
            for i in range(0, 2):
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, agents[i].get_running_mean_return(checkpoint_episodes)))
                torch.save(agents[i].full_save_dict(),"{}_episode_{}.pth".format(agents[i].name, i_episode))
