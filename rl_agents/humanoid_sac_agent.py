import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class HumanoidSACAgent:
    """
    Soft Actor-Critic (SAC) agent optimized for humanoid robot control
    """
    
    def __init__(self, state_dim, action_dim, max_action, 
                 lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, buffer_size=int(1e6), batch_size=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Initialize networks
        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(self.device)
        self.critic = DoubleQCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DoubleQCritic(state_dim, action_dim).to(self.device)
        
        # Initialize target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Temperature parameter (alpha) for entropy regularization
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Training state
        self.total_it = 0
        
    def select_action(self, state, evaluate=False):
        """
        Select action using the policy
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.cpu().data.numpy().flatten()
    
    def train(self, batch_size=256):
        """
        Train the agent on a batch of experiences
        """
        if len(self.replay_buffer) < batch_size:
            return  # Not enough samples yet
        
        # Sample batch from replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        not_done = torch.FloatTensor(not_done).to(self.device).unsqueeze(1)
        
        # Compute critic loss
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            next_q1, next_q2 = self.critic_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward + not_done * self.gamma * next_q
        
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss
        pi, log_pi, _ = self.actor.sample(state)
        q1, q2 = self.critic(state, pi)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_pi - q).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Optimize alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.total_it += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def save(self, filename):
        """
        Save the model to a file
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_it': self.total_it
        }, filename)
    
    def load(self, filename):
        """
        Load the model from a file
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.total_it = checkpoint['total_it']

class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action spaces
    """
    def __init__(self, num_inputs, num_actions, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, num_actions)
        self.log_std_linear = nn.Linear(256, num_actions)

        self.apply(self._init_weights)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean

class DoubleQCritic(nn.Module):
    """
    Twin Q networks for SAC
    """
    def __init__(self, num_inputs, num_actions):
        super(DoubleQCritic, self).__init__()
        
        # Q1
        self.linear1 = nn.Linear(num_inputs + num_actions, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        
        # Q2
        self.linear4 = nn.Linear(num_inputs + num_actions, 256)
        self.linear5 = nn.Linear(256, 256)
        self.linear6 = nn.Linear(256, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x1, x2
    
    def Q1(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        
        return x1

class ReplayBuffer:
    """
    Experience replay buffer for SAC
    """
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.not_done = None
    
    def add(self, state, action, next_state, reward, done):
        dim_state = len(state)
        dim_action = len(action)
        
        if self.state is None:
            self.state = np.zeros((self.max_size, dim_state))
            self.action = np.zeros((self.max_size, dim_action))
            self.next_state = np.zeros((self.max_size, dim_state))
            self.reward = np.zeros((self.max_size, 1))
            self.not_done = np.zeros((self.max_size, 1))
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )
    
    def __len__(self):
        return self.size