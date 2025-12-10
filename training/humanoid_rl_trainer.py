# Training pipeline for humanoid robot RL
# training/humanoid_rl_trainer.py

import torch
import torch.nn as nn
import numpy as np
import random
import os
import pickle
from collections import deque
import json

class HumanoidSACAgent:
    """
    Soft Actor-Critic agent optimized for humanoid robot control
    """
    def __init__(self, state_dim, action_dim, action_space, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.critic = DoubleQCritic(state_dim, action_dim, config['hidden_dim']).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])

        self.actor = GaussianPolicy(state_dim, action_dim, config['hidden_dim'], action_space).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['alpha_lr'])

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).item() if config['automatic_entropy_tuning'] else -action_dim
        self.alpha = config['alpha']

        self.gamma = config['gamma']
        self.tau = config['tau']
        self.target_update_interval = config['target_update_interval']
        self.learn_iteration = 0

        self.learn_start_step = config.get('learn_start_step', 1000)
        self.batch_size = config.get('batch_size', 256)

    def select_action(self, state, evaluate=False):
        """
        Select action using the actor network
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        """
        Update network parameters using SAC algorithm
        """
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate overestimation bias

        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st) + γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st) + γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st~D,εt~N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For Tensorboard logging
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For Tensorboard logging


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

class HumanoidReplayBuffer:
    """
    Custom replay buffer optimized for humanoid robot training
    """
    def __init__(self, max_size=int(1e6), obs_shape=None, act_shape=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Pre-allocate numpy arrays for efficiency
        self.state = np.zeros((max_size, *obs_shape))
        self.action = np.zeros((max_size, *act_shape))
        self.next_state = np.zeros((max_size, *obs_shape))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        """
        Add transition to replay buffer
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample batch from replay buffer
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )

class HumanoidTrainer:
    """
    Main training loop for humanoid robot RL
    """
    def __init__(self, config_path):
        """
        Initialize the humanoid trainer with configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set up environment (using Isaac Sim)
        self.setup_environment()
        
        # Initialize agent
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim'] 
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        self.agent = HumanoidSACAgent(
            self.state_dim, 
            self.action_dim, 
            self.action_space,
            self.config['rl_params']
        )
        
        # Initialize replay buffer
        self.replay_buffer = HumanoidReplayBuffer(
            max_size=self.config['buffer_size'],
            obs_shape=(self.state_dim,),
            act_shape=(self.action_dim,)
        )
        
        # Logging
        self.model_save_path = self.config['model_save_path']
        self.log_path = self.config['log_path']
        
        # Training parameters
        self.max_timesteps = self.config['max_timesteps']
        self.eval_freq = self.config['eval_freq']
        self.save_freq = self.config['save_freq']
        
        # Curriculum learning parameters
        self.curriculum = self.config.get('curriculum', {})
        self.current_curriculum_stage = 0
        
        print("Humanoid RL Trainer initialized successfully!")
    
    def setup_environment(self):
        """
        Setup the Isaac Sim environment for training
        """
        # Initialize Isaac Sim
        from omni.isaac.kit import SimulationApp
        
        self.sim_app = SimulationApp({"headless": False})  # Set to True for headless training
        
        # Import Isaac Sim components
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        
        # Create world instance
        self.world = World(stage_units_in_meters=1.0)
        
        # Configure physics parameters
        self.world.scene.add_default_ground_plane()
        self.world.reset()
    
    def train(self):
        """
        Main training loop
        """
        total_timesteps = 0
        episode_num = 0
        episode_reward = 0
        episode_timesteps = 0
        
        # Initialize environment
        obs = self.world.reset()
        
        print(f"Starting training for {self.max_timesteps} timesteps...")
        
        while total_timesteps < self.max_timesteps:
            # Select action with noise for exploration
            if total_timesteps < self.config['start_timesteps']:
                action = self.action_space.sample()
            else:
                action = self.agent.select_action(np.array(obs))
                # Add exploration noise
                noise = np.random.normal(0, self.config['exploration_noise'], size=self.action_dim)
                action = (action + noise).clip(-1, 1)
            
            # Perform action in environment
            new_obs, reward, done, info = self.world.step(action)
            
            # Store data in replay buffer
            done_bool = float(done) if episode_timesteps < self.world.max_episode_steps else 0
            self.replay_buffer.add(obs, action, new_obs, reward, done_bool)
            
            obs = new_obs
            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
            
            # Train agent
            if total_timesteps >= self.config['start_timesteps']:
                for _ in range(self.config['train_freq']):
                    self.agent.update_parameters(
                        self.replay_buffer, 
                        self.config['batch_size'], 
                        total_timesteps
                    )
            
            # Check for episode termination
            if done or episode_timesteps >= self.config['episode_length']:
                # Print episode info
                print(f"Total T: {total_timesteps} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                
                # Reset environment
                obs, done = self.world.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
                
                # Evaluate agent periodically
                if total_timesteps % self.eval_freq == 0:
                    eval_reward = self.evaluate_policy(total_timesteps)
                    print(f"Evaluation Reward: {eval_reward:.3f}")
                
                # Save model periodically
                if total_timesteps % self.save_freq == 0:
                    self.save_model(total_timesteps)
        
        # Final evaluation and model save
        final_reward = self.evaluate_policy(total_timesteps)
        self.save_model("final")
        
        print(f"Training completed! Final evaluation reward: {final_reward:.3f}")
        
        # Close Isaac Sim
        self.sim_app.close()
    
    def evaluate_policy(self, timesteps, eval_episodes=10):
        """
        Evaluate the current policy
        """
        avg_reward = 0.
        self.world.reset()
        
        for _ in range(eval_episodes):
            obs = self.world.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.agent.select_action(np.array(obs), evaluate=True)
                obs, reward, done, _ = self.world.step(action)
                episode_reward += reward
            
            avg_reward += episode_reward
        
        avg_reward /= eval_episodes
        
        # Log evaluation
        print(f"Evaluation at {timesteps} timesteps: Average Reward: {avg_reward:.3f}")
        return avg_reward
    
    def save_model(self, timesteps):
        """
        Save the trained model
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Save model checkpoints
        torch.save(
            self.agent.critic.state_dict(), 
            f"{self.model_save_path}/critic_{timesteps}.pth"
        )
        torch.save(
            self.agent.critic_optimizer.state_dict(), 
            f"{self.model_save_path}/critic_optimizer_{timesteps}.pth"
        )
        
        torch.save(
            self.agent.actor.state_dict(), 
            f"{self.model_save_path}/actor_{timesteps}.pth"
        )
        torch.save(
            self.agent.actor_optimizer.state_dict(), 
            f"{self.model_save_path}/actor_optimizer_{timesteps}.pth"
        )
        
        print(f"Model saved at {self.model_save_path} with timesteps {timesteps}")

    def load_model(self, model_path):
        """
        Load a pre-trained model
        """
        self.agent.critic.load_state_dict(torch.load(f"{model_path}/critic.pth"))
        self.agent.critic_optimizer.load_state_dict(torch.load(f"{model_path}/critic_optimizer.pth"))
        self.agent.actor.load_state_dict(torch.load(f"{model_path}/actor.pth"))
        self.agent.actor_optimizer.load_state_dict(torch.load(f"{model_path}/actor_optimizer.pth"))
        
        print(f"Model loaded from {model_path}")

def main():
    """
    Main function to run the humanoid RL training
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Humanoid RL Training')
    parser.add_argument('--config', type=str, default='config/humanoid_config.json', 
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train', 
                       help='Training mode: train or eval')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to pre-trained model (for evaluation)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = HumanoidTrainer(args.config)
    
    if args.mode == 'train':
        # Start training
        trainer.train()
    else:
        # Load model and evaluate
        if args.model_path:
            trainer.load_model(args.model_path)
            avg_reward = trainer.evaluate_policy(0, eval_episodes=20)
            print(f"Evaluation completed. Average reward: {avg_reward:.3f}")
        else:
            print("Model path required for evaluation mode")

if __name__ == "__main__":
    main()