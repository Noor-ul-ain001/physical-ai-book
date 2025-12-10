#!/usr/bin/env python3
"""
Training script for Humanoid Reinforcement Learning
This script trains a humanoid robot to walk and balance using SAC
"""

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from isaac_env.isaac_sim_humanoid_env import IsaacSimHumanoidEnvironment
from rl_agents.humanoid_sac_agent import HumanoidSACAgent
import argparse
from datetime import datetime

def train_humanoid_rl():
    """
    Main training function for humanoid robot
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train humanoid robot with RL')
    parser.add_argument('--max_episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--episode_length', type=int, default=1000, help='Steps per episode')
    parser.add_argument('--save_freq', type=int, default=100, help='Model save frequency')
    parser.add_argument('--render', action='store_true', help='Render training')
    parser.add_argument('--model_path', type=str, default='', help='Path to load pre-trained model')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='Directory for logs')
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize the Isaac Sim environment
    env = IsaacSimHumanoidEnvironment(headless=not args.render)
    
    # Define state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize the SAC agent
    agent = HumanoidSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )
    
    # Load pre-trained model if specified
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        agent.load(args.model_path)
    
    # Training variables
    episode_rewards = []
    episode_lengths = []
    avg_episode_rewards = []
    avg_episode_lengths = []
    
    total_timesteps = 0
    start_time = time.time()
    
    print(f"Starting training with {state_dim}-dimensional state and {action_dim}-dimensional action space...")
    print(f"Training for {args.max_episodes} episodes")
    
    # Training loop
    for episode in range(args.max_episodes):
        # Reset environment
        obs, _ = env.reset()
        
        episode_reward = 0
        episode_timesteps = 0
        done = False
        
        # Episode loop
        while not done and episode_timesteps < args.episode_length:
            # Select action from agent
            action = agent.select_action(obs)

            # Execute action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.add(obs, action, next_obs, reward, done)

            # Train agent if buffer has enough samples
            if len(agent.replay_buffer) >= 1000:  # Minimum buffer size for training
                agent.train(batch_size=256)

            # Update observation and counters
            obs = next_obs
            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_timesteps)
        
        # Compute rolling averages (last 100 episodes)
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_episode_rewards.append(avg_reward)
            avg_episode_lengths.append(avg_length)
        else:
            avg_episode_rewards.append(np.mean(episode_rewards))
            avg_episode_lengths.append(np.mean(episode_lengths))
        
        # Print progress
        if episode % 10 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_length_100 = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)

            elapsed_time = time.time() - start_time
            print(f"Episode {episode} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward 100: {avg_reward_100:.2f} | "
                  f"Length: {episode_timesteps} | "
                  f"Avg Length 100: {avg_length_100:.2f} | "
                  f"Steps: {total_timesteps} | "
                  f"Time: {elapsed_time:.2f}s")
        
        # Save model periodically
        if episode % args.save_freq == 0:
            model_path = os.path.join(args.log_dir, f"humanoid_model_{episode}.pth")
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    # Final model save
    final_model_path = os.path.join(args.log_dir, f"humanoid_final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    agent.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, avg_episode_rewards, args.log_dir)
    
    # Save training history
    training_history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avg_episode_rewards': avg_episode_rewards,
        'avg_episode_lengths': avg_episode_lengths,
        'total_timesteps': total_timesteps,
        'training_time_seconds': time.time() - start_time
    }
    
    history_path = os.path.join(args.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    
    print(f"Training completed. History saved to {history_path}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")
    
    # Close environment
    env.close()

def plot_training_curves(episode_rewards, avg_episode_rewards, log_dir):
    """
    Plot training curves for visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot episode rewards
    ax.plot(episode_rewards, label='Episode Reward', alpha=0.3)
    ax.plot(avg_episode_rewards, label='Rolling Avg (100 episodes)', linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Humanoid Robot Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'training_curve.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training curve saved to {plot_path}")

def evaluate_policy(agent, env, num_episodes=10):
    """
    Evaluate the trained policy
    """
    total_reward = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, evaluate=True)  # Use deterministic policy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    avg_reward = total_reward / num_episodes
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    
    return avg_reward

if __name__ == "__main__":
    train_humanoid_rl()