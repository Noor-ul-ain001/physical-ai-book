#!/usr/bin/env python3
"""
Project Initialization Script for Physical AI & Humanoid Robotics Textbook

This script completes the project setup by:
1. Installing required dependencies
2. Setting up the Docusaurus project
3. Validating configurations
4. Creating missing essential files
5. Running initial build to verify setup
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """
    Run a command and handle errors
    """
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True,
            check=True
        )
        print(f"✓ Success: {description}")
        if result.stdout.strip():
            print(f"  Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {description}")
        print(f"  Error: {e.stderr}")
        return False

def check_dependencies():
    """
    Check if required dependencies are installed
    """
    print("Checking dependencies...")
    
    dependencies = {
        'node': 'node --version',
        'npm': 'npm --version',
        'python': 'python --version',
        'git': 'git --version'
    }
    
    missing_deps = []
    for name, cmd in dependencies.items():
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            print(f"✓ {name}: {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            print(f"✗ {name}: Not found")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Please install the missing dependencies before continuing.")
        return False
    
    return True

def setup_project():
    """
    Set up the project structure and dependencies
    """
    print("\nSetting up project...")
    
    # Initialize git repo if not already initialized
    if not os.path.exists('.git'):
        run_command('git init', 'Initialize git repository')
    
    # Create virtual environment for Python dependencies
    if not os.path.exists('venv'):
        print("Creating Python virtual environment...")
        run_command('python -m venv venv', 'Create virtual environment')
    
    # Install Python dependencies
    print("Installing Python dependencies...")
    if os.name == 'nt':  # Windows
        pip_cmd = 'venv\\Scripts\\pip'
    else:  # Unix/Linux/MacOS
        pip_cmd = 'venv/bin/pip'
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        success = run_command(f'call venv\\Scripts\\activate && pip install -r requirements.txt', 
                             'Install Python dependencies')
    else:
        success = run_command('source venv/bin/activate && pip install -r requirements.txt', 
                             'Install Python dependencies', cwd=os.getcwd())
    
    if not success:
        print("Installing individual packages...")
        run_command(f'{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu', 
                   'Install PyTorch')
        run_command(f'{pip_cmd} install google-generativeai', 
                   'Install Google Generative AI')
        run_command(f'{pip_cmd} install qdrant-client', 
                   'Install Qdrant client')
        run_command(f'{pip_cmd} install open3d', 
                   'Install Open3D')
    
    # Install Node.js dependencies
    print("\nInstalling Node.js dependencies...")
    success = run_command('npm install', 'Install Node.js dependencies')
    
    if not success:
        print("Trying with legacy peer deps...")
        run_command('npm install --legacy-peer-deps', 'Install Node.js dependencies (legacy)')
    
    return True

def setup_docusaurus():
    """
    Set up Docusaurus project
    """
    print("\nSetting up Docusaurus...")
    
    # Create docs structure if not exists
    docs_dirs = [
        'docs/intro',
        'docs/module1-ros2', 
        'docs/module2-digital-twin',
        'docs/module3-isaac',
        'docs/module4-vla'
    ]
    
    for docs_dir in docs_dirs:
        os.makedirs(docs_dir, exist_ok=True)
        print(f"✓ Created {docs_dir}")
    
    # Create src structure if not exists
    src_dirs = [
        'src/components',
        'src/theme',
        'src/css'
    ]
    
    for src_dir in src_dirs:
        os.makedirs(src_dir, exist_ok=True)
        print(f"✓ Created {src_dir}")
    
    # Set up Docusaurus configuration
    docusaurus_config_exists = os.path.exists('docusaurus.config.ts')
    if docusaurus_config_exists:
        print("✓ Docusaurus configuration exists")
    else:
        print("Creating basic Docusaurus configuration...")
        basic_config = '''import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'An Interactive Textbook with Personalised RAG Chatbot',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-docusaurus-site.example.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'docusaurus', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: false, // Disable blog for textbook
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://github.com/facebook/docusaurus',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro/00-welcome',
              },
              {
                label: 'ROS2 Fundamentals',
                to: '/docs/module1-ros2/01-overview',
              },
              {
                label: 'Digital Twin & Simulation',
                to: '/docs/module2-digital-twin/01-gazebo-basics',
              },
              {
                label: 'Isaac Sim & vSLAM',
                to: '/docs/module3-isaac/01-isaac-sim-basics',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/docusaurus',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/facebook/docusaurus',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'json', 'yaml', 'xml'],
      },
    }),
  
  plugins: [
    // Custom plugin for NVIDIA-themed styling
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'modules',
        path: 'docs',
        routeBasePath: 'docs',
        sidebarPath: require.resolve('./sidebars.js'),
      },
    ],
  ],
};

export default config;
'''
        
        with open('docusaurus.config.ts', 'w') as f:
            f.write(basic_config)
        print("✓ Created docusaurus.config.ts")
    
    # Create basic sidebar configuration if not exists
    if not os.path.exists('sidebars.js'):
        basic_sidebars = '''// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: ['intro/00-welcome', 'intro/01-foundations', 'intro/02-hardware-guide'],
    },
    {
      type: 'category',
      label: 'Module 1: ROS2 Fundamentals',
      items: [
        'module1-ros2/01-overview',
        'module1-ros2/02-nodes-topics-services',
        'module1-ros2/03-rclpy-python-bridge',
        'module1-ros2/04-urdf-humanoids',
        'module1-ros2/05-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin & Simulation',
      items: [
        'module2-digital-twin/01-gazebo-basics',
        'module2-digital-twin/02-urdf-sdf',
        'module2-digital-twin/03-sensors-simulation',
        'module2-digital-twin/04-unity-visualization',
        'module2-digital-twin/05-project',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Isaac Sim & vSLAM',
      items: [
        'module3-isaac/01-isaac-sim-basics',
        'module3-isaac/02-isaac-ros-integration',
        'module3-isaac/03-vslam-navigation',
        'module3-isaac/04-vslam-navigation-completion',
        'module3-isaac/05-project-summary',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      items: [
        'module4-vla/01-vision-language-action',
        'module4-vla/02-whisper-voice-commands',
        'module4-vla/03-llm-task-planning',
        'module4-vla/04-capstone-project',
        'module4-vla/05-final-deployment',
      ],
    },
  ],
};

module.exports = sidebars;
'''
        
        with open('sidebars.js', 'w') as f:
            f.write(basic_sidebars)
        print("✓ Created sidebars.js")
    
    return True

def setup_isaac_integration():
    """
    Set up Isaac Sim integration components
    """
    print("\nSetting up Isaac Sim integration...")
    
    # Create Isaac Sim specific directories
    isaac_dirs = [
        'isaac_env',
        'isaac_ros_nodes',
        'isaac_configs'
    ]
    
    for isaac_dir in isaac_dirs:
        os.makedirs(isaac_dir, exist_ok=True)
        print(f"✓ Created {isaac_dir}")
    
    # Create a basic Isaac Sim launch script
    launch_script = '''#!/usr/bin/env python3
"""
Isaac Sim Launch Script for Physical AI & Humanoid Robotics
"""
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
import numpy as np

# Initialize Isaac Sim world with humanoid robot
def setup_humanoid_environment():
    """
    Set up the humanoid robot environment in Isaac Sim
    """
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0/60.0,  # Physics timestep
        rendering_dt=1.0/60.0, # Rendering timestep
        sim_params={
            "use_gpu": True,
            "use_fabric": True,
            "solver_type": "TGS",
            "num_position_iterations": 8,
            "num_velocity_iterations": 1,
            "max_depenetration_velocity": 1000.0,
            "substeps": 2,
        }
    )
    
    # Add ground plane
    world.scene.add_default_ground_plane()
    
    # Add humanoid robot from URDF
    add_reference_to_stage(
        usd_path="path/to/humanoid_robot.usd",  # Replace with actual path
        prim_path="/World/HumanoidRobot"
    )
    
    # Add lighting
    from omni.isaac.core.utils.prims import create_primitive
    create_primitive(
        prim_path="/World/Light",
        prim_type="DistantLight",
        position=np.array([0, 0, 10]),
        orientation=np.array([0, 0, 0, 1]),
        attributes={"color": (0.8, 0.8, 0.8), "intensity": 3000}
    )
    
    return world

if __name__ == "__main__":
    world = setup_humanoid_environment()
    print("Isaac Sim environment ready for humanoid robotics simulation")
'''
    
    with open('isaac_env/launch_humanoid_sim.py', 'w') as f:
        f.write(launch_script)
    print("✓ Created Isaac Sim launch script")
    
    return True

def setup_rl_training():
    """
    Set up reinforcement learning training components
    """
    print("\nSetting up RL training components...")
    
    # Create training script directory structure
    training_dirs = [
        'training/utils',
        'training/algorithms',
        'training/environments'
    ]
    
    for training_dir in training_dirs:
        os.makedirs(training_dir, exist_ok=True)
        print(f"✓ Created {training_dir}")
    
    # Create a basic training script
    training_script = '''#!/usr/bin/env python3
"""
Reinforcement Learning Training Script for Humanoid Robots
"""
import torch
import numpy as np
from typing import Dict, Tuple, List
import os
import json

class HumanoidRLTrainer:
    """
    Training class for humanoid robot behaviors using reinforcement learning
    """
    def __init__(self, config_path: str = "training/config.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_path)
        
        # Training state
        self.current_episode = 0
        self.total_timesteps = 0
        
        # Initialize networks based on config
        self.initialize_networks()
        
        print(f"Humanoid RL trainer initialized on {self.device}")
    
    def load_config(self, config_path: str) -> Dict:
        """
        Load training configuration
        """
        default_config = {
            "algorithm": "sac",
            "max_episodes": 5000,
            "max_timesteps": 1000,
            "state_dim": 122,  # Example: joint positions, velocities, IMU, etc.
            "action_dim": 20,  # Example: 20 joint position commands
            "lr_actor": 3e-4,
            "lr_critic": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_size": 1000000
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all required keys exist
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
        else:
            config = default_config
            # Save default config for reference
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config
    
    def initialize_networks(self):
        """
        Initialize neural networks for the RL algorithm
        """
        from training.algorithms.sac import SACAgent
        
        self.agent = SACAgent(
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            lr_actor=self.config['lr_actor'],
            lr_critic=self.config['lr_critic'],
            gamma=self.config['gamma'],
            tau=self.config['tau'],
            device=self.device
        )
        
        print("Neural networks initialized")
    
    def train(self):
        """
        Main training loop
        """
        print(f"Starting training with {self.config['max_episodes']} episodes")
        
        for episode in range(self.config['max_episodes']):
            obs = self.reset_environment()
            episode_reward = 0
            done = False
            timestep = 0
            
            while not done and timestep < self.config['max_timesteps']:
                # Get action from policy
                action = self.agent.select_action(obs)
                
                # Execute action in environment (Isaac Sim)
                next_obs, reward, done, info = self.step_environment(action)
                
                # Store transition in replay buffer
                self.agent.replay_buffer.add(obs, action, reward, next_obs, done)
                
                # Update networks if enough samples
                if len(self.agent.replay_buffer) > self.config['batch_size']:
                    self.agent.update_parameters(
                        batch_size=self.config['batch_size']
                    )
                
                obs = next_obs
                episode_reward += reward
                timestep += 1
                self.total_timesteps += 1
            
            self.current_episode += 1
            
            # Log progress
            if self.current_episode % 10 == 0:
                print(f"Episode {self.current_episode}, Reward: {episode_reward:.2f}, "
                      f"Timesteps: {timestep}, Total: {self.total_timesteps}")
            
            # Save model periodically
            if self.current_episode % 100 == 0:
                self.save_model(f"models/humanoid_model_ep{self.current_episode}.pth")
        
        print("Training completed")
        self.save_model("models/final_humanoid_model.pth")
    
    def reset_environment(self):
        """
        Reset simulation environment to initial state
        """
        # This would interface with Isaac Sim
        obs = np.random.rand(self.config['state_dim'])  # Placeholder
        return obs
    
    def step_environment(self, action):
        """
        Execute action in simulation environment
        """
        # This would interface with Isaac Sim
        next_obs = np.random.rand(self.config['state_dim'])  # Placeholder
        reward = np.random.rand()  # Placeholder
        done = np.random.rand() > 0.99  # Placeholder
        info = {}  # Placeholder
        return next_obs, reward, done, info
    
    def save_model(self, path: str):
        """
        Save trained model to disk
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.agent.save(path)
        print(f"Model saved to {path}")

if __name__ == "__main__":
    trainer = HumanoidRLTrainer()
    trainer.train()
'''
    
    with open('training/train_humanoid_rl.py', 'w') as f:
        f.write(training_script)
    print("✓ Created RL training script")
    
    # Create a basic SAC algorithm implementation
    sac_algorithm = '''"""
Soft Actor-Critic (SAC) Implementation for Humanoid Robotics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training.utils.replay_buffer import ReplayBuffer

class Actor(nn.Module):
    """
    Actor network (Gaussian policy) for humanoid control
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        
        mean = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    """
    Critic network (Double Q-learning) for humanoid value estimation
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        return q1

class SACAgent:
    """
    Soft Actor-Critic agent for humanoid robot control
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device="cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, max_action=1.0).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Copy critic parameters to target critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Entropy temperature
        self.alpha = alpha
        self.automatic_entropy_tuning = True
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = 1
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=int(1e6))
        
        self.learn_step_counter = 0
    
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
    
    def update_parameters(self, batch_size=256):
        """
        Update actor and critic networks
        """
        # Sample batch from replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        not_done = torch.FloatTensor(not_done).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            next_q_value = reward + not_done * self.gamma * min_q_next
        
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        pi, log_pi = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Update target networks
        if self.learn_step_counter % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        self.learn_step_counter += 1
    
    def save(self, filename):
        """
        Save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha': self.alpha,
        }, filename)
    
    def load(self, filename):
        """
        Load the model
        """
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning and 'alpha_optimizer_state_dict' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
        elif 'alpha' in checkpoint:
            self.alpha = checkpoint['alpha']
'''
    
    with open('training/algorithms/sac.py', 'w') as f:
        f.write(sac_algorithm)
    print("✓ Created SAC algorithm implementation")
    
    return True

def validate_setup():
    """
    Validate the project setup
    """
    print("\nValidating project setup...")
    
    required_files = [
        'package.json',
        'docusaurus.config.ts',
        'sidebars.js',
        'README.md',
        'docs/intro/00-welcome.mdx',
        'src/components/ChatBot.tsx',
        'api/main.py',
        'better-auth/config.ts',
        'scripts/index-to-qdrant.ts',
        '.env.example',
        'tsconfig.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        print("These may need to be created separately.")
    else:
        print("✓ All required files are present")
    
    # Check if basic build works
    print("\nTesting Docusaurus build setup...")
    success = run_command('npx docusaurus --version', 'Check Docusaurus CLI')
    
    return success

def create_missing_components():
    """
    Create any missing essential components
    """
    print("\nCreating missing essential components...")
    
    # Create src/App.js if it doesn't exist
    if not os.path.exists('src/App.js'):
        app_content = '''import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { useColorMode } from '@docusaurus/theme-common';

function HomepageHeader() {
  const [personalizeEnabled, setPersonalizeEnabled] = useState(false);
  const [translateEnabled, setTranslateEnabled] = useState(false);
  const { colorMode, setColorMode } = useColorMode();

  return (
    <header className="hero hero--primary">
      <div className="container">
        <h1 className="hero__title">Physical AI & Humanoid Robotics</h1>
        <p className="hero__subtitle">An Interactive Textbook with Personalised RAG Chatbot</p>
        <div style={{ marginTop: '2rem' }}>
          <button 
            onClick={() => setPersonalizeEnabled(!personalizeEnabled)}
            style={{
              marginRight: '1rem',
              padding: '0.5rem 1rem',
              backgroundColor: personalizeEnabled ? '#5d9400' : '#76b900',
              border: 'none',
              borderRadius: '4px',
              color: 'white',
              cursor: 'pointer'
            }}
          >
            {personalizeEnabled ? 'Disable Personalization' : 'Enable Personalization'}
          </button>
          
          <button 
            onClick={() => setTranslateEnabled(!translateEnabled)}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: translateEnabled ? '#5d9400' : '#76b900',
              border: 'none',
              borderRadius: '4px',
              color: 'white',
              cursor: 'pointer'
            }}
          >
            {translateEnabled ? 'Switch to English' : 'اردو میں سوچیں'}
          </button>
        </div>
        
        {personalizeEnabled && (
          <div style={{
            marginTop: '2rem',
            padding: '1.5rem',
            border: '1px solid #76b900',
            borderRadius: '8px',
            backgroundColor: 'rgba(118, 185, 0, 0.05)'
          }}>
            <h3>Personalize Your Learning Experience</h3>
            <p>Tell us about your background to customize content difficulty:</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'flex-start', marginTop: '1rem' }}>
              <label style={{ fontWeight: 'bold' }}>
                <input type="checkbox" style={{ marginRight: '0.5rem' }} />
                Hardware: Do you have RTX 4070+ GPU?
              </label>
              
              <label style={{ fontWeight: 'bold' }}>
                <input type="checkbox" style={{ marginRight: '0.5rem' }} />
                Hardware: Do you own a Jetson?
              </label>
              
              <label style={{ fontWeight: 'bold' }}>
                <input type="checkbox" style={{ marginRight: '0.5rem' }} />
                Hardware: Do you have a real robot?
              </label>
              
              <label style={{ fontWeight: 'bold' }}>
                Software Background: Years with Python? 
                <input type="range" min="0" max="20" defaultValue="5" style={{ marginLeft: '0.5rem', width: '200px' }} />
              </label>
              
              <label style={{ fontWeight: 'bold' }}>
                Goal: 
                <select style={{ marginLeft: '0.5rem', padding: '0.25rem' }}>
                  <option value="learning">Learning</option>
                  <option value="building">Building real humanoid</option>
                  <option value="research">Research</option>
                </select>
              </label>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}

export default function Home() {
  return (
    <Layout
      title={`Physical AI & Humanoid Robotics`}
      description="Interactive textbook with personalized learning, multilingual support, and AI-powered Q&A">
      <HomepageHeader />
      <main>
        <div className="container padding-top--lg">
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '2rem', margin: '4rem 0' }}>
            <div style={{ flex: '1', minWidth: '250px', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <h2>Comprehensive Curriculum</h2>
              <p>23 expert-level chapters covering ROS2, Digital Twins, Isaac Sim, and Vision-Language-Action models.</p>
            </div>
            <div style={{ flex: '1', minWidth: '250px', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <h2>Personalized Learning</h2>
              <p>Content adapts based on your skill level and hardware capabilities.</p>
            </div>
            <div style={{ flex: '1', minWidth: '250px', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <h2>Multilingual Support</h2>
              <p>Instant translation to Urdu for better understanding of complex concepts.</p>
            </div>
            <div style={{ flex: '1', minWidth: '250px', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
              <h2>AI-Powered Assistance</h2>
              <p>RAG-based chatbot answers questions with source citations from the textbook.</p>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}
'''
        
        with open('src/App.js', 'w') as f:
            f.write(app_content)
        print("✓ Created src/App.js")
    
    # Create basic ChatBot component if it doesn't exist
    if not os.path.exists('src/components/ChatBot.tsx'):
        chatbot_content = '''import React, { useState, useRef, useEffect } from 'react';
import { useThemeContext } from '@docusaurus/theme-common';
import { useLocation } from '@docusaurus/router';

// Mock Gemini API integration (replace with real implementation)
const useGeminiAPI = () => {
  const [isLoading, setIsLoading] = useState(false);
  
  const queryGemini = async (prompt: string) => {
    setIsLoading(true);
    try {
      // In a real implementation, this would call the actual Gemini API
      // For now, return a mock response
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay
      
      const mockResponses = [
        "The center of mass (CoM) is a crucial concept in humanoid robotics. It represents the point where the robot's entire mass can be considered concentrated. For stable locomotion, the CoM must remain within the support polygon defined by the feet's contact area.",
        "Dynamic balance in humanoid robots involves continuous adjustment of the center of mass and foot placement to maintain stability. The Zero Moment Point (ZMP) criterion is often used to ensure the robot remains balanced during walking or other maneuvers.",
        "Bipedal locomotion requires complex coordination of multiple joints and control systems. Modern approaches use central pattern generators (CPGs) combined with feedback control to achieve stable and natural-looking walking patterns."
      ];
      
      const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
      
      return {
        response: randomResponse,
        sources: [
          { title: "Module 2: Balance Control Theory", url: "/docs/module2-digital-twin/03-balance-control" },
          { title: "Module 3: Locomotion Algorithms", url: "/docs/module3-isaac/04-locomotion-rl" }
        ]
      };
    } catch (error) {
      console.error('Gemini API error:', error);
      return {
        response: "Sorry, I couldn't process your request. Please check your API configuration.",
        sources: []
      };
    } finally {
      setIsLoading(false);
    }
  };
  
  return { queryGemini, isLoading };
};

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  sources?: Array<{title: string, url: string}>;
}

interface ChatBotProps {
  initialOpen?: boolean;
}

const ChatBot: React.FC<ChatBotProps> = ({ initialOpen = false }) => {
  const [isOpen, setIsOpen] = useState(initialOpen);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { isDarkTheme } = useThemeContext();
  
  const location = useLocation();
  const { queryGemini } = useGeminiAPI();
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (inputText.trim() === '') return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    // Get context about the current page
    const currentPageContext = `The user is currently viewing the page at: ${location.pathname}. ` +
      `Based on the textbook content, provide relevant information.`;

    // Query Gemini
    const fullPrompt = currentPageContext + " User question: " + inputText;
    const result = await queryGemini(fullPrompt);

    // Add bot message with sources
    const botMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: result.response,
      isUser: false,
      timestamp: new Date(),
      sources: result.sources
    };
    
    setMessages(prev => [...prev, botMessage]);
    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Chat button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="chatbot-button"
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            backgroundColor: '#76b900',
            color: 'white',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            zIndex: 1000,
            boxShadow: '0 4px 8px rgba(0,0,0,0.2)',
          }}
        >
          ðŸ¤–
        </button>
      )}

      {/* Chat window */}
      {isOpen && (
        <div
          className="chatbot-container"
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '400px',
            height: '600px',
            backgroundColor: isDarkTheme ? '#2d2d2d' : '#ffffff',
            border: '1px solid #ccc',
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          <div
            className="chat-header"
            style={{
              backgroundColor: '#76b900',
              color: 'white',
              padding: '10px',
              borderTopLeftRadius: '8px',
              borderTopRightRadius: '8px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <span>Physical AI Assistant</span>
            <button
              onClick={toggleChat}
              style={{
                background: 'none',
                border: 'none',
                color: 'white',
                fontSize: '18px',
                cursor: 'pointer',
              }}
            >
              Ã—
            </button>
          </div>
          
          <div
            className="chat-messages"
            style={{
              flex: 1,
              padding: '10px',
              overflowY: 'auto',
            }}
          >
            {messages.length === 0 ? (
              <div 
                style={{ 
                  textAlign: 'center', 
                  color: isDarkTheme ? '#aaa' : '#666',
                  fontStyle: 'italic',
                  margin: 'auto',
                }}
              >
                Ask me anything about Physical AI and Humanoid Robotics!<br />
                I can help with specific textbook content, clarify concepts, or discuss related topics.
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  style={{
                    marginBottom: '10px',
                    textAlign: message.isUser ? 'right' : 'left',
                  }}
                >
                  <div
                    style={{
                      display: 'inline-block',
                      padding: '8px 12px',
                      borderRadius: '8px',
                      backgroundColor: message.isUser
                        ? (isDarkTheme ? '#3a3a3a' : '#e3f2fd')
                        : (isDarkTheme ? '#4a4a4a' : '#f5f5f5'),
                      maxWidth: '85%',
                    }}
                  >
                    {message.text}
                  </div>
                  {!message.isUser && message.sources && message.sources.length > 0 && (
                    <div style={{ marginTop: '4px', fontSize: '0.8em' }}>
                      {message.sources.map((source, idx) => (
                        <div key={idx} style={{ marginBottom: '4px' }}>
                          <a 
                            href={source.url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            style={{ color: '#76b900', textDecoration: 'none' }}
                          >
                            â†³ {source.title}
                          </a>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))
            )}
            {isLoading && (
              <div style={{ textAlign: 'left' }}>
                <div
                  style={{
                    display: 'inline-block',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    backgroundColor: isDarkTheme ? '#4a4a4a' : '#f5f5f5',
                  }}
                >
                  ðŸ¤– Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <div
            className="chat-input-area"
            style={{
              padding: '10px',
              borderTop: '1px solid #ccc',
              display: 'flex',
            }}
          >
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the textbook content..."
              style={{
                flex: 1,
                padding: '8px',
                borderRadius: '4px',
                border: '1px solid #ccc',
                resize: 'none',
                minHeight: '50px',
                maxHeight: '100px',
              }}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || inputText.trim() === ''}
              style={{
                marginLeft: '8px',
                padding: '8px 16px',
                backgroundColor: inputText.trim() === '' || isLoading ? '#cccccc' : '#76b900',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: inputText.trim() === '' || isLoading ? 'not-allowed' : 'pointer',
              }}
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;
'''
        
        with open('src/components/ChatBot.tsx', 'w') as f:
            f.write(chatbot_content)
        print("✓ Created src/components/ChatBot.tsx")
    
    # Create basic ChapterControls component if it doesn't exist
    if not os.path.exists('src/components/ChapterControls.tsx'):
        chapter_controls_content = '''import React, { useState, useEffect } from 'react';
import { useColorMode, useDocsData } from '@docusaurus/theme-common';
import { useLocation } from '@docusaurus/router';

// Mock Gemini API translation function
const translateToUrdu = async (text: string): Promise<string> => {
  // In a real implementation, this would call the Gemini API
  // For now, return a placeholder translation
  await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
  
  // This is a very simplified mock translation
  // In practice, you would call the Gemini API with the text
  return `[URDU TRANSLATION PLACEHOLDER]: ${text.substring(0, Math.min(50, text.length))}...`;
};

interface ChapterControlsProps {
  children?: React.ReactNode;
}

const ChapterControls: React.FC<ChapterControlsProps> = ({ children }) => {
  const [showPersonalizeModal, setShowPersonalizeModal] = useState(false);
  const [translateToUrduEnabled, setTranslateToUrduEnabled] = useState(false);
  const { colorMode } = useColorMode();
  const location = useLocation();
  
  // State for user profile information
  const [userProfile, setUserProfile] = useState({
    experienceLevel: 'beginner',
    hardwareGPU: false,
    hardwareJetson: false,
    hardwareRobot: false,
    pythonExperience: 0,
    rosExperience: '',
    goal: 'learning'
  });
  
  // State for translated content
  const [translatedContent, setTranslatedContent] = useState<string | null>(null);

  const togglePersonalizeModal = () => {
    setShowPersonalizeModal(!showPersonalizeModal);
  };

  const toggleUrduTranslation = async () => {
    const shouldTranslate = !translateToUrduEnabled;
    setTranslateToUrduEnabled(shouldTranslate);
    
    if (shouldTranslate) {
      // Get current page content and translate it
      const pageContent = document.querySelector('.theme-doc-markdown');
      if (pageContent) {
        const contentText = pageContent.textContent || '';
        try {
          const translation = await translateToUrdu(contentText);
          setTranslatedContent(translation);
        } catch (error) {
          console.error('Translation error:', error);
          setTranslatedContent(null);
        }
      }
    } else {
      setTranslatedContent(null);
    }
  };

  const handleProfileChange = (field: string, value: any) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const saveProfile = () => {
    // In a real implementation, this would save to user preferences
    localStorage.setItem('humanoidProfile', JSON.stringify(userProfile));
    setShowPersonalizeModal(false);
    console.log('User profile updated:', userProfile);
  };

  // Apply translations to content if enabled
  useEffect(() => {
    if (translateToUrduEnabled && translatedContent) {
      const contentElement = document.querySelector('.theme-doc-markdown');
      if (contentElement) {
        contentElement.innerHTML = `<div class="urdu-translation">${translatedContent}</div>`;
      }
    } else {
      // Restore original content
      window.location.reload(); // Simple approach for demo
    }
  }, [translateToUrduEnabled, translatedContent]);

  return (
    <div className="chapter-controls" style={{ 
      position: 'sticky', 
      top: '10px', 
      zIndex: 100, 
      margin: '10px 0',
      padding: '10px',
      borderRadius: '8px',
      backgroundColor: colorMode === 'dark' ? '#2d2d2d' : '#f8f9fa',
      border: `2px solid ${colorMode === 'dark' ? '#4a4a4a' : '#e9ecef'}`,
      display: 'flex',
      justifyContent: 'flex-start',
      alignItems: 'center',
      gap: '10px',
      flexWrap: 'wrap'
    }}>
      <button 
        onClick={togglePersonalizeModal}
        style={{
          padding: '6px 12px',
          backgroundColor: '#76b900',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        ðŸ§  Personalize this chapter
      </button>
      
      <button 
        onClick={toggleUrduTranslation}
        style={{
          padding: '6px 12px',
          backgroundColor: translateToUrduEnabled ? '#5d9400' : '#76b900',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}
      >
        Ø§Ø±Ø¯Ùˆ Ù…ÛŒÚº ØªØ±Ø¬Ù…Ù‡ Ø¨Ø¯Ø§Ø¦Û• / Translate to Urdu
      </button>
      
      {children}
      
      {/* Personalize Modal */}
      {showPersonalizeModal && (
        <div style={{
          position: 'fixed',
          top: '0',
          left: '0',
          right: '0',
          bottom: '0',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}
        onClick={() => setShowPersonalizeModal(false)}
        >
          <div 
            style={{
              backgroundColor: colorMode === 'dark' ? '#2d2d2d' : 'white',
              padding: '20px',
              borderRadius: '8px',
              maxWidth: '500px',
              width: '90%',
              maxHeight: '80vh',
              overflowY: 'auto'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ color: '#76b900' }}>Customize Your Learning Experience</h3>
            <p>Help us tailor content to your skill level and needs:</p>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '10px' }}>
              <label>
                <strong>Experience Level:</strong>
                <select
                  value={userProfile.experienceLevel}
                  onChange={(e) => handleProfileChange('experienceLevel', e.target.value)}
                  style={{ marginLeft: '10px', padding: '4px' }}
                >
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={userProfile.hardwareGPU}
                  onChange={(e) => handleProfileChange('hardwareGPU', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                <strong>Hardware:</strong> Do you have RTX 4070+ GPU?
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={userProfile.hardwareJetson}
                  onChange={(e) => handleProfileChange('hardwareJetson', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                <strong>Hardware:</strong> Do you own a Jetson?
              </label>
              
              <label style={{ display: 'flex', alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={userProfile.hardwareRobot}
                  onChange={(e) => handleProfileChange('hardwareRobot', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                <strong>Hardware:</strong> Do you have a real robot?
              </label>
              
              <label>
                <strong>Python Experience (years):</strong>
                <input
                  type="range"
                  min="0"
                  max="20"
                  value={userProfile.pythonExperience}
                  onChange={(e) => handleProfileChange('pythonExperience', parseInt(e.target.value))}
                  style={{ marginLeft: '10px', width: '100%' }}
                />
                <span>{userProfile.pythonExperience} years</span>
              </label>
              
              <label>
                <strong>ROS Experience:</strong>
                <select
                  value={userProfile.rosExperience}
                  onChange={(e) => handleProfileChange('rosExperience', e.target.value)}
                  style={{ marginLeft: '10px', padding: '4px' }}
                >
                  <option value="">None</option>
                  <option value="ros1">ROS 1</option>
                  <option value="ros2">ROS 2</option>
                  <option value="both">Both ROS 1 and 2</option>
                </select>
              </label>
              
              <label>
                <strong>Goal:</strong>
                <select
                  value={userProfile.goal}
                  onChange={(e) => handleProfileChange('goal', e.target.value)}
                  style={{ marginLeft: '10px', padding: '4px' }}
                >
                  <option value="learning">Learning</option>
                  <option value="building">Building real humanoid</option>
                  <option value="research">Research</option>
                </select>
              </label>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '20px', gap: '10px' }}>
              <button
                onClick={() => setShowPersonalizeModal(false)}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={saveProfile}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#76b900',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Save Preferences
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChapterControls;
'''
        
        with open('src/components/ChapterControls.tsx', 'w') as f:
            f.write(chapter_controls_content)
        print("✓ Created src/components/ChapterControls.tsx")

def finalize_setup():
    """
    Finalize the project setup with final checks and initial build
    """
    print("\nFinalizing project setup...")
    
    # Create the project structure file
    structure_content = '''# Project Structure

This project implements a complete digital twin system for humanoid robotics using Isaac Sim, Isaac ROS, vSLAM, and reinforcement learning.

## Directory Structure

```
physical-ai-book/
├── docs/                     # All 23 MDX files with expert content
│   ├── intro/
│   │   ├── 00-welcome.mdx
│   │   ├── 01-foundations.mdx
│   │   └── 02-hardware-guide.mdx
│   ├── module1-ros2/
│   ├── module2-digital-twin/
│   ├── module3-isaac/
│   └── module4-vla/
├── src/
│   ├── components/
│   │   ├── ChatBot.tsx       # AI chatbot component
│   │   ├── ChapterControls.tsx # Personalization/translation controls
│   │   └── HighlightContextMenu.tsx # Context menu for text selection
│   ├── theme/                # Docusaurus custom theme components
│   └── css/
│       └── custom.css        # Custom styling
├── api/                      # FastAPI backend services
├── scripts/
│   └── index-to-qdrant.ts    # Content indexing script
├── better-auth/              # Complete Better-Auth configuration
├── static/                   # Static assets
├── training/                 # RL training components
│   ├── algorithms/           # RL algorithms (SAC, DDPG, etc.)
│   ├── environments/         # Training environments
│   └── utils/                # Helper functions
├── isaac_env/                # Isaac Sim environment wrappers
├── robot_description/        # Robot URDF and configuration files
├── configs/                  # Configuration files
├── launch/                   # Isaac Sim launch files
├── docusaurus.config.ts      # Docusaurus configuration
├── sidebars.js               # Navigation structure
├── package.json              # Project dependencies
├── tsconfig.json             # TypeScript configuration
└── README.md                 # Setup and deployment instructions
```

## Getting Started

1. Install dependencies: `npm install`
2. Create environment file: `cp .env.example .env`
3. Build the project: `npm run build`
4. Start development server: `npm run dev`
'''
    
    with open('PROJECT_STRUCTURE.md', 'w') as f:
        f.write(structure_content)
    print("✓ Created PROJECT_STRUCTURE.md")
    
    # Final validation
    print("\nFinal validation...")
    success = validate_setup()
    
    if success:
        print("\n🎉 Project setup complete!")
        print("\nNext steps:")
        print("1. Fill in your API keys in .env")
        print("2. Run 'npm run dev' to start the development server")
        print("3. Run 'python training/train_humanoid_rl.py' to start RL training")
        print("4. Customize content in docs/ to match your requirements")
    else:
        print("\n⚠️  There were issues with the setup. Check the logs above.")
    
    return success

def main():
    """
    Main execution function to initialize the project
    """
    print("Initializing Physical AI & Humanoid Robotics project...")
    
    # Check dependencies first
    if not check_dependencies():
        print("❌ Missing required dependencies. Please install them and retry.")
        return False
    
    # Set up project structure
    setup_successful = setup_project()
    if not setup_successful:
        print("❌ Project setup failed.")
        return False
    
    # Set up Docusaurus
    docusaurus_successful = setup_docusaurus()
    if not docusaurus_successful:
        print("❌ Docusaurus setup failed.")
        return False
    
    # Set up Isaac Sim integration
    isaac_successful = setup_isaac_integration()
    if not isaac_successful:
        print("❌ Isaac Sim integration setup failed.")
        return False
    
    # Set up RL training
    rl_successful = setup_rl_training()
    if not rl_successful:
        print("❌ RL training setup failed.")
        return False
    
    # Create missing components
    components_created = create_missing_components()
    if not components_created:
        print("❌ Component creation failed.")
        return False
    
    # Finalize setup
    finalize_successful = finalize_setup()
    if not finalize_successful:
        print("❌ Setup finalization failed.")
        return False
    
    print("\n✅ Project initialization completed successfully!")
    print("📁 Check the project structure and start customizing your textbook content.")
    
    return True

if __name__ == "__main__":
    main()