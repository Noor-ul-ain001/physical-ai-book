# Main project integration file
# project_integration.py

from isaac_ros_core import HumanoidWorld
from training.humanoid_rl_trainer import HumanoidTrainer
from models.humanoid_networks import *
import json
import os

class HumanoidDigitalTwinIntegration:
    """
    Integration class that ties together Isaac Sim, vSLAM, ROS, and RL for complete digital twin
    """
    def __init__(self, config_path="config/humanoid_training_config.json"):
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.simulation_world = None
        self.rl_agent = None
        self.vslam_system = None
        self.ros_bridge = None
        self.navigation_system = None
        
    def load_config(self, config_path):
        """
        Load configuration from JSON file
        """
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def initialize_simulation(self):
        """
        Initialize Isaac Sim environment for humanoid robot
        """
        print("Initializing Isaac Sim environment...")
        
        # Create Isaac Sim world instance
        self.simulation_world = HumanoidWorld()
        
        # Configure physics parameters
        self.simulation_world.set_physics_engine_params(
            backend="torch",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            dt=1.0/500.0,  # 500Hz physics
            substeps=2
        )
        
        # Set up scene
        self.simulation_world.setup_humanoid_environment(self.config)
        
        print("Isaac Sim environment initialized successfully")
    
    def initialize_rl_agent(self):
        """
        Initialize the RL agent for humanoid control
        """
        print("Initializing RL agent...")
        
        # Create appropriate RL agent based on config
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        max_action = 1.0  # For normalized action space
        
        # Initialize SAC agent with humanoid-specific parameters
        self.rl_agent = HumanoidSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            **self.config['rl_params']
        )
        
        print("RL agent initialized successfully")
    
    def initialize_vslam(self):
        """
        Initialize vSLAM system for navigation and mapping
        """
        print("Initializing vSLAM system...")
        
        # In a real implementation, this would connect to Isaac Sim cameras
        # and run visual SLAM algorithms
        from modules.vslam import IsaacSimVSLAM
        
        self.vslam_system = IsaacSimVSLAM(
            camera_topics=self.config.get('camera_topics', ['/front_cam/image_raw']),
            map_resolution=self.config.get('vslam_resolution', 0.1),
            max_range=self.config.get('vslam_range', 10.0)
        )
        
        print("vSLAM system initialized successfully")
    
    def initialize_ros_bridge(self):
        """
        Initialize ROS bridge for external communication
        """
        print("Initializing ROS bridge...")
        
        # Initialize ROS2 node for communication with external systems
        import rclpy
        from std_msgs.msg import String
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import JointState
        
        rclpy.init()
        
        # Create ROS node
        self.ros_bridge = rclpy.create_node('humanoid_digital_twin_bridge')
        
        # Publishers for robot state
        self.joint_state_pub = self.ros_bridge.create_publisher(JointState, '/joint_states', 10)
        self.robot_pose_pub = self.ros_bridge.create_publisher(PoseStamped, '/robot_pose', 10)
        
        # Subscribers for commands
        self.cmd_vel_sub = self.ros_bridge.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        
        print("ROS bridge initialized successfully")
    
    def initialize_navigation(self):
        """
        Initialize navigation system with path planning and obstacle avoidance
        """
        print("Initializing navigation system...")
        
        # Initialize 3D navigation system for humanoid
        from modules.navigation import Humanoid3DNavigator
        
        self.navigation_system = Humanoid3DNavigator(
            map_resolution=self.config.get('navigation_resolution', 0.2),
            robot_radius=self.config.get('robot_radius', 0.4),
            robot_height=self.config.get('robot_height', 1.6),
            max_slope=self.config.get('max_slope', 0.3)  # 30-degree max slope
        )
        
        print("Navigation system initialized successfully")
    
    def cmd_vel_callback(self, msg):
        """
        ROS callback for velocity commands
        """
        # Convert Twist command to humanoid action
        # This would interface with the RL agent or navigation system
        pass
    
    def run_integration_test(self):
        """
        Run integration test to verify all components work together
        """
        print("Running integration test...")
        
        # Test simulation
        if self.simulation_world:
            print("✓ Simulation world is active")
        else:
            print("✗ Simulation world not initialized")
            return False
        
        # Test RL agent
        if self.rl_agent:
            print("✓ RL agent is active")
        else:
            print("✗ RL agent not initialized")
            return False
        
        # Test vSLAM
        if self.vslam_system:
            print("✓ vSLAM system is active")
        else:
            print("✗ vSLAM system not initialized")
            return False
        
        # Test ROS bridge
        if self.ros_bridge:
            print("✓ ROS bridge is active")
        else:
            print("✗ ROS bridge not initialized")
            return False
        
        # Test navigation
        if self.navigation_system:
            print("✓ Navigation system is active")
        else:
            print("✗ Navigation system not initialized")
            return False
        
        print("Integration test passed! All components are ready.")
        return True
    
    def execute_training_run(self):
        """
        Execute a complete training run using all integrated components
        """
        if not self.run_integration_test():
            print("Integration test failed. Cannot proceed with training.")
            return
        
        print("Starting integrated training run...")
        
        # Initialize trainer with all components
        trainer = HumanoidTrainer(self.config)
        
        # Set up simulation environment in trainer
        trainer.world = self.simulation_world
        trainer.agent = self.rl_agent
        
        # Start training
        try:
            trainer.train()
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise
    
    def sync_to_real_robot(self):
        """
        Synchronize digital twin to real robot (when available)
        """
        print("Preparing for real robot deployment...")
        
        # In a real implementation, this would:
        # 1. Convert trained policy to real robot control format
        # 2. Handle sim-to-real domain transfer
        # 3. Set up real robot interface
        
        # For now, return a deployment package
        deployment_package = {
            'policy_weights': self.rl_agent.actor.state_dict(),
            'control_parameters': self.get_deployable_control_params(),
            'calibration_data': self.estimate_sim_to_real_params()
        }
        
        return deployment_package
    
    def get_deployable_control_params(self):
        """
        Get robot-specific control parameters for deployment
        """
        return {
            'control_frequency': 500,  # Hz
            'torque_limits': self.config.get('torque_limits', [100.0] * 20),  # 20 joints
            'position_limits': self.config.get('position_limits', [2.0] * 20),
            'max_velocity': self.config.get('max_velocity', 5.0),
            'safety_margins': self.config.get('safety_margins', 0.1)
        }
    
    def estimate_sim_to_real_params(self):
        """
        Estimate parameters for sim-to-real transfer
        """
        return {
            'mass_calibration': 1.05,  # 5% increase for real robot
            'inertia_scaling': 1.1,    # 10% increase for real robot
            'friction_compensation': 0.9,  # 10% decrease for smoother motion
            'actuator_delay': 0.02,    # 20ms actuator delay in real robot
            'sensor_noise_multiplier': 1.5  # 50% more noise in real sensors
        }

def main():
    """
    Main function to run the complete digital twin integration
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Humanoid Digital Twin Integration')
    parser.add_argument('--config', type=str, default='config/humanoid_training_config.json',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                       help='Mode: train, eval, or deploy')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pre-trained model for evaluation')
    
    args = parser.parse_args()
    
    # Create integration instance
    integ = HumanoidDigitalTwinIntegration(args.config)
    
    # Initialize all components
    integ.initialize_simulation()
    integ.initialize_rl_agent()
    integ.initialize_vslam()
    integ.initialize_ros_bridge()
    integ.initialize_navigation()
    
    if args.mode == 'train':
        # Run training
        integ.execute_training_run()
    elif args.mode == 'eval':
        # Run evaluation (if checkpoint provided)
        if args.checkpoint:
            # Load pre-trained model and evaluate
            import torch
            integ.rl_agent.actor.load_state_dict(
                torch.load(f"{args.checkpoint}/actor.pth")
            )
            
            # Run evaluation
            eval_reward = integ.evaluate_policy()
            print(f"Evaluation completed. Average reward: {eval_reward:.3f}")
        else:
            print("Checkpoint path required for evaluation mode")
    elif args.mode == 'deploy':
        # Generate deployment package
        deployment_pkg = integ.sync_to_real_robot()
        print("Deployment package generated successfully!")
        
        # Save deployment package
        import pickle
        with open('deployment_package.pkl', 'wb') as f:
            pickle.dump(deployment_pkg, f)
        
        print("Deployment package saved to deployment_package.pkl")
    else:
        print(f"Unknown mode: {args.mode}. Use train, eval, or deploy.")

if __name__ == "__main__":
    main()