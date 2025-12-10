# isaac_sim_humanoid_env.py
"""
Isaac Sim environment for humanoid robot reinforcement learning
"""
import gym
from gym import spaces
import numpy as np
import torch
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.sensors import ImuSensor
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, Sdf, UsdGeom
import carb

class IsaacSimHumanoidEnvironment(gym.Env):
    """
    Isaac Sim environment for humanoid robot reinforcement learning
    """
    
    def __init__(self, headless=True, physics_dt=1.0/600.0, rendering_dt=1.0/60.0):
        super(IsaacSimHumanoidEnvironment, self).__init__()
        
        # Initialize Isaac Sim world
        self._world = World(
            stage_units_in_meters=1.0,
            rendering_dt=rendering_dt,
            sim_params={
                "use_gpu": True,
                "use_fabric": True,
                "solver_type": "TGS",
                "num_position_iterations": 8,
                "num_velocity_iterations": 1,
                "max_depenetration_velocity": 1000.0,
                "substeps": 2,
                "dt": physics_dt  # Physics timestep: 600Hz
            }
        )
        
        # Environment parameters
        self.headless = headless
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.episode_length = 1000  # 1000 steps per episode at 600Hz = ~1.67 seconds per episode
        self.step_counter = 0
        
        # Robot parameters
        self.action_dim = 20  # 20 joints for humanoid
        self.state_dim = 122  # Includes joint positions, velocities, IMU, etc.
        
        # Define action and observation spaces
        # Actions: joint position targets (delta from current position)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.action_dim,), 
            dtype=np.float32
        )
        
        # Observation: [joint_positions(20), joint_velocities(20), base_orientation(4), 
        #               base_angular_vel(3), base_linear_vel(3), com_pos(3), 
        #               com_vel(3), contact_states(2), imu_data(10), command(6)]
        obs_high = np.inf * np.ones(self.state_dim)
        self.observation_space = spaces.Box(
            low=-obs_high, 
            high=obs_high, 
            dtype=np.float32
        )
        
        # Robot properties
        self.robot = None
        self.robot_name = "HumanoidRobot"
        self.robot_usd_path = "/Users/user/Desktop/fn-bk/robot_description/humanoid.urdf.xacro"
        
        # Initialize the robot in the environment
        self._setup_scene()
        
        # Initialize state variables
        self.current_joint_positions = np.zeros(self.action_dim)
        self.current_joint_velocities = np.zeros(self.action_dim)
        self.base_orientation = np.array([0, 0, 0, 1])  # quaternion
        self.base_angular_velocity = np.zeros(3)
        self.base_linear_velocity = np.zeros(3)
        self.com_position = np.zeros(3)
        self.com_velocity = np.zeros(3)
        self.contact_states = np.zeros(2)  # left_foot, right_foot contact
        self.imu_data = np.zeros(10)  # simulated IMU data
        self.command = np.zeros(6)  # target position and orientation
        
        self.initial_positions = None
        
    def _setup_scene(self):
        """
        Set up the simulation scene with the humanoid robot
        """
        # Add ground plane
        self._world.scene.add_default_ground_plane()
        
        # Set up lighting
        dome_light = self._world.scene.add(
            omni.isaac.core.utils.prims.define_and_create_prim(
                prim_path="/World/Light",
                prim_type_name="DomeLight",
                attributes={"color": (0.8, 0.8, 0.8), "intensity": 3000}
            )
        )
        
        # Add the humanoid robot to the scene
        try:
            # Load robot from URDF
            add_reference_to_stage(usd_path=self.robot_usd_path, prim_path="/World/HumanoidRobot")
            
            # Create robot object
            self.robot = self._world.scene.add(
                Robot(
                    prim_path="/World/HumanoidRobot",
                    name=self.robot_name,
                    position=np.array([0.0, 0.0, 1.0]),
                    orientation=np.array([0.0, 0.0, 0.0, 1.0])
                )
            )
            
            # Get initial joint positions
            self.initial_positions = self.robot.get_joint_positions()
            
            # Add IMU sensor to torso
            self.imu_sensor = self._world.scene.add(
                ImuSensor(
                    prim_path="/World/HumanoidRobot/torso/imu",
                    name="torso_imu",
                    position=np.array([0.0, 0.0, 0.15]),
                    frequency=100  # 100Hz IMU
                )
            )
            
        except Exception as e:
            carb.log_error(f"Failed to set up scene: {e}")
            raise
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state
        """
        super().reset(seed=seed)
        
        # Reset the Isaac Sim world
        self._world.reset()
        
        # Reset robot to initial configuration
        if self.robot:
            self.robot.set_joint_positions(self.initial_positions)
            self.robot.set_world_poses(
                positions=np.array([[0.0, 0.0, 1.0]]),
                orientations=np.array([[0.0, 0.0, 0.0, 1.0]])
            )
            
            # Set initial velocities to zero
            self.robot.set_velocities(
                linear_velocities=np.array([[0.0, 0.0, 0.0]]),
                angular_velocities=np.array([[0.0, 0.0, 0.0]])
            )
        
        # Reset counters
        self.step_counter = 0
        
        # Get initial observation
        observation = self._compute_observation()
        
        # Create info dictionary
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment
        """
        # Validate action
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action to robot (in this case, joint position targets)
        if self.robot:
            # Convert action to joint position deltas
            position_deltas = action * 0.02  # Small deltas for stability
            current_positions = self.robot.get_joint_positions()
            target_positions = current_positions + position_deltas
            
            # Apply position targets
            self.robot.set_joint_position_targets(target_positions)
        
        # Step the physics simulation
        self._world.step(render=False)
        
        # Get new state
        observation = self._compute_observation()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.step_counter >= self.episode_length
        
        # Update step counter
        self.step_counter += 1
        
        # Create info dictionary
        info = {
            'episode': {'r': reward, 'l': self.step_counter},
            'success': not terminated and not truncated
        }
        
        return observation, reward, terminated, truncated, info
    
    def _compute_observation(self):
        """
        Compute the current observation vector
        """
        if not self.robot:
            return np.zeros(self.state_dim)
        
        # Get joint states
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        
        # Get robot base state
        base_positions, base_orientations = self.robot.get_world_poses()
        base_linear_velocities, base_angular_velocities = self.robot.get_velocities()
        
        # Calculate center of mass (simplified as torso position)
        com_pos = base_positions[0]  # Using pelvis/torso as CoM approximation
        com_vel = base_linear_velocities[0]
        
        # Get contact information (simplified)
        # In a real implementation, this would come from contact sensors
        contact_states = self._get_foot_contact_states()
        
        # Get IMU data (simplified - in real implementation, get from actual sensor)
        imu_data = self._get_simulated_imu_data(base_orientations[0], base_angular_velocities[0])
        
        # Create command vector (simplified - target for forward walking)
        command = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0])  # Target forward velocity [x, y, z, ?, ?, ?]
        
        # Normalize joint positions to [-1, 1]
        normalized_joint_pos = np.clip(joint_positions / np.pi, -1.0, 1.0)
        
        # Normalize joint velocities 
        max_vel = 10.0  # rad/s
        normalized_joint_vel = np.clip(joint_velocities / max_vel, -1.0, 1.0)
        
        # Combine all observation components
        observation = np.concatenate([
            normalized_joint_pos[:20],           # 20 joint positions
            normalized_joint_vel[:20],           # 20 joint velocities
            base_orientations[0],                # 4 base orientation (quaternion)
            base_angular_velocities[0],          # 3 base angular velocity
            base_linear_velocities[0],           # 3 base linear velocity
            com_pos - np.array([0, 0, 1.0]),    # 3 CoM position relative to starting height
            com_vel,                             # 3 CoM velocity
            contact_states,                      # 2 foot contact states
            imu_data,                            # 10 IMU data (simplified)
            command                              # 6 command vector
        ]).astype(np.float32)
        
        # Ensure correct dimension
        assert len(observation) == self.state_dim, f"Observation dimension mismatch: {len(observation)} != {self.state_dim}"
        
        return observation
    
    def _get_foot_contact_states(self):
        """
        Get contact states for feet (simplified)
        """
        # In a real implementation, this would use contact sensors
        # For now, return simulated contact based on position
        if self.robot:
            base_pos, _ = self.robot.get_world_poses()
            z_pos = base_pos[0][2]
            
            # Simple contact detection (if feet are near ground)
            left_contact = 1.0 if z_pos < 0.1 else 0.0  # Simplified
            right_contact = 1.0 if z_pos < 0.1 else 0.0  # Simplified
            
            return np.array([left_contact, right_contact])
        else:
            return np.array([0.0, 0.0])
    
    def _get_simulated_imu_data(self, orientation, angular_velocity):
        """
        Simulate IMU data based on current state
        """
        # In a real implementation, this would come from actual IMU sensor
        # For simulation, create synthetic data based on robot state
        
        # Add some noise to the data
        noise = np.random.normal(0, 0.01, 10)
        
        # Create synthetic IMU data [linear_acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, temp]
        data = np.concatenate([
            np.array([0.0, 0.0, 9.81]),  # Gravity in z direction
            angular_velocity,             # Angular velocity from robot
            np.array([0.2, -0.1, 0.3]),  # Magnetic field approximation
            np.array([25.0])              # Temperature
        ])
        
        return data + noise
    
    def _compute_reward(self):
        """
        Compute reward for the current state
        """
        if not self.robot:
            return 0.0
        
        # Get current state
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        
        base_positions, base_orientations = self.robot.get_world_poses()
        base_linear_vels, base_angular_vels = self.robot.get_velocities()
        
        # Reward components
        alive_bonus = 1.0
        progress_reward = base_linear_vels[0][0] * 0.5  # Forward velocity reward
        
        # Upward orientation reward (keep robot upright)
        z_axis = self._quat_rotate_vector(base_orientations[0], np.array([0, 0, 1]))
        upright_reward = max(0, z_axis[2]) * 2.0  # Only reward when pointing up
        
        # Energy penalty (avoid excessive joint velocities)
        energy_penalty = -0.01 * np.sum(np.square(joint_velocities))
        
        # Joint limit penalty
        joint_limit_penalty = self._compute_joint_limit_penalty(joint_positions)
        
        # Compute total reward
        total_reward = (
            alive_bonus +
            progress_reward +
            upright_reward +
            energy_penalty +
            joint_limit_penalty
        )
        
        return total_reward
    
    def _compute_joint_limit_penalty(self, joint_positions):
        """
        Compute penalty for approaching joint limits
        """
        # Define approximate joint limits (in radians)
        # These would come from robot description in practice
        min_limits = np.full(20, -2.0)  # Placeholder values
        max_limits = np.full(20, 2.0)   # Placeholder values
        
        # Calculate how close joints are to limits
        penalties = 0.0
        for i, pos in enumerate(joint_positions[:20]):  # Only check first 20 for now
            if i < len(min_limits):
                # Calculate normalized distance to limits
                range_size = max_limits[i] - min_limits[i]
                if range_size > 0.1:  # Avoid division by zero
                    dist_to_min = abs(pos - min_limits[i]) / range_size
                    dist_to_max = abs(max_limits[i] - pos) / range_size
                    closest_limit_norm = min(dist_to_min, dist_to_max)
                    
                    # Apply higher penalty when close to limits (< 10% range from limit)
                    if closest_limit_norm < 0.1:
                        penalties += (0.1 - closest_limit_norm) * 100.0  # High penalty near limits
        
        return -penalties
    
    def _is_terminated(self):
        """
        Check if the episode is terminated (robot fell, etc.)
        """
        if not self.robot:
            return True
            
        # Get robot pose
        base_positions, base_orientations = self.robot.get_world_poses()
        
        # Check if robot has fallen (simplified criterion)
        height = base_positions[0][2]
        if height < 0.5:  # Robot considered fallen if below 0.5m
            return True
            
        # Check if robot is tilted too much
        z_axis = self._quat_rotate_vector(base_orientations[0], np.array([0, 0, 1]))
        if z_axis[2] < 0.5:  # Tilted more than ~60 degrees from upright
            return True
        
        return False
    
    def _quat_rotate_vector(self, quat, vec):
        """
        Rotate a vector by a quaternion
        """
        qw, qx, qy, qz = quat
        vx, vy, vz = vec
        
        # Quaternion rotation formula
        ww = qw*qw
        xx = qx*qx
        yy = qy*qy
        zz = qz*qz
        wx = qw*qx
        wy = qw*qy
        wz = qw*qz
        xy = qx*qy
        xz = qx*qz
        yz = qy*qz
        
        rotated_x = ww*vx + 2*qy*vy*wz - 2*qz*vz*wy + 2*qz*vy*wx - 2*qx*vz*wz + 2*qx*vy*wy - 2*qy*vz*wx - xx*vx - yy*vx + zz*vx
        rotated_y = ww*vy - 2*qx*vx*wz + 2*qz*vz*wx + 2*qx*vx*wy - 2*qy*vy*wx + 2*qy*vx*wz - 2*qz*vz*qy - xx*vy + yy*vy - zz*vy
        rotated_z = ww*vz + 2*qx*vx*wy - 2*qy*vx*wz - 2*qx*vy*wz + 2*qy*vx*wy + 2*qz*vx*wx - 2*qw*vy*wz - xx*vz + yy*vz - zz*vz
        
        return np.array([rotated_x, rotated_y, rotated_z])
    
    def render(self, mode='human'):
        """
        Render the environment (not implemented for Isaac Sim in this wrapper)
        """
        # Isaac Sim handles rendering internally
        pass
    
    def close(self):
        """
        Close the environment
        """
        if self._world:
            self._world.clear()
        super().close()