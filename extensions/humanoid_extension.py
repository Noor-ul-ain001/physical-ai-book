# Isaac Sim extension for humanoid robot integration
# extensions/humanoid_extension.py

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera, Imu
import numpy as np
import carb
import omni.ui as ui
from pxr import Gf, Sdf, UsdGeom
import torch
import torch.nn as nn

class HumanoidRobot(Robot):
    """
    Custom humanoid robot class extending Isaac Sim's Robot class
    """
    def __init__(
        self,
        prim_path: str,
        name: str = "humanoid_robot",
        usd_path: str = None,
        position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        orientation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0]),
        articulation_controller = None
    ):
        self._usd_path = usd_path
        self._position = position
        self._orientation = orientation
        
        # Physical properties
        self._mass_distribution = {
            'pelvis': 8.0, 'torso': 5.0, 'head': 3.0,
            'arm': 2.0, 'leg': 5.0
        }
        
        # Joint limits
        self._joint_limits = {
            'hip_yaw': [-0.52, 0.52],
            'hip_pitch': [-1.57, 1.57],
            'hip_roll': [-0.35, 1.05],
            'knee': [-1.57, 0.0],
            'ankle_pitch': [-0.78, 0.52],
            'ankle_roll': [-0.35, 0.35]
        }
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            usd_path=usd_path,
            position=position,
            orientation=orientation,
            articulation_controller=articulation_controller
        )

    def set_joint_position_targets(self, positions, joint_indices=None):
        """
        Set joint position targets for the humanoid robot
        """
        if joint_indices is not None:
            self.get_articulation_controller().set_command_targets(
                values=positions,
                joint_indices=joint_indices
            )
        else:
            self.get_articulation_controller().set_command_targets(values=positions)

    def set_joint_velocity_targets(self, velocities, joint_indices=None):
        """
        Set joint velocity targets for the humanoid robot
        """
        if joint_indices is not None:
            self.get_articulation_controller().set_command_targets(
                values=velocities,
                joint_indices=joint_indices,
                targets_type="velocities"
            )
        else:
            self.get_articulation_controller().set_command_targets(
                values=velocities,
                targets_type="velocities"
            )

    def get_joint_states(self):
        """
        Get current joint positions, velocities, and efforts
        """
        positions = self.get_joint_positions()
        velocities = self.get_joint_velocities()
        efforts = self.get_applied_joint_efforts()
        
        return {
            'positions': positions,
            'velocities': velocities,
            'efforts': efforts
        }

    def get_imu_data(self):
        """
        Get IMU data from the robot's head/center
        """
        # This would return orientation, angular velocity, linear acceleration
        try:
            # Get root pose
            pos, orn = self.get_world_poses()
            
            # Get root velocity
            lin_vel, ang_vel = self.get_velocities()
            
            return {
                'orientation': orn,
                'angular_velocity': ang_vel,
                'linear_acceleration': self.estimate_linear_acceleration(lin_vel)
            }
        except:
            # Return default values if not available
            return {
                'orientation': [0, 0, 0, 1],
                'angular_velocity': [0, 0, 0],
                'linear_acceleration': [0, 0, -9.81]
            }

    def estimate_linear_acceleration(self, linear_velocity):
        """
        Estimate linear acceleration from linear velocity
        """
        # In simulation we can compute this more accurately
        # For now, returning a simplified approach
        # In practice, would use IMU or differentiate velocity
        return [0, 0, 0]

class HumanoidWorld(World):
    """
    Customized World class for humanoid robot simulation
    """
    def __init__(self):
        super().__init__(stage_units_in_meters=1.0)
        
        # Humanoid-specific properties
        self.humanoids = []
        self.terrain_generator = None
        self.obstacle_manager = None
        self.terrain_complexity = 0.5  # 0.0 to 1.0, affects difficulty
        
        # Physics parameters optimized for humanoid simulation
        self.set_physics_engine_params(
            backend="torch",
            device="cuda:0",
            dt=1.0/500.0,  # 500Hz physics
            substeps=2
        )
    
    def setup_humanoid_environment(self, env_config):
        """
        Set up the humanoid-specific environment
        """
        # Add ground plane
        self.scene.add_default_ground_plane()
        
        # Configure lighting
        self.setup_lighting()
        
        # Set up physics parameters
        self.configure_physics_params()
        
        # Load humanoid robot
        self.load_humanoid_robot(env_config)
        
        # Set up sensors
        self.setup_sensors()
        
    def setup_lighting(self):
        """
        Set up appropriate lighting for humanoid simulation
        """
        # Add dome light
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.lighting.distant_light import DistantLight
        
        # Create distant light (sun)
        self.sun = DistantLight(
            prim_path="/World/DistantLight",
            intensity=3000,
            color=(0.9, 0.9, 0.9)
        )
        
        # Add some fill lights for better visibility
        from omni.isaac.core.lighting.light import Light
        create_prim(
            "/World/KeyLight",
            "DistantLight",
            position=np.array([5, 5, 10]),
            attributes={"inputs:intensity": 1000}
        )
    
    def configure_physics_params(self):
        """
        Configure physics parameters optimized for humanoid simulation
        """
        # Set gravity
        self.scene.set_gravity([0.0, 0.0, -9.81])
        
        # Physics solver settings
        carb.settings.get_settings().set("/physics_solver_type", "TGS")
        carb.settings.get_settings().set("/physics_solver_position_iteration_count", 10)
        carb.settings.get_settings().set("/physics_solver_velocity_iteration_count", 10)
        
        # Contact handling
        carb.settings.get_settings().set("/physics_contact_collection", 0)  # Collect all contacts
        carb.settings.get_settings().set("/physics_max_depenetration_velocity", 10.0)
    
    def load_humanoid_robot(self, config):
        """
        Load the humanoid robot into the simulation
        """
        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return False
        
        # Load the humanoid robot
        humanoid_asset_path = config.get("robot_usd_path", "path/to/humanoid.usd")
        
        # Add reference to stage
        add_reference_to_stage(usd_path=humanoid_asset_path, prim_path="/World/HumanoidRobot")
        
        # Create robot instance
        self.humanoid_robot = HumanoidRobot(
            prim_path="/World/HumanoidRobot",
            name="humanoid_robot",
            usd_path=humanoid_asset_path,
            position=np.array(config.get("initial_position", [0.0, 0.0, 1.0])),
            orientation=np.array(config.get("initial_orientation", [0.0, 0.0, 0.0, 1.0]))
        )
        
        # Add to scene
        self.scene.add(self.humanoid_robot)
        
        # Initialize the robot after adding to scene
        self._world.reset()
        self.humanoid_robot.initialize(world=self._world)
        
        self.humanoids.append(self.humanoid_robot)
        
        carb.log_info("Humanoid robot loaded successfully")
        return True
    
    def setup_sensors(self):
        """
        Set up sensors for the humanoid robot
        """
        # Set up IMU in the head/torso
        self.head_imu = Imu(
            prim_path="/World/HumanoidRobot/head/imu",
            frequency=100  # 100 Hz
        )
        
        # Set up cameras
        self.head_camera = Camera(
            prim_path="/World/HumanoidRobot/head/head_camera",
            frequency=30,  # 30 Hz
            resolution=(640, 480)
        )
        
        # Set up force/torque sensors in feet
        self.left_foot_sensor = None  # Would need to be implemented in USD
        self.right_foot_sensor = None  # Would need to be implemented in USD
        
        carb.log_info("Sensors set up for humanoid robot")

    def add_terrain(self, terrain_type="flat", complexity=0.3):
        """
        Add terrain to the environment
        """
        if terrain_type == "flat":
            # Already have ground plane
            pass
        elif terrain_type == "rough":
            self._add_rough_terrain(complexity)
        elif terrain_type == "stairs":
            self._add_stairs(complexity)
        elif terrain_type == "obstacles":
            self._add_obstacle_course(complexity)
        
        carb.log_info(f"Added {terrain_type} terrain with complexity {complexity}")

    def _add_rough_terrain(self, complexity):
        """
        Add procedurally generated rough terrain
        """
        # This would use Isaac Sim's terrain generation capabilities
        # For now, we'll add some simple uneven ground
        from omni.isaac.core.objects.ground_plane import GroundPlane
        from omni.isaac.core.prims.cube import Cube
        
        # Add some randomly positioned cubes to create uneven terrain
        import random
        
        for i in range(int(complexity * 20)):  # More obstacles with higher complexity
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            z = random.uniform(0.01, complexity * 0.1)  # Height variation
            
            obstacle = Cube(
                prim_path=f"/World/Obstacle{i}",
                name=f"obstacle_{i}",
                position=np.array([x, y, z/2]),
                size=z,
                color=np.array([0.5, 0.5, 0.5])
            )
            self.scene.add(obstacle)
    
    def _add_stairs(self, complexity):
        """
        Add staircases of varying difficulty
        """
        from omni.isaac.core.prims.cube import Cube
        
        # Create a simple staircase
        step_height = 0.1 * complexity
        step_depth = 0.3
        step_width = 1.0
        num_steps = int(5 * complexity)
        
        for i in range(num_steps):
            x_pos = 2.0 + i * step_depth
            y_pos = 0.0
            z_pos = (i + 1) * step_height
            
            step = Cube(
                prim_path=f"/World/Step{i}",
                name=f"step_{i}",
                position=np.array([x_pos, y_pos, z_pos/2]),
                size=0.1,
                scale=np.array([step_depth, step_width, step_height]),
                color=np.array([0.3, 0.3, 0.3])
            )
            self.scene.add(step)

    def _add_obstacle_course(self, complexity):
        """
        Add an obstacle course with various challenges
        """
        from omni.isaac.core.prims.cube import Cube
        
        # Add various obstacles based on complexity
        obstacles = []
        positions = [
            [2, 0, 0.1], [3, 1, 0.1], [4, -1, 0.1],  # Boxes
            [5, 0, 0.2], [6, 0.5, 0.15],  # More obstacles
        ]
        
        for i, pos in enumerate(positions[:int(complexity * len(positions))]):
            obstacle = Cube(
                prim_path=f"/World/ObstacleCourse{i}",
                name=f"oc_{i}",
                position=np.array(pos),
                size=0.2,
                color=np.array([0.7, 0.2, 0.2])
            )
            self.scene.add(obstacle)
            obstacles.append(obstacle)
        
        return obstacles

    def get_humanoid_state(self):
        """
        Get comprehensive state of the humanoid robot
        """
        if not self.humanoids:
            return None
            
        robot = self.humanoids[0]  # Assume single robot for this example
        
        # Get joint states
        joint_states = robot.get_joint_states()
        
        # Get IMU data
        imu_data = robot.get_imu_data()
        
        # Get robot pose and velocity
        world_pos, world_orn = robot.get_world_poses()
        world_lin_vel, world_ang_vel = robot.get_velocities()
        
        # Calculate center of mass information
        com_pos, com_vel = self._calculate_center_of_mass(robot)
        
        # Get contact information (simplified)
        contact_info = self._get_contact_information(robot)
        
        state_vector = np.concatenate([
            joint_states['positions'],
            joint_states['velocities'],
            imu_data['orientation'],  # 4 values (quaternion)
            imu_data['angular_velocity'],  # 3 values
            world_lin_vel,  # 3 values (global velocity)
            com_pos,  # 3 values (CoM position relative to world)
            com_vel,  # 3 values (CoM velocity)
            contact_info,  # 2 values (left and right foot contact)
        ])
        
        return state_vector
    
    def _calculate_center_of_mass(self, robot):
        """
        Calculate center of mass position and velocity (simplified)
        """
        # In a real implementation, this would calculate actual CoM
        # For now, we'll use the robot's root position as an approximation
        world_pos, _ = robot.get_world_poses()
        
        # CoM velocity approximation
        world_lin_vel, _ = robot.get_velocities()
        
        return world_pos, world_lin_vel
    
    def _get_contact_information(self, robot):
        """
        Get contact information for feet (simplified)
        """
        # In a real implementation, this would check contact sensors
        # For now, return dummy values
        # [left_foot_contact, right_foot_contact]
        return np.array([0.0, 0.0])  # No contact initially

    def apply_action(self, action):
        """
        Apply action to the humanoid robot
        """
        if not self.humanoids:
            return
            
        robot = self.humanoids[0]
        
        # Action could be joint positions, velocities, or torques
        # For this example, we'll treat it as desired joint positions
        joint_targets = action
        
        robot.set_joint_position_targets(joint_targets)
        
        # Optionally add noise to simulate real actuator behavior
        if hasattr(self, 'action_noise') and self.action_noise > 0:
            noisy_action = joint_targets + np.random.normal(0, self.action_noise, joint_targets.shape)
            robot.set_joint_position_targets(noisy_action)
    
    def get_reward(self, action):
        """
        Calculate reward based on current state and action
        """
        if not self.humanoids:
            return 0.0
            
        robot = self.humanoids[0]
        
        # Get current state
        state = self.get_humanoid_state()
        
        # Parse state components
        joint_positions = state[:20]  # First 20 are joint positions (example)
        joint_velocities = state[20:40]  # Next 20 are joint velocities
        orientation = state[40:44]  # Next 4 are orientation (quaternion)
        global_velocity = state[44:47]  # Next 3 are global velocity
        com_position = state[47:50]  # Next 3 are CoM position
        
        # Calculate different reward components
        forward_progress_reward = global_velocity[0] * 10.0  # Encourage forward movement
        
        balance_reward = self._calculate_balance_reward(orientation, com_position)
        
        energy_penalty = -0.01 * np.sum(np.square(action))  # Penalize high effort
        
        # Survival bonus
        survival_bonus = 1.0  # Small bonus for staying alive
        
        # Combined reward
        total_reward = (
            forward_progress_reward * 0.5 +
            balance_reward * 2.0 +
            energy_penalty * 1.0 +
            survival_bonus * 0.1
        )
        
        return total_reward
    
    def _calculate_balance_reward(self, orientation, com_position):
        """
        Calculate balance-based reward component
        """
        # Ensure orientation is normalized quaternion
        quat = orientation / (np.linalg.norm(orientation) + 1e-8)
        
        # Calculate "up" vector in world frame based on orientation
        # For quaternion [x,y,z,w], transform (0,0,1) vector
        z_axis = self._rotate_vector_by_quaternion(np.array([0, 0, 1]), quat)
        
        # Closer to upright (z=0,0,1) is better
        # Perfect upright would give z_axis[2] = 1.0
        # If upside down z_axis[2] = -1.0
        upright_alignment = z_axis[2]  # Value between -1 and 1
        
        # Also consider CoM position relative to feet
        # For now, a simple penalty if CoM is getting too far from center
        com_xy_deviation = np.linalg.norm(com_position[:2])  # Distance from (0,0) in XY plane
        max_acceptable_deviation = 0.3  # 30cm from center
        com_penalty = max(0, (com_xy_deviation - max_acceptable_deviation) * -5.0)
        
        # Combine both components
        balance_score = upright_alignment * 2.0 + com_penalty  # Scale upright contribution
        return np.clip(balance_score, -5.0, 5.0)  # Limit reward magnitude
    
    def _rotate_vector_by_quaternion(self, vector, quaternion):
        """
        Rotate a vector by a quaternion
        """
        # Convert quaternion wxyz to xyzw format (Isaac Sim convention)
        q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        
        # Rotate using quaternion formula
        v = np.array([vector[0], vector[1], vector[2], 0.0])  # Homogeneous coordinates
        
        # Quaternion multiplication: q * v * q_conjugate
        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
            z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
            return np.array([w, x, y, z])
        
        # Rotate vector: q * v * q*
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])  # Conjugate
        rotated = quat_mult(quat_mult(q, v), q_conj)
        return rotated[:3]

    def is_done(self):
        """
        Check if episode is done (e.g., robot fell)
        """
        if not self.humanoids:
            return True
            
        robot = self.humanoids[0]
        
        # Get robot state
        world_pos, world_orn = robot.get_world_poses()
        
        # Check if robot has fallen (low height)
        height_threshold = 0.5  # If base drops below 50cm, consider fallen
        if world_pos[2] < height_threshold:
            return True
        
        # Check orientation (if excessively tilted)
        quat = world_orn / (np.linalg.norm(world_orn) + 1e-8)
        z_axis = self._rotate_vector_by_quaternion(np.array([0, 0, 1]), quat)
        
        # If not mostly upright (less than 60 degrees from upright)
        if z_axis[2] < 0.5:  # cos(60°) ≈ 0.5
            return True
        
        return False

# RL Environment wrapper for the humanoid world
class HumanoidRLEnv:
    """
    Reinforcement Learning environment wrapper for humanoid training
    """
    def __init__(self, config_path=None):
        self.world = HumanoidWorld()
        
        # Load configuration
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'action_space': 20,  # 20 joints
                'observation_space': 122,  # Example dimension
                'max_episode_length': 1000,
                'action_std': 0.1,
                'reward_weights': {
                    'forward_progress': 1.0,
                    'balance': 2.0,
                    'energy_efficiency': -0.01
                }
            }
        
        # Episode tracking
        self.current_step = 0
        self.episode_num = 0
        self.episode_reward = 0
        
    def reset(self):
        """
        Reset the environment to initial state
        """
        # Reset the Isaac Sim world
        self.world.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0
        
        # Return initial observation
        return self.get_observation()
    
    def step(self, action):
        """
        Take a step in the environment
        """
        # Apply action to robot
        self.world.apply_action(action)
        
        # Step the physics simulation
        self.world.step(render=True)
        
        # Get next observation
        obs = self.get_observation()
        
        # Calculate reward
        reward = self.world.get_reward(action)
        self.episode_reward += reward
        
        # Check if episode is done
        done = self.world.is_done() or self.current_step >= self.config['max_episode_length']
        
        # Update step counter
        self.current_step += 1
        
        # Info dictionary for additional debugging information
        info = {
            'episode_num': self.episode_num,
            'step_num': self.current_step,
            'episode_reward': self.episode_reward,
            'done_reason': 'fallen' if self.world.is_done() else ('max_steps' if self.current_step >= self.config['max_episode_length'] else 'continuing')
        }
        
        return obs, reward, done, info
    
    def get_observation(self):
        """
        Get current observation from the environment
        """
        return self.world.get_humanoid_state()
    
    def close(self):
        """
        Close the environment
        """
        self.world.clear()