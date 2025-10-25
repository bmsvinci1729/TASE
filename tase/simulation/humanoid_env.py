# tase/simulation/humanoid_env.py

import os
import time
import pybullet as p
import pybullet_data
import gymnasium as gym
import numpy as np


class HumanoidWalkEnv(gym.Env):
    """
    A custom Gymnasium environment for a humanoid agent learning to walk,
    initialized from a specific pose.
    """

    def __init__(self, urdf_path, render_mode='human'):
        """
        Initializes the simulation environment.

        Args:
            urdf_path (str): Path to the humanoid URDF file.
            render_mode (str): 'human' for GUI, 'rgb_array' for non-graphical.
        """
        # based on render mode, u connect to the physics servers
        self.render_mode = render_mode

        # Connect to the PyBullet physics server
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Add pybullet_data to the search path for loading standard assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # humanoid.urdf is in pybullet_data

        # Load the ground plane and set gravity
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf") # p already has the data path set, so we can access plane.urdf directly

        # Load the humanoid model
        self.urdf_path = urdf_path # the one in the assets folder, we have set up pipeline in the main.py
        start_pos = [0, 0, 1.5]  # avoiding ground contact initially
        self.humanoid_id = p.loadURDF(self.urdf_path, start_pos) # loading the humanoid urdf

        # Get joint information from the loaded URDF
        self._get_joint_info() # this function is defined below

        # Map perception module joint names to PyBullet joint indices
        # after parsing list and dict are filled, and so: see below
        self.angle_name_to_joint_index = {
            'left_knee': self.joint_name_to_id.get('left_knee'), 
            'right_knee': self.joint_name_to_id.get('right_knee'),
            'left_elbow': self.joint_name_to_id.get('left_elbow'),
            'right_elbow': self.joint_name_to_id.get('right_elbow'),
        }

        # --- Define Observation and Action Spaces ---
        obs_space_dim = 4 + 2 * len(self.controllable_joint_indices) # 4 for base orientation (quaternion) + 2 per joint (position + velocity)
        # discrete observation space for the humanoid env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32
        )

        # Action: torque applied to each controllable joint
        action_space_dim = len(self.controllable_joint_indices)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_space_dim,), dtype=np.float32
        )
#         If your robot has 6 controllable joints → action vector = [a1, a2, a3, a4, a5, a6] Each a_i is between -1.0 and 1.0

        # Store max torque from URDF for scaling actions (fallback = 100)
        self.max_torques = [self.joint_max_force.get(i, 100.0) for i in self.controllable_joint_indices]
        # so, action is btw -1 and 1, we scale it m multiplying ti with the max torque for that joint in the step function

    # --------------------------------------------------------------------------
    def _get_joint_info(self):
        # all we do here is parse urdf file, get joint info -  name, id, type, controllable or not, max force/torque    
        """
        Parses the URDF file to get information about the joints.
        This helps us identify which joints are controllable and their limits.
        """
        # what are these dict and list for tell for each ?
        self.joint_name_to_id = {}  # Maps joint names to their IDs
        self.controllable_joint_indices = []  # List of joint indices that can be controlled
        self.joint_max_force = {}  # Maps joint IDs to their maximum force/torque

        num_joints = p.getNumJoints(self.humanoid_id)
        print("No. of joints in humanoid URDF:", num_joints) # 14
        print("\n--- Humanoid Joint Information ---")
        for i in range(num_joints):
            info = p.getJointInfo(self.humanoid_id, i) # is this from the api doc ? yes
            # info is a tuple with joint details
            # print("Joint Info:", info) output below
# Joint Info: (5, b'left_shoulder_x', 0, 12, 11, 1, 0.0, 0.0, -1.57, 1.57, 50.0, 5.0, b'left_upper_arm', (1.0, 0.0, 0.0), (0.0, 0.2, 0.10999999999999999), (0.0, 0.0, 0.0, 1.0), 0)
# Joint 'left_shoulder_x' (ID: 5) is controllable.
            joint_id = info[0]
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]

            self.joint_name_to_id[joint_name] = joint_id

            # Try to read max force/torque; use fallback if not available
            try:
                max_force = float(info[10]) if len(info) > 10 else 100.0
            except Exception:
                max_force = 100.0

            self.joint_max_force[joint_id] = max_force

            # Only revolute or prismatic joints are controllable
            # what are these joint types ?
            # like examples: names of joints ? left_shoulder_x, left_shoulder_y, left_shoulder_z, left_elbow, right_shoulder_x, right_shoulder_y, right_shoulder_z, right_elbow
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joint_indices.append(joint_id)
                print(f"Joint '{joint_name}' (ID: {joint_id}) is controllable.")

        print(f"Found {len(self.controllable_joint_indices)} controllable joints.")
        print("--------------------------------\n")

    # --------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """
        Resets the environment. If an initial pose is provided in 'options',
        it sets the humanoid's joints to those angles.
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset simulation and reload environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        start_pos = [0, 0, 1.5]
        self.humanoid_id = p.loadURDF(self.urdf_path, start_pos)

        # Re-fetch joint info after reload
        self._get_joint_info()

        # Apply initial pose if provided
        if options and 'initial_pose_angles' in options:
            initial_angles = options['initial_pose_angles'] # where do we get the initial pose angles dict from ? from main.py where there ? 
            # we get the initial pose angles dict from the main.py file, specifically from the 'initial_pose_angles' key in the options dictionary
            print("Setting initial pose from provided angles...")
            # for each joint angle name in the initial angles dict, set the joint state in pybullet
            for angle_name, angle_value in initial_angles.items():
                if angle_name in self.angle_name_to_joint_index and self.angle_name_to_joint_index[angle_name] is not None:
                    joint_index = self.angle_name_to_joint_index[angle_name]
                    p.resetJointState(
                        bodyUniqueId=self.humanoid_id,
                        jointIndex=joint_index,
                        targetValue=float(angle_value)
                    )
                    print(f"  - Set joint '{angle_name}' to {np.degrees(angle_value):.2f}°.")
                else:
                    print(f"  - Warning: Joint '{angle_name}' not found in mapping.")
        else:
            print("No initial pose provided. Using default URDF pose.")

        observation = self._get_observation()
        info = {}
        return observation, info

    # --------------------------------------------------------------------------
    def _get_observation(self):
        """
        Gathers the current state of the simulation to form an observation vector.
        """
        _, base_orientation = p.getBasePositionAndOrientation(self.humanoid_id)

        joint_states = p.getJointStates(self.humanoid_id, self.controllable_joint_indices)
        joint_positions = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_velocities = np.array([s[1] for s in joint_states], dtype=np.float32)

        observation = np.concatenate([np.array(base_orientation, dtype=np.float32),
                                      joint_positions, joint_velocities])
        return observation.astype(np.float32) # base orientation, joint positions, joint velocities together form the observation

    # --------------------------------------------------------------------------
    def step(self, action):
        """
        Applies an action, steps the simulation, and returns the results.
        """
        action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        scaled_torques = action * np.array(self.max_torques, dtype=np.float32)

        p.setJointMotorControlArray(
            bodyUniqueId=self.humanoid_id,
            jointIndices=self.controllable_joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=scaled_torques.tolist()
        )

        p.stepSimulation()

        observation = self._get_observation()

        # Simple reward: alive bonus
        reward = 0.1
        torso_pos, _ = p.getBasePositionAndOrientation(self.humanoid_id)
        terminated = torso_pos[2] < 0.8  # fell if too low
        if terminated:
            reward = -100.0

        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    # --------------------------------------------------------------------------
    def close(self):
        """
        Cleans up the environment by disconnecting from the physics server.
        """
        try:
            p.disconnect(self.physics_client)
        except Exception:
            pass


# ------------------------------------------------------------------------------
# TEST BLOCK
# ------------------------------------------------------------------------------
# (In tase/simulation/humanoid_env.py)

# (In tase/simulation/humanoid_env.py)

    def __init__(self, urdf_path, render_mode='human'):
        self.render_mode = render_mode
        
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        
        self.urdf_path = urdf_path
        start_pos = [0, 0, 1.5]
        self.humanoid_id = p.loadURDF(self.urdf_path, start_pos)

        self._get_joint_info()
        
        # --- FINAL, CORRECTED JOINT MAPPING for humanoid_v2.urdf ---
        self.angle_name_to_joint_index = {
            'left_knee': self.joint_name_to_id.get('left_knee'),
            'right_knee': self.joint_name_to_id.get('right_knee'),
            'left_elbow': self.joint_name_to_id.get('left_elbow'),
            'right_elbow': self.joint_name_to_id.get('right_elbow'),
            
            # Roll is rotation around the local X-axis in this URDF
            'left_shoulder_roll': self.joint_name_to_id.get('left_shoulder_x'),
            'right_shoulder_roll': self.joint_name_to_id.get('right_shoulder_x'),
            'left_hip_roll': self.joint_name_to_id.get('left_hip_x'),
            'right_hip_roll': self.joint_name_to_id.get('right_hip_x'),

            # Pitch is rotation around the local Y-axis in this URDF
            'left_shoulder_pitch': self.joint_name_to_id.get('left_shoulder_y'),
            'right_shoulder_pitch': self.joint_name_to_id.get('right_shoulder_y'),
            'left_hip_pitch': self.joint_name_to_id.get('left_hip_y'),
            'right_hip_pitch': self.joint_name_to_id.get('right_hip_y'),
        }
        self.angle_name_to_joint_index = {k: v for k, v in self.angle_name_to_joint_index.items() if v is not None}

        obs_space_dim = 4 + 2 * len(self.controllable_joint_indices)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32
        )
        action_space_dim = len(self.controllable_joint_indices)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_space_dim,), dtype=np.float32
        )
        self.max_torques = [self.joint_max_force[i] for i in self.controllable_joint_indices]