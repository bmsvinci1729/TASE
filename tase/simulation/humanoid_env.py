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
            # Map shoulder/hip multi-axis joints from perception to URDF joint names
            'left_shoulder_roll': self.joint_name_to_id.get('left_shoulder_x'),
            'left_shoulder_pitch': self.joint_name_to_id.get('left_shoulder_y'),
            'right_shoulder_roll': self.joint_name_to_id.get('right_shoulder_x'),
            'right_shoulder_pitch': self.joint_name_to_id.get('right_shoulder_y'),
            'left_hip_roll': self.joint_name_to_id.get('left_hip_x'),
            'left_hip_pitch': self.joint_name_to_id.get('left_hip_y'),
            'right_hip_roll': self.joint_name_to_id.get('right_hip_x'),
            'right_hip_pitch': self.joint_name_to_id.get('right_hip_y'),
        }
        # Filter out None values (in case some joints don't exist)
        self.angle_name_to_joint_index = {k: v for k, v in self.angle_name_to_joint_index.items() if v is not None}

        # --- Define Observation and Action Spaces ---
        obs_space_dim = 4 + 2 * len(self.controllable_joint_indices) # 4 for base orientation (quaternion) + 2 per joint (position + velocity)
        # can u tell what is this dimension representing ?
        # Yes, the observation space dimension represents the state of the humanoid in the simulation.
        # It includes the orientation of the base (as a quaternion) and the positions and velocities of all controllable joints.
        # discrete observation space for the humanoid env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_dim,), dtype=np.float32
        )
        # give an example of obs space dim if there are 6 controllable joints
        # If there are 6 controllable joints, the observation space dimension would be:
        # 4 (base orientation) + 2 * 6 (joint positions and velocities)
        # = 4 + 12 = 16
        # So the observation space would be a Box with shape (16,)
        
        # Action: torque applied to each controllable joint
        action_space_dim = len(self.controllable_joint_indices)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_space_dim,), dtype=np.float32
        )
#         If your robot has 6 controllable joints → action vector = [a1, a2, a3, a4, a5, a6] Each a_i is between -1.0 and 1.0

        # Store max torque from URDF for scaling actions (fallback = 100)
        self.max_torques = [self.joint_max_force.get(i, 100.0) for i in self.controllable_joint_indices]
        # so, action is btw -1 and 1, we scale it m multiplying ti with the max torque for that joint in the step function
        # so ideally action beteeen -1 and 1 what does that mean physically
        # It means applying torque in the negative or positive direction up to the maximum torque limit of the joint.
    # --------------------------------------------------------------------------
    def _get_joint_info(self):
        # all we do here is parse urdf file, get joint info -  name, id, type, controllable or not, max force/torque    
        # how do we know if a joint is controllable or not ? by its type - revolute or prismatic
        # revolute and prismatic joints are controllable, fixed joints are not
        # why ? because fixed joints do not allow movement, so no point in controlling them
        # WHERE IS THE STANDARD INFO OF JOINTS RETRIEVED FROM ? from the pybullet api p.getJointInfo function
        """
        Parses the URDF file to get information about the joints.
        This helps us identify which joints are controllable and their limits.
        """
        # what are these dict and list for tell for each ?
        self.joint_name_to_id = {}  # Maps joint names to their IDs
        self.controllable_joint_indices = []  # List of joint indices that can be controlled
        self.joint_max_force = {}  # Maps joint IDs to their maximum force/torque

        num_joints = p.getNumJoints(self.humanoid_id)
        print("[LOG] No. of joints in humanoid URDF:", num_joints) # 14
        print("\n[LOG] --- Humanoid Joint Information ---")
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
            # what joints are not controllable ?
            if joint_type == p.JOINT_FIXED:
                print(f"[LOG] Joint '{joint_name}' (ID: {joint_id}) is NOT controllable.")
            

            # Only revolute or prismatic joints are controllable
            # what are these joint types ?
            # like examples: names of joints ? left_shoulder_x, left_shoulder_y, left_shoulder_z, left_elbow, right_shoulder_x, right_shoulder_y, right_shoulder_z, right_elbow
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joint_indices.append(joint_id)
                print(f"[LOG] Joint '{joint_name}' (ID: {joint_id}) is controllable.")

        print(f"[LOG] Found {len(self.controllable_joint_indices)} controllable joints.")
        print("[LOG] ------------------------------------------------\n")

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
        
        # Rebuild angle mapping after joint info is updated
        self.angle_name_to_joint_index = {
            'left_knee': self.joint_name_to_id.get('left_knee'), 
            'right_knee': self.joint_name_to_id.get('right_knee'),
            'left_elbow': self.joint_name_to_id.get('left_elbow'),
            'right_elbow': self.joint_name_to_id.get('right_elbow'),
            # Map shoulder/hip multi-axis joints from perception to URDF joint names
            'left_shoulder_roll': self.joint_name_to_id.get('left_shoulder_x'),
            'left_shoulder_pitch': self.joint_name_to_id.get('left_shoulder_y'),
            'right_shoulder_roll': self.joint_name_to_id.get('right_shoulder_x'),
            'right_shoulder_pitch': self.joint_name_to_id.get('right_shoulder_y'),
            'left_hip_roll': self.joint_name_to_id.get('left_hip_x'),
            'left_hip_pitch': self.joint_name_to_id.get('left_hip_y'),
            'right_hip_roll': self.joint_name_to_id.get('right_hip_x'),
            'right_hip_pitch': self.joint_name_to_id.get('right_hip_y'),
        }
        # Filter out None values
        self.angle_name_to_joint_index = {k: v for k, v in self.angle_name_to_joint_index.items() if v is not None}

        # Apply initial pose if provided
        if options and 'initial_pose_angles' in options:
            initial_angles = options['initial_pose_angles'] # where do we get the initial pose angles dict from ? from main.py where there ? 
            # we get the initial pose angles dict from the main.py file, specifically from the 'initial_pose_angles' key in the options dictionary
            print("[LOG] Setting initial pose from provided angles...")
            # for each joint angle name in the initial angles dict, set the joint state in pybullet
            for angle_name, angle_value in initial_angles.items():
                if angle_name in self.angle_name_to_joint_index and self.angle_name_to_joint_index[angle_name] is not None:
                    joint_index = self.angle_name_to_joint_index[angle_name]
                    p.resetJointState(
                        bodyUniqueId=self.humanoid_id,
                        jointIndex=joint_index,
                        targetValue=float(angle_value)
                    )
                    print(f"[LOG]   - Set joint '{angle_name}' to {np.degrees(angle_value):.2f}°.")
                else:
                    print(f"[WARNING]   - Joint '{angle_name}' not found in mapping.")
        else:
            print("[LOG] No initial pose provided. Using default URDF pose.")

        observation = self._get_observation()
        # why this ?
        # base orientation, joint positions, joint velocities together form the observation
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
        This implements the full Task 3.3 Reward Function.
        """
        # 1. Apply Action
        # Clip action to be in [-1, 1] range
        action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        # Scale normalized action to the joint's max torque
        scaled_torques = action * np.array(self.max_torques, dtype=np.float32)

        p.setJointMotorControlArray(
            bodyUniqueId=self.humanoid_id,
            jointIndices=self.controllable_joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=scaled_torques.tolist()
        )

        p.stepSimulation()

        # 2. Get New Observation
        observation = self._get_observation()
        
        # 3. Calculate Reward (Task 3.3) 
        torso_pos, torso_quat = p.getBasePositionAndOrientation(self.humanoid_id)
        torso_z = torso_pos[2]

        # --- Reward Components ---
        
        # r_vel: Forward Velocity Reward 
        # Get world-frame linear velocity 
        torso_vel_world, _ = p.getBaseVelocity(self.humanoid_id)
        # Get rotation matrix from quaternion to find local forward vector 
        rot_matrix = p.getMatrixFromQuaternion(torso_quat)
        local_forward_vec = [rot_matrix[0], rot_matrix[3], rot_matrix[6]] # Local X-axis
        
        # We only care about forward velocity in the X-Y plane
        world_vel_vec_2d = [torso_vel_world[0], torso_vel_world[1]]
        local_forward_vec_2d = [local_forward_vec[0], local_forward_vec[1]]
        
        # Dot product of 2D velocity and 2D forward vector
        r_vel = np.dot(world_vel_vec_2d, local_forward_vec_2d)
        
        # r_live: Alive Bonus 
        r_live = 0.1

        # T_energy: Energy/Torque Penalty 
        # In TORQUE_CONTROL, applied torque is the 'forces' param
        t_energy = np.sum(np.square(scaled_torques))

        # Stability Penalty (to discourage wobbling)
        roll, pitch, _ = p.getEulerFromQuaternion(torso_quat)
        r_stability_penalty = (roll**2 + pitch**2)

        # --- Define Weights (These are hyperparameters you can tune) ---
        W_VEL = 1.0       # Primary objective
        W_LIVE = 0.1      # Small incentive to stay up
        W_ENERGY = 0.005  # Small penalty for high torque
        W_STABILITY = 0.05 # Small penalty for wobbling
        
        reward = (W_VEL * r_vel) + \
                 (W_LIVE * r_live) - \
                 (W_ENERGY * t_energy) - \
                 (W_STABILITY * r_stability_penalty)

        # 4. Check Termination
        terminated = False
        if torso_z < 0.8: # Fell over
            terminated = True
            reward = -10.0 # Large fall penalty 

        truncated = False # We can use this later for a time limit
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

