import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# Import the network definition and constants from the sibling file
from .q_network import QNetwork, INPUT_DIM, OUTPUT_DIM, NUM_CONTROLLABLE_JOINTS, NUM_TORQUE_BINS, TORQUE_BINS

# --- Hyperparameters ---
BUFFER_SIZE = 100000     # Replay buffer size
BATCH_SIZE = 128         # Minibatch size
GAMMA = 0.99           # Discount factor
TAU = 1e-3             # For soft target network update
LR = 5e-4              # Learning rate
UPDATE_EVERY = 4       # How often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience", field_names=["state", "action_indices", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def store(self, state, action_indices, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action_indices, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # Note: action_indices are stored, not the torques
        action_indices = torch.from_numpy(np.vstack([e.action_indices for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, action_indices, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, seed):
        """Initialize an Agent object."""
        self.state_dim = INPUT_DIM
        self.action_dim = OUTPUT_DIM
        self.seed = random.seed(seed)

        # Q-Network (Task 3.1) 
        self.policy_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def select_action_epsilon_greedy(self, state, epsilon=0.):
        """
        Returns actions for given state as per current policy.
        This implements the multi-head exploration strategy.
        
        Returns:
            action_indices (np.array[int]): The 14 chosen indices (0-4).
            action_torques (np.array[float]): The 14 torque values (-1.0 to 1.0).
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.policy_net.eval()
        with torch.no_grad():
            # Get the Q-values for all 70 actions
            q_values_all_70 = self.policy_net(state_tensor)
        self.policy_net.train()

        # Epsilon-greedy action selection (Task 3.2 logic) 
        if random.random() > epsilon:
            # --- EXPLOIT ---
            # Reshape from 70 to [14, 5]
            q_values_reshaped = q_values_all_70.cpu().data.numpy().reshape(NUM_CONTROLLABLE_JOINTS, NUM_TORQUE_BINS)
            # Find the best bin index for each of the 14 joints
            action_indices = np.argmax(q_values_reshaped, axis=1)
        else:
            # --- EXPLORE ---
            # Select a random bin index (0-4) for each of the 14 joints
            action_indices = np.random.randint(0, NUM_TORQUE_BINS, size=NUM_CONTROLLABLE_JOINTS)
        
        # Map indices to actual torque values from the TORQUE_BINS array
        action_torques = TORQUE_BINS[action_indices]
        
        return action_indices, action_torques
    
    def step(self, state, action_indices, reward, next_state, done):
        """Save experience in replay memory, and use random sample to learn."""
        # Save experience
        self.memory.store(state, action_indices, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        This function implements the multi-head Q-value summation.
        """
        states, action_indices_batch, rewards, next_states, dones = experiences

        # --- 1. Compute Target Q-Values (y_j) ---
        
        # Get Q-values for next states from target network
        Q_targets_next_all_70 = self.target_net(next_states) # Shape: [batch_size, 70]
        
        # Reshape to [batch_size, 14, 5] to find max for each of the 14 heads 
        Q_targets_next_reshaped = Q_targets_next_all_70.view(-1, NUM_CONTROLLABLE_JOINTS, NUM_TORQUE_BINS)
        
        # Find max Q-value for each head (dim=2): Shape becomes [batch_size, 14]
        max_q_per_head, _ = Q_targets_next_reshaped.max(dim=2)
        
        # Sum the max Q-values from all 14 heads: Shape becomes [batch_size, 1]
        summed_max_q = max_q_per_head.sum(dim=1, keepdim=True)

        # Compute Q targets for current states: r + γ * sum(max_a'(Q_target(s', a')))
        Q_targets = rewards + (GAMMA * summed_max_q * (1 - dones))

        # --- 2. Compute Predicted Q-Values (Q(s, a)) ---
        
        # Get Q-values for current states from policy network
        Q_expected_all_70 = self.policy_net(states) # Shape: [batch_size, 70]
        
        # Reshape to [batch_size, 14, 5]
        Q_expected_reshaped = Q_expected_all_70.view(-1, NUM_CONTROLLABLE_JOINTS, NUM_TORQUE_BINS)
        
        # We need to select the Q-value for the action *actually taken*
        # action_indices_batch is [batch_size, 14]. Need to expand for gather.
        # Shape becomes [batch_size, 14, 1]
        action_indices_expanded = action_indices_batch.unsqueeze(2) 

        # Use gather to pick the Q-value corresponding to the action index
        # for each of the 14 heads. Shape: [batch_size, 14, 1]
        predicted_q_per_head = Q_expected_reshaped.gather(2, action_indices_expanded)
        
        # Sum the Q-values from all heads. Shape: [batch_size, 1]
        Q_expected = predicted_q_per_head.sum(dim=1)

        # --- 3. Compute Loss ---
        # Use Huber loss (F.smooth_l1_loss) for more stable training [3, 4]
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # --- 4. Optimize the model ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- 5. Update target network (soft update) ---
        self.soft_update(self.policy_net, self.target_net)

    def soft_update(self, policy_net, target_net):
        """Soft update model parameters.
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        """
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
    # --- We will add the learning methods (step, learn) in the next batch ---