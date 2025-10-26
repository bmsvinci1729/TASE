import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


NUM_CONTROLLABLE_JOINTS = 14 # 14 controllable joints in humanoid_env (p.JOINT_REVOLUTE + p.JOINT_PRISMATIC)
NUM_TORQUE_BINS = 5 # 5 bins taken cause its ther in 3.2


INPUT_DIM = 4 + 2 * NUM_CONTROLLABLE_JOINTS # robot torsso's orientation(4 dimension) + joint's orientation(1 for angle and 1 for rotational speed)

OUTPUT_DIM = NUM_CONTROLLABLE_JOINTS * NUM_TORQUE_BINS # Action vector size: 14 * 5 (Joints * Bins) = 70


TORQUE_BINS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
HIDDEN_SIZE = 512 # asked 256 - 512 neurons

class QNetwork(nn.Module):
    """
    The Deep Q-Network (DQN) architecture (MLP) for the Humanoid Locomotion task.

    This network approximates the Q-function: Q(s, a).
    It takes the state vector (s) and outputs the Q-value for all 70 possible discrete actions.
    """
    def __init__(self):
        super(QNetwork, self).__init__()

        # --- Layer Definitions ---
        # Input: 32 (State vector)
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_SIZE) # Input Layer - (32 input -> 512 neurons)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) # Hidden layer - (512 -> 512)
        # Output: 70 (Total discrete actions: 14 joints * 5 bins)
        self.fc3 = nn.Linear(HIDDEN_SIZE, OUTPUT_DIM) #final output

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Processes the state tensor to output Q-values.
        """
        # Ensure input is float and process through ReLU activations
        x = state.float()
        x = F.relu(self.fc1(x)) # ReLU after every layer
        x = F.relu(self.fc2(x)) 

        # The final output layer is linear, representing the Q-values.
        q_values = self.fc3(x)
        return q_values

def select_action(q_values_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts the network's Q-value output into a 14-element torque vector 
    using the argmax (greedy) selection strategy for each joint.
    
    This fulfills Task 3.2: Action Space Discretization.

    Args:
        q_values_tensor: The tensor output from the QNetwork (size 70).

    Returns:
        np.ndarray: A 1D array of 14 torque values (between -1.0 and 1.0).
    """
    # 1. Prepare data: Detach from graph, move to CPU, and convert to NumPy
    q_values = q_values_tensor.cpu().detach().numpy()
    
    # 2. Reshape to treat the output as 14 separate "heads"
    # Shape changes from [70] to [14 joints, 5 bins]
    q_values_reshaped = q_values.reshape(NUM_CONTROLLABLE_JOINTS, NUM_TORQUE_BINS)
    
    # 3. Find the index (bin) with the highest Q-value for each joint
    # Argmax is performed across the bin dimension (axis=1)
    action_indices = np.argmax(q_values_reshaped, axis=1)
    
    # 4. Map the chosen index (0-4) back to the actual torque value in the bin array
    action_torques = TORQUE_BINS[action_indices]
    
    # action_torques is a 1D NumPy array of size 14, ready for env.step()
    return action_torques

# -------------------------------------------------------------------
# Example Usage Block (for testing, not used in main)
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Initialize the network
    net = QNetwork()
    
    # Create a dummy state (one sample)
    dummy_state = torch.rand(1, INPUT_DIM)
    
    # Perform a forward pass
    q_output = net(dummy_state)
    
    print(f"Input State Dimension: {dummy_state.shape}")
    print(f"QNetwork Output Dimension: {q_output.shape}") # Should be [1, 70]
    
    # Select the action based on the Q-values
    action_vector = select_action(q_output)
    
    print(f"\nFinal Action Vector (Torques): {action_vector}") # Should be 14 elements
    print(f"Action Vector Dimension: {action_vector.shape}") # Should be (14,)
    
    # Verify all torques are one of the defined bins
    is_valid = np.all(np.isin(action_vector, TORQUE_BINS))
    print(f"All selected torques are valid bins: {is_valid}")
    
