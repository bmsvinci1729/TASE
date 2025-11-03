import numpy as np


# -------------------------------------------------------------------
# Lightweight replacement for MediaPipe's deprecated PoseLandmark enum
# -------------------------------------------------------------------
class PoseLandmark:
    """Minimal replacement for MediaPipe's old PoseLandmark enum."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# -------------------------------------------------------------------
# Joint-to-landmark mapping
# -------------------------------------------------------------------
JOINT_LANDMARK_MAP = { # indices of landmarks for each joint
    'left_elbow': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    'right_elbow': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    'left_knee': (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    'right_knee': (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    'left_shoulder': (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    'right_shoulder': (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    'left_hip': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    'right_hip': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
}


# -------------------------------------------------------------------
# Reference T-pose limb direction vectors (URDF coordinate system)
# -------------------------------------------------------------------
T_POSE_VECTORS = {
    'left_shoulder': np.array([0.0, 1.0, 0.0]),    # Arm points left
    # what do the numberes mean ?
    # These numbers represent the direction vector of the limb in the URDF coordinate system.
    # u mean x, y, z unit vectors ? yes
    # so what does this mean for the arm direction ? X- 
    'right_shoulder': np.array([0.0, -1.0, 0.0]),  # Arm points right
    'left_hip': np.array([0.0, 0.0, -1.0]),        # Leg points down
    'right_hip': np.array([0.0, 0.0, -1.0]),       # Leg points down
}


# -------------------------------------------------------------------
# Convert landmarks → dictionary (with URDF coordinate transform)
# -------------------------------------------------------------------
def landmarks_to_dict(pose_world_landmarks):
    """
    Converts MediaPipe landmarks into a dictionary of 3-D NumPy arrays.
    Also transforms coordinates from MediaPipe → URDF frame.
    """
    if not pose_world_landmarks:
        return None

    landmark_dict = {}
    for i, lm in enumerate(pose_world_landmarks):
        #what does the pose_world_landmarks contain ?
        # The pose_world_landmarks contain 3D coordinates of key points on the human body. like 33 a ? yes
        # MediaPipe: +X right, +Y down, +Z toward camera
        # URDF:      +X forward, +Y left, +Z up
        # urdf to mediapipe transformation
        
        transformed = np.array([
            -lm.z,   # forward
            -lm.x,   # left
            -lm.y    # up
        ])
        landmark_dict[i] = transformed

    return landmark_dict


# -------------------------------------------------------------------
# Angle calculation helper
# -------------------------------------------------------------------
def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle (radians) formed by three 3D points at vertex p2."""
    v1, v2 = p1 - p2, p3 - p2 # why this way ?
    # This is done to create two vectors (v1 and v2) that share a common point (p2).
    # is p1 a single point if so then why is it a np array
    # Yes, p1 is a single point in 3D space, and it's represented as a NumPy array for easier mathematical operations.
    dot = np.dot(v1, v2)
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    # arccos calculation from dot product formula
    if mag1 == 0 or mag2 == 0:
        return 0.0 # implies immovable joint ? yes or no
    # and what does this physically mean ?
    # It means that the joint is not able to move in the 3D space defined by the points.
    cosine = np.clip(dot / (mag1 * mag2), -1.0, 1.0) # clipping results between -1 and 1 but isnt that obvious from the dot product formula ?
    # Yes, it's a common step in numerical computing to avoid invalid values in arccos.
    return np.arccos(cosine) # great


# -------------------------------------------------------------------
# Rotation matrix aligning vec1 → vec2
# -------------------------------------------------------------------
def get_rotation_matrix(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Computes the 3×3 rotation matrix that aligns vec1 to vec2."""
    # unit vectors
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    # print("VVVVVVVVVVVVVVV", v)
# what are the 3 elements in v representing decompose what is v[0], v[1], v[2] ?
# The elements of vector v represent the components of the axis of rotation in 3D space
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0: # parallel vectors exactly...
        return np.identity(3) if c > 0 else -np.identity(3) # return an identity matrix or a 180-degree rotation
    
# This k is called the skew-symmetric matrix of the vector v.
# Rodrigues' rotation formula uses this skew-symmetric matrix to compute the rotation matrix.
# Rodrigues’ rotation formula comes in —
# it gives us the exact 3×3 rotation matrix for any axis and angle in a clean, compact form.
# https://chatgpt.com/share/68fc6e31-0ad0-8011-8938-ce5b41b68a13 check the last part of this chat

    k = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.identity(3) + k + k @ k * ((1 - c) / (s ** 2))


# -------------------------------------------------------------------
# Decompose rotation matrix → pitch & roll
# -------------------------------------------------------------------
def decompose_rotation_to_pitch_roll(rotation_matrix: np.ndarray) -> tuple[float, float]:
    """Decomposes a 3×3 rotation matrix into pitch (Y-axis) and roll (X-axis)."""
    if rotation_matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3×3")

    pitch = np.arctan2(
        -rotation_matrix[2, 0], # why negative here ? 
        # ? This negative sign is used to account for the coordinate system conventions and to ensure that the pitch angle is calculated correctly.
        np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2)
    )
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    return pitch, roll

# this decomposes the rotation matrix into pitch and roll angles using the arctan2 function.
# ZYX (yaw–pitch–roll) Euler order is commonly used in aerospace and robotics applications.
# -------------------------------------------------------------------