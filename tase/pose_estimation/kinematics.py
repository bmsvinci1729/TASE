"""Kinematic helpers to convert BODY25 keypoints to a simple humanoid pose.

Functions provided:
  - body25_to_humanoid_pose(body25_4d): returns a 1D array of joint angles in MuJoCo order
"""

import numpy as np
# marking beginning of body part indices

B = {
    "nose": 0,
    "neck": 1,
    "r_shoulder": 2,
    "r_elbow": 3,
    "r_wrist": 4,
    "l_shoulder": 5,
    "l_elbow": 6,
    "l_wrist": 7,
    "mid_hip": 8,
    "r_hip": 9,
    "r_knee": 10,
    "r_ankle": 11,
    "l_hip": 12,
    "l_knee": 13,
    "l_ankle": 14
}


def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-6)


# --- angle_between ---
def angle_between(v1, v2):
    v1 = unit(v1)
    v2 = unit(v2)
    dot = np.clip(np.dot(v1, v2), -1, 1)
    return np.arccos(dot)


# --- compute_spine_angles ---
def compute_spine_angles(neck, hip):
    spine = unit(neck - hip)
    yaw = np.arctan2(spine[0], spine[2])
    pitch = -np.arctan2(spine[1], spine[2])
    roll = np.arctan2(spine[1], spine[0])
    return yaw, pitch, roll


# --- leg_angles ---
def leg_angles(hip, knee, ankle):
    thigh = knee - hip
    shin = ankle - knee
    t = unit(thigh)

    hip_roll = np.arctan2(t[1], t[0])
    hip_yaw = np.arctan2(t[0], t[2])
    hip_pitch = np.arctan2(-t[1], t[2])
    knee_angle = angle_between(thigh, shin)
    return hip_roll, hip_yaw, hip_pitch, knee_angle


# --- arm_angles ---
def arm_angles(shoulder, elbow, wrist):
    upper = elbow - shoulder
    lower = wrist - elbow
    u = unit(upper)

    shoulder_pitch = np.arctan2(-u[1], u[2])
    shoulder_roll = np.arctan2(u[0], u[2])
    elbow_angle = angle_between(upper, lower)
    return shoulder_pitch, shoulder_roll, elbow_angle


# --- body25_to_humanoid_pose ---
def body25_to_humanoid_pose(body25_4d):
    """Convert BODY25-like keypoints (25 x 4) into a MuJoCo humanoid joint vector.

    Expects body25_4d with columns [x,y,z,visibility]. Returns float32 1D array.
    """
    body25 = body25_4d[:, :3]

    yaw, pitch, roll = compute_spine_angles(body25[B["neck"]], body25[B["mid_hip"]])

    r_hr, r_hz, r_hy, r_k = leg_angles(body25[B["r_hip"]], body25[B["r_knee"]], body25[B["r_ankle"]])
    l_hr, l_hz, l_hy, l_k = leg_angles(body25[B["l_hip"]], body25[B["l_knee"]], body25[B["l_ankle"]])

    r_s1, r_s2, r_e = arm_angles(body25[B["r_shoulder"]], body25[B["r_elbow"]], body25[B["r_wrist"]])
    l_s1, l_s2, l_e = arm_angles(body25[B["l_shoulder"]], body25[B["l_elbow"]], body25[B["l_wrist"]])

    return np.array([
        yaw, pitch, roll,
        r_hr, r_hz, r_hy, r_k,
        l_hr, l_hz, l_hy, l_k,
        r_s1, r_s2, r_e,
        l_s1, l_s2, l_e
    ], dtype=np.float32)
