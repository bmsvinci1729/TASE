import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# above's purpose ? 

# above's purpose ? 

from tase.perception.kinematics import JOINT_LANDMARK_MAP, landmarks_to_dict, calculate_angle
from tase.perception.kinematics import T_POSE_VECTORS, get_rotation_matrix, decompose_rotation_to_pitch_roll

# -------------------------------------------------------------------
# Extract 3D landmarks from image
# -------------------------------------------------------------------
def extract_3d_landmarks(image_path: str, model_path: str):
    """
    Processes a single image to extract 3D pose world landmarks using MediaPipe Tasks API.

    Args:
        image_path (str): The full path to the input image file.
        model_path (str): The full path to the MediaPipe pose landmarker model file.

    Returns:
        A PoseLandmarkerResult object containing the detected landmarks,
        or None if no pose is detected.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found at {image_path}")
        return None

    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # smthing like a custom option is create with base options and running mode, base following the pose_landmarler_heavy.task
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
        # what does running mode image mean ?
        # It means the model will process the image as a single frame.
        # what are different ruuning modes ?
        # Other modes include VIDEO (for processing video streams) and LIVE_STREAM (for real-time
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        # first get the mediapipe image
        mp_image = mp.Image.create_from_file(image_path)
        # print(mp_image)
        # print("HEEEEYAAAA FAAA")

        # detect over it
        result = landmarker.detect(mp_image)

        if not result.pose_world_landmarks:
            print(f"[WARNING] No pose detected in {image_path}")
            return None

        return result

### so till here 3d landmarks are extracted, /detected from img and stored in the result variable

def visualize_landmarks(image_path, result):
    """
    Draws 2D pose landmarks on the image using MediaPipe's drawing utilities.
    """
    import mediapipe as mp
    import cv2
    from mediapipe.framework.formats import landmark_pb2

    # Load the original image
    image = cv2.imread(image_path)

    # Drawing utils
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    annotated_image = image.copy()

    # Guard against wrong input types
    if not hasattr(result, 'pose_landmarks'):
        print("[WARNING] visualize_landmarks expected a MediaPipe result, got incompatible type.")
        return

    # Convert list of landmarks to a NormalizedLandmarkList for drawing
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(
            x=lmk.x,
            y=lmk.y,
            z=lmk.z,
            visibility=lmk.visibility
        ) for lmk in result.pose_landmarks[0]
    ])

    # Draw the landmarks and connections
    mp_drawing.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # Show the annotated image
    cv2.imshow("Pose Landmarks", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save it
    cv2.imwrite("images/annotated_pose.jpg", annotated_image)
    print("[LOG] Annotated image saved as images/annotated_pose.jpg")

# -------------------------------------------------------------------
# Convert image → landmarks → JOINT ANGLES
# -------------------------------------------------------------------

def get_pose_angles_from_image(image_path: str, model_path: str):
    """
    End-to-end function to get a dictionary of joint angles from an image,
    now including multi-axis joints (hips and shoulders).
    """
    landmarker_result = extract_3d_landmarks(image_path, model_path) # called here the abvoe function
    if not landmarker_result:
        return None # redundant checky but safe
    # print("HOOOOLAAAAA ", len(landmarker_result.pose_world_landmarks))
# Landmark(x=-0.1774439960718155, y=-0.6302552223205566, z=-0.23247210681438446, visibility=0.9999808073043823, presence=0.9999581575393677), 
    # --- FIXED: handle multiple possible output formats ---
    pose_world_landmarks = landmarker_result.pose_world_landmarks # where is pose worldlandmarks defined ? in the mediapipe result variable
    if isinstance(pose_world_landmarks[0], (list, tuple)):
        world_landmarks = pose_world_landmarks[0]
    elif hasattr(pose_world_landmarks[0], "landmarks"):
        world_landmarks = pose_world_landmarks[0].landmarks
    else:
        world_landmarks = pose_world_landmarks

    landmark_coords = landmarks_to_dict(world_landmarks) # u get a dict of z, x, y for each landmark index  
    if not landmark_coords:
        return None

    joint_angles = {}

    # --- 1. Single-axis joints (elbows, knees) --- dof - deg of freedom
    single_dof_joints = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
    for joint_name in single_dof_joints:
        p1_enum, p2_enum, p3_enum = JOINT_LANDMARK_MAP[joint_name] # get the landmark indices for that joint from the map
        #    Example ref # 'left_elbow': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
        p1, p2, p3 = landmark_coords[p1_enum], landmark_coords[p2_enum], landmark_coords[p3_enum]
        # so for each joint u get the 3d coords of the 3 landmarks involved in that joint as above in line one before the one just above
        # what is p1, p2, p3 here ?
        # p1: shoulder, p2: elbow, p3: wrist points in 3d space for left elbow for example
        # so is p1, p2 p3 = z, x, y of those landmarks ? yes
        angle = np.pi - calculate_angle(p1, p2, p3)
        # why a pi - of that angle ?
        # This adjustment is made to align with conventional joint angle definitions, that means ?
        # It means we are converting the angle from a mathematical representation to a more intuitive one, where 0 degrees is the neutral position.
        joint_angles[joint_name] = angle

    # --- 2. Multi-axis joints (shoulders, hips) ---
    multi_dof_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    for joint_name in multi_dof_joints:
        p1_enum, p2_enum, p3_enum = JOINT_LANDMARK_MAP[joint_name]
        p1, p2, p3 = landmark_coords[p1_enum], landmark_coords[p2_enum], landmark_coords[p3_enum]

        current_limb_vec = p3 - p2
        reference_limb_vec = T_POSE_VECTORS[joint_name]

        rot_matrix = get_rotation_matrix(reference_limb_vec, current_limb_vec)
        pitch, roll = decompose_rotation_to_pitch_roll(rot_matrix)

        joint_angles[f"{joint_name}_pitch"] = pitch
        joint_angles[f"{joint_name}_roll"] = roll

    return joint_angles


# -------------------------------------------------------------------
# Test block not used in the main.pyy its only while building this module
# -------------------------------------------------------------------
if __name__ == '__main__':
    MODEL_ASSET_PATH = os.path.join('assets', 'pose_landmarker_heavy.task')
    TEST_IMAGE_PATH = os.path.join('images', 'test_pose.jpg')

    if not os.path.exists(MODEL_ASSET_PATH):
        print("[ERROR] Model file not found. Please place it in the 'assets' folder.")
    elif not os.path.exists(TEST_IMAGE_PATH):
        print("[ERROR] Test image not found. Please place one in the 'images' folder.")
    else:
        print(f"[LOG] Processing image: {TEST_IMAGE_PATH}")
        angles = get_pose_angles_from_image(TEST_IMAGE_PATH, MODEL_ASSET_PATH)

        if angles:
            print("\n[LOG] --- Calculated Joint Angles (radians / degrees) ---")
            for joint, angle in angles.items():
                print(f"[LOG] {joint:<20}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
        else:
            print("\n[WARNING] Could not calculate joint angles. Pose detection may have failed.")
        # Visualize landmarks using MediaPipe detection result, not the angles dict
        vis_result = extract_3d_landmarks(TEST_IMAGE_PATH, MODEL_ASSET_PATH)
        if vis_result:
            visualize_landmarks(TEST_IMAGE_PATH, vis_result)

    # Visualize the landmarks on the image
    