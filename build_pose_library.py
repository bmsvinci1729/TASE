import os
import glob
import pickle

# --- IMPORTANT ---
# This import assumes your pose estimation function is in this location.
# Adjust if your file structure is different.
from tase.perception.pose_estimator import get_pose_angles_from_image

# --- Configuration ---
# Update these paths to match your project
MODEL_ASSET_PATH = os.path.join('assets', 'pose_landmarker_heavy.task')
IMAGE_DIR = 'images/'
OUTPUT_FILE = 'pose_library.pkl'

def build_library():
    """
    Runs the perception module on all images in the IMAGE_DIR
    and saves the resulting pose angle dictionaries to a pickle file.
    This implements the "pre-generated library" from the Final Module.
    """
    image_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, '*.png')))
    
    pose_library = []
    print(f"Found {len(image_paths)} images. Processing...")
    
    if not os.path.exists(MODEL_ASSET_PATH):
        print(f" Model asset not found at {MODEL_ASSET_PATH}")
        print("Please ensure the model file is in the 'assets' directory.")
        return

    for image_path in image_paths:
        print(f"  -> Processing {image_path}")
        try:
            # This function comes from your perception module 
            initial_angles = get_pose_angles_from_image(image_path, MODEL_ASSET_PATH)
            
            if initial_angles:
                pose_library.append(initial_angles)
                print(f"   ... Success! Added pose.")
            else:
                print(f"   ... Warning: No pose detected or failed.")
        except Exception as e:
            print(f"   ... ERROR processing {image_path}: {e}")

    if pose_library:
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(pose_library, f)
        print(f"\nSuccess! Saved {len(pose_library)} poses to {OUTPUT_FILE}.")
    else:
        print("\nNo poses were collected. Library file not created.")

if __name__ == '__main__':
    build_library()