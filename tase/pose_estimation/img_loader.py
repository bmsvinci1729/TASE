import cv2
import numpy as np

def load_image(file_path: str) -> np.ndarray:
    """Load an image from disk and return an RGB float32 array in [0,1].

    Raises FileNotFoundError if the path is invalid.
    """
    image_bgr = cv2.imread(file_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {file_path}")
    image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_np.astype(np.float32) / 255.0

# --- preprocess_image ---
def preprocess_image(image: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    """Resize RGB image to target_size for downstream pose models."""
    return cv2.resize(image, target_size)