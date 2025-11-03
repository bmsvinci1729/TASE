import os
import json
from typing import Any, Dict, List
import cv2
import mediapipe as mp

from pose_estimator import extract_3d_landmarks


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _as_list(container: Any) -> List[Any]:
    # MediaPipe may return list-like or proto-style structures
    if isinstance(container, (list, tuple)):
        return container
    if hasattr(container, "landmarks"):
        return container.landmarks
    return container


def serialize_world_landmarks(result: Any) -> List[Dict[str, float]]:
    world = getattr(result, "pose_world_landmarks", None)
    if not world:
        return []
    points = _as_list(world)[0] if len(world) > 0 else []
    return [
        {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
        for p in points
    ]


def serialize_2d_landmarks(result: Any) -> List[Dict[str, float]]:
    lmk = getattr(result, "pose_landmarks", None)
    if not lmk:
        return []
    points = _as_list(lmk)[0] if len(lmk) > 0 else []
    out: List[Dict[str, float]] = []
    for p in points:
        item: Dict[str, float] = {"x": float(p.x), "y": float(p.y)}
        # Some versions may include z and/or visibility
        if hasattr(p, "z"):
            item["z"] = float(p.z)
        if hasattr(p, "visibility"):
            item["visibility"] = float(p.visibility)
        out.append(item)
    return out


def iter_images(images_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(exts)
    ]


def main() -> None:
    images_dir = os.path.join("images")
    model_path = os.path.join("assets", "pose_landmarker_heavy.task")
    out_dir = os.path.join("results", "poses")
    annotated_dir = os.path.join("results", "annotated")

    ensure_dir(out_dir)
    ensure_dir(annotated_dir)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        return
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found at {images_dir}")
        return

    image_paths = iter_images(images_dir)
    if not image_paths:
        print(f"[WARNING] No images found in {images_dir}")
        return

    total = len(image_paths)
    success_count = 0
    skipped = 0

    for idx, img_path in enumerate(image_paths, start=1):
        fname = os.path.basename(img_path)
        print(f"[LOG] ({idx}/{total}) Processing: {fname}")

        try:
            result = extract_3d_landmarks(img_path, model_path)
            if not result:
                print(f"[WARNING] Pose not detected for {img_path}")
                skipped += 1
                continue

            world_pts = serialize_world_landmarks(result)
            lmk2d_pts = serialize_2d_landmarks(result)

            payload: Dict[str, Any] = {
                "image": fname,
                "image_path": img_path,
                "world_landmarks": world_pts,
                "landmarks_2d": lmk2d_pts,
            }

            out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            success_count += 1
            print(f"[LOG] Saved pose to {out_path}")

            # Also save annotated image with 2D landmarks overlaid
            try:
                image = cv2.imread(img_path)
                if image is not None and lmk2d_pts:
                    # Build a NormalizedLandmarkList-like structure for drawing
                    from mediapipe.framework.formats import landmark_pb2
                    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    # The 2D points from Tasks API are typically in [0,1] normalized coords
                    # We directly construct proto for drawing utils
                    pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(
                            x=pt.get("x", 0.0),
                            y=pt.get("y", 0.0),
                            z=pt.get("z", 0.0),
                            visibility=pt.get("visibility", 0.0),
                        ) for pt in lmk2d_pts
                    ])

                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    annotated = image.copy()
                    mp_drawing.draw_landmarks(
                        annotated,
                        pose_landmarks_proto,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    annotated_path = os.path.join(
                        annotated_dir, f"{os.path.splitext(fname)[0]}_annotated.jpg"
                    )
                    cv2.imwrite(annotated_path, annotated)
                    print(f"[LOG] Saved annotated image to {annotated_path}")
                else:
                    print("[WARNING] Could not load image or no 2D landmarks to draw.")
            except Exception as draw_err:
                print(f"[WARNING] Failed to save annotated image for {fname}: {draw_err}")

        except Exception as e:
            print(f"[ERROR] Failed to process {fname}: {e}")
            skipped += 1

    print(
        f"[LOG] Completed. Successful: {success_count}, Skipped/Failed: {skipped}, Total: {total}"
    )


if __name__ == "__main__":
    main()

# import os
# import glob
# import numpy as np
# from pose_estimator import get_pose_angles_from_image, visualize_landmarks  # replace with actual module name

# # -------------------------------------------------------------------
# # Batch process all images in a directory
# # -------------------------------------------------------------------
# def process_all_images_in_directory(image_dir, model_path, visualize=False, save_results=True):
#     """
#     Runs get_pose_angles_from_image() on all images in a folder.

#     Args:
#         image_dir (str): Path to directory containing images.
#         model_path (str): Path to the MediaPipe pose model file.
#         visualize (bool): Whether to visualize landmarks.
#         save_results (bool): Whether to save angles to a text file.
#     """
#     # Accept common image extensions
#     image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
#     image_paths = []
#     for ext in image_extensions:
#         image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

#     if not image_paths:
#         print(f"[ERROR] No images found in directory: {image_dir}")
#         return

#     os.makedirs("results", exist_ok=True)
#     output_file = os.path.join("results", "pose_angles_summary.txt")

#     with open(output_file, "w") as f:
#         for idx, image_path in enumerate(sorted(image_paths)):
#             print(f"\n[LOG] ({idx+1}/{len(image_paths)}) Processing: {os.path.basename(image_path)}")
#             angles = get_pose_angles_from_image(image_path, model_path)

#             if not angles:
#                 print(f"[WARNING] Pose not detected for {image_path}")
#                 continue

#             # Log angles
#             f.write(f"\n=== {os.path.basename(image_path)} ===\n")
#             for joint, angle in angles.items():
#                 f.write(f"{joint:<25}: {angle:.4f} rad ({np.degrees(angle):.2f}°)\n")

#             # Optional visualization
#             if visualize:
#                 visualize_landmarks(image_path, get_pose_angles_from_image(image_path, model_path))

#     print(f"\n✅ All images processed. Results saved to: {output_file}")


# # -------------------------------------------------------------------
# # Example Usage
# # -------------------------------------------------------------------
# if __name__ == "__main__":
#     MODEL_ASSET_PATH = os.path.join('assets', 'pose_landmarker_heavy.task')
#     IMAGE_DIRECTORY = os.path.join('images', '')  # <-- change this to your folder

#     process_all_images_in_directory(
#         IMAGE_DIRECTORY,
#         MODEL_ASSET_PATH,
#         visualize=False,   # Set True if you want OpenCV windows to pop up
#         save_results=True
#     )

import os
import glob
import numpy as np
from pose_estimator import get_pose_angles_from_image, visualize_landmarks  # replace with actual module name
import traceback  # for detailed error logs

# -------------------------------------------------------------------
# Batch process all images in a directory
# -------------------------------------------------------------------
def process_all_images_in_directory(image_dir, model_path, visualize=False, save_results=True):
    """
    Runs get_pose_angles_from_image() on all images in a folder, continuing even if errors occur.
    """
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        print(f"[ERROR] No images found in directory: {image_dir}")
        return

    os.makedirs("results", exist_ok=True)
    output_file = os.path.join("results", "pose_angles_summary.txt")

    with open(output_file, "w") as f:
        for idx, image_path in enumerate(sorted(image_paths)):
            print(f"\n[LOG] ({idx+1}/{len(image_paths)}) Processing: {os.path.basename(image_path)}")

            try:
                angles = get_pose_angles_from_image(image_path, model_path)

                if not angles:
                    print(f"[WARNING] Pose not detected for {image_path}")
                    continue

                # Log angles
                f.write(f"\n=== {os.path.basename(image_path)} ===\n")
                for joint, angle in angles.items():
                    f.write(f"{joint:<25}: {angle:.4f} rad ({np.degrees(angle):.2f}°)\n")

                # Optional visualization
                visualize_landmarks(image_path, get_pose_angles_from_image(image_path, model_path))

            except Exception as e:
                print(f"[ERROR] Failed to process {os.path.basename(image_path)}: {e}")
                # Uncomment the line below if you want to see full error trace (for debugging)
                # traceback.print_exc()
                continue  # continue to next image even if this one fails

    print(f"\n✅ All images processed (with skipped errors). Results saved to: {output_file}")


# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_ASSET_PATH = os.path.join('assets', 'pose_landmarker_heavy.task')
    IMAGE_DIRECTORY = os.path.join('images', '')  # <-- change this to your folder

    process_all_images_in_directory(
        IMAGE_DIRECTORY,
        MODEL_ASSET_PATH,
        visualize=False,   # Set True if you want OpenCV windows to pop up
        save_results=True
    )
