import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from core.constants import STATIC_ASSETS_DIR
from core.mediapipe_utils import (
    POSE_KPS_CONNECTIONS,
    mp_idx_to_reduced_kps_idx,
    pose_kps_mp_idx,
    reduced_face_kps,
    reduced_hand_kps,
    simplified_face_contours,
    simplified_face_paths,
    simplified_hand_connections,
)

if __name__ == "__main__":
    try:
        kps_json_path = os.path.join(
            STATIC_ASSETS_DIR, "simplified_kps_connections.json"
        )
        with open(kps_json_path, "w+") as f:
            json.dump(
                {
                    "pose_kps": pose_kps_mp_idx,
                    "face_kps": reduced_face_kps,
                    "hand_kps": reduced_hand_kps,
                    "pose_connections": POSE_KPS_CONNECTIONS,
                    "face_contours": simplified_face_contours,
                    "face_paths": simplified_face_paths,
                    "hand_connections": simplified_hand_connections,
                    "mp_idx_to_kps_idx": mp_idx_to_reduced_kps_idx,
                },
                f,
            )

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
