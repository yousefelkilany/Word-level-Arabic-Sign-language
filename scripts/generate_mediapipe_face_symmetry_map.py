import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from core.constants import FACE_SYMMETRY_MAP_PATH, LANDMARKERS_DIR, LOCAL_INPUT_DATA_DIR


def init_face_model():
    face_base_options = BaseOptions(
        model_asset_path=os.path.join(LANDMARKERS_DIR, "face_landmarker.task"),
        delegate=BaseOptions.Delegate.CPU,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(face_options)


def get_face_mesh_symmetry_indices(face_model, image_path):
    if not os.path.exists(image_path):
        raise ValueError(f"Invalid path: {image_path = }")

    image = cv2.imread(image_path)
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = face_model.detect(frame)
    if results.face_landmarks is None or len(results.face_landmarks) == 0:
        raise ValueError("No face detected in reference image.")

    face_kps = np.array([(lm.x, lm.y, lm.z) for lm in results.face_landmarks[0]])
    centroid = np.mean(face_kps, axis=0)
    centered_points = face_kps - centroid
    flipped_points = centered_points.copy()
    flipped_points[:, 0] = -flipped_points[:, 0]

    dists = cdist(flipped_points, centered_points)
    row_ind, col_ind = linear_sum_assignment(dists)
    symmetry_map = [0] * len(face_kps)
    for r, c in zip(row_ind, col_ind):
        symmetry_map[r] = c
    return symmetry_map


def gen_symmetry_map(face_model, image_path: str) -> np.ndarray:
    symmetry_map = get_face_mesh_symmetry_indices(face_model, image_path)
    print(f"Generated mapping for {len(symmetry_map)} points.")

    symmetry_arr = np.array([symmetry_map[i] for i in range(0, 478)])
    missing_points = len(symmetry_arr) - len(set(symmetry_arr))
    assert missing_points == 0, (
        f"Invalid mapping, duplicated indices means some points are mapped to the same point\nThere're {missing_points} missing points"
    )

    sanity_checks = [(33, 263), (61, 291), (133, 362), (50, 280)]
    for u, v in sanity_checks:
        assert symmetry_arr[u] == v, "Mapping has some incorrect matchings"

    for k, v in enumerate(symmetry_arr):
        assert symmetry_arr[v] == k, f"Mapping is not bidirectional for {k} -> {v}"

    print("Sanity checks passed!")
    return symmetry_arr


if __name__ == "__main__":
    try:
        face_model = init_face_model()

        # any pair of the ai generated faces 1-5 match successfully
        # https://generated.photos/faces/front-facing/male
        image_path = os.path.join(LOCAL_INPUT_DATA_DIR, "frontal-face-1.jpg")
        symmetry_arr1 = gen_symmetry_map(face_model, image_path)
        image_path = os.path.join(LOCAL_INPUT_DATA_DIR, "frontal-face-2.jpg")
        symmetry_arr2 = gen_symmetry_map(face_model, image_path)

        mismatching_mask = symmetry_arr1 != symmetry_arr2
        mistmatching_points = np.sum(mismatching_mask)
        if mistmatching_points > 0:
            print(f"{symmetry_arr1[mismatching_mask] = }")
            print(f"{symmetry_arr2[mismatching_mask] = }")
            raise ValueError("some points-pairs dont match on both images")

        np.save(
            FACE_SYMMETRY_MAP_PATH,
            symmetry_arr1,
        )
        print("saved symmetry_map as numpy successfully")

    except Exception as e:
        print(f"[ERROR] {e = }")
