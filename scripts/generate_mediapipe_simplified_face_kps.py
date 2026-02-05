import json
import os
import sys

import cv2
import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from core.constants import (
    FACE_SYMMETRY_MAP_PATH,
    LANDMARKERS_DIR,
    LOCAL_INPUT_DATA_DIR,
    SIMPLIFIED_FACE_CONNECTIONS_PATH,
)
from data.mediapipe_contours import (
    FACEMESH_CONTOURS,
    get_contour_from_path,
    get_path_from_contour,
)


def init_face_model():
    from mediapipe.tasks.python import BaseOptions, vision

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


def get_face_landmarks(face_model, image):
    import mediapipe as mp

    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = face_model.detect(frame)
    if results.face_landmarks is None or len(results.face_landmarks) == 0:
        raise ValueError("No face detected in reference image.")

    return results.face_landmarks[0]


def simplify_contour_path(
    raw_landmarks, contour_path, width, height, tolerance=5.0, target_count=None
):
    """
    raw_landmarks: The full list of 478 MediaPipe landmarks
    contour_indices: The ordered list of indices for a feature (e.g., LIPS)
    width, height: The dimensions of image (for aspect ratio)
    """

    def run_visvalingam(points, threshold=None, target_count=None):
        """
        Args:
            points: List of dicts [{'x': float, 'y': float, 'id': int}, ...]
            threshold: Stop removing points if the smallest area is larger than this.
            target_count: Stop removing points if we reach this specific number of points.

        Returns:
            List of dicts (the simplified path).
        """

        if threshold is None and target_count is None:
            raise ValueError("You should pass either threshold or target_count")

        def calculate_triangle_area(p1, p2, p3):
            """
            Calculates the area of a triangle defined by three points (x, y).
            Standard Shoelace formula.
            """
            return 0.5 * abs(
                p1["x"] * (p2["y"] - p3["y"])
                + p2["x"] * (p3["y"] - p1["y"])
                + p3["x"] * (p1["y"] - p2["y"])
            )

        current_points = points[:]
        while True:
            if len(current_points) <= 2:
                break
            if target_count is not None and len(current_points) <= target_count:
                break

            min_area = float("inf")
            min_index = -1
            for i in range(1, len(current_points) - 1):
                area = calculate_triangle_area(
                    current_points[i - 1], current_points[i], current_points[i + 1]
                )
                if area < min_area:
                    min_area = area
                    min_index = i

            if threshold is not None and min_area >= threshold:
                break

            current_points.pop(min_index)

        return current_points

    path = [
        {
            "x": raw_landmarks[idx].x * width,
            "y": raw_landmarks[idx].y * height,
            "id": idx,
        }
        for idx in contour_path
    ]
    return [p["id"] for p in run_visvalingam(path, tolerance, target_count)]


def get_simplified_contours(
    face_landmarks,
    width,
    height,
    tolerance=5.0,
    target_count_fraction=1,
    min_kps_per_contour=5,
):
    contours = [
        FACEMESH_CONTOURS["face_oval"],
        FACEMESH_CONTOURS["inner_lips"],
        FACEMESH_CONTOURS["outer_lips"],
    ]
    symmetric_left_contours = [
        FACEMESH_CONTOURS["left_eye"],
        FACEMESH_CONTOURS["left_eyebrow"],
    ]
    FACE_SYMMETRY_MAP = np.load(FACE_SYMMETRY_MAP_PATH).tolist()

    simplified_contours = [
        get_contour_from_path(
            simplify_contour_path(
                face_landmarks,
                get_path_from_contour(contour),
                width,
                height,
                tolerance,
                max(min_kps_per_contour, len(contour) * target_count_fraction),
            )
        )
        for contour in contours
    ]
    for contour in symmetric_left_contours:
        left_contour = simplify_contour_path(
            face_landmarks,
            get_path_from_contour(contour),
            width,
            height,
            tolerance,
            max(min_kps_per_contour, len(contour) * target_count_fraction),
        )
        right_contour = [FACE_SYMMETRY_MAP[idx] for idx in left_contour]
        simplified_contours.extend(
            [get_contour_from_path(left_contour), get_contour_from_path(right_contour)]
        )

    return simplified_contours


def validate_simplified_contours(
    simplified_contours,
    face_landmarks,
    width,
    height,
    min_covered_area_percent=0.9,
    verbose=False,
):
    from shapely.geometry import Polygon

    total_kps = 0
    total_simplified_kps = 0

    simplified_contours_areas = []
    for (name, contour), simplified_contour in zip(
        FACEMESH_CONTOURS.items(), simplified_contours
    ):
        contour_lms = [
            (face_landmarks[idx].x * width, face_landmarks[idx].y * height)
            for idx in get_path_from_contour(contour)
        ]
        total_kps += len(contour_lms) - 1

        simplified_contour_lms = [
            (face_landmarks[idx].x * width, face_landmarks[idx].y * height)
            for idx in get_path_from_contour(simplified_contour)
        ]
        total_simplified_kps += len(simplified_contour_lms) - 1

        polygon1 = Polygon(contour_lms)
        polygon2 = Polygon(simplified_contour_lms)
        intersection = polygon1.intersection(polygon2)
        simplified_contours_areas.append(intersection.area / polygon1.area)

        if verbose:
            print(
                f"{name}: {len(contour)} -> {len(simplified_contour)} -- covered area percent = {intersection.area / polygon1.area * 100:.2f}%"
            )
    if verbose:
        print("-" * 30)

    simplified_contours_areas = np.array(simplified_contours_areas)
    covered_area_percent = np.quantile(simplified_contours_areas, 0.35)

    if verbose:
        print(f"Total KPS: {total_kps} -> Total Simplified KPS: {total_simplified_kps}")
        print(f"Min covered area percent: {simplified_contours_areas.min() * 100:.2f}%")
        print(f"Max covered area percent: {simplified_contours_areas.max() * 100:.2f}%")
        print(f"Avg covered area percent: {covered_area_percent * 100:.2f}%")
    assert covered_area_percent > min_covered_area_percent, (
        f"Avg covered area percent is too low: {covered_area_percent * 100:.2f}%. "
        "Increase target_count_fraction or decrease tolerance."
    )

    return True


def generate_simplified_face_contours(
    tolerance=None,
    target_count_fraction=0.6,
    min_kps_per_contour=6,
    min_covered_area_percent=0.90,
    verbose=False,
):
    image_path = os.path.join(LOCAL_INPUT_DATA_DIR, "frontal-face-1.jpg")

    try:
        face_model = init_face_model()
        if not os.path.exists(image_path):
            raise ValueError(f"Invalid path: {image_path = }")

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        face_landmarks = get_face_landmarks(face_model, image)

        simplified_contours = get_simplified_contours(
            face_landmarks,
            width,
            height,
            tolerance,
            target_count_fraction,
            min_kps_per_contour,
        )

        validate_simplified_contours(
            simplified_contours,
            face_landmarks,
            width,
            height,
            min_covered_area_percent,
            verbose,
        )

        simplified_contours = {
            k: v for k, v in zip(FACEMESH_CONTOURS.keys(), simplified_contours)
        }

        return simplified_contours
    except Exception as e:
        print(f"[ERROR] {e = }")


def generate_simplified_face_paths(
    tolerance=None,
    target_count_fraction=0.6,
    min_kps_per_contour=6,
    min_covered_area_percent=0.90,
    verbose=False,
):
    simplified_contours = generate_simplified_face_contours(
        tolerance,
        target_count_fraction,
        min_kps_per_contour,
        min_covered_area_percent,
        verbose,
    )

    return {k: get_path_from_contour(v) for k, v in simplified_contours.items()}


if __name__ == "__main__":
    face_contours = generate_simplified_face_contours(
        tolerance=None,
        target_count_fraction=0.6,
        min_kps_per_contour=6,
        min_covered_area_percent=0.90,
    )

    face_paths = {k: get_path_from_contour(v) for k, v in face_contours.items()}

    with open(SIMPLIFIED_FACE_CONNECTIONS_PATH, "w+") as f:
        json.dump({"face_contours": face_contours, "face_paths": face_paths}, f)
