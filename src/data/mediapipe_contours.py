def join_upper_lower_connections(upper, lower):
    return tuple([*upper, *[(v, u) for (u, v) in lower[::-1]]])


def get_contour_from_path(path):
    return [(u, v) for u, v in zip(path[:-1], path[1:])]


def get_path_from_contour(contour):
    path = [*contour[0]]
    for i, (u, v) in enumerate(contour[1:]):
        if path[-1] != u:
            path.append(u)
        if path[-1] != v:
            path.append(v)
    return path


# reference: https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/face_mesh_connections.py
# reference: https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

FACEMESH_OUTER_LIPS_UPPER = tuple(
    (
        (61, 185),
        (185, 40),
        (40, 39),
        (39, 37),
        (37, 0),
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
    )
)
FACEMESH_OUTER_LIPS_LOWER = tuple(
    (
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
    )
)
FACEMESH_OUTER_LIPS = join_upper_lower_connections(
    FACEMESH_OUTER_LIPS_UPPER, FACEMESH_OUTER_LIPS_LOWER
)

FACEMESH_INNER_LIPS_UPPER = tuple(
    (
        (78, 191),
        (191, 80),
        (80, 81),
        (81, 82),
        (82, 13),
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    )
)
FACEMESH_INNER_LIPS_LOWER = tuple(
    (
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
    )
)
FACEMESH_INNER_LIPS = join_upper_lower_connections(
    FACEMESH_INNER_LIPS_UPPER, FACEMESH_INNER_LIPS_LOWER
)

FACEMESH_LEFT_EYE_UPPER = tuple(
    (
        (263, 466),
        (466, 388),
        (388, 387),
        (387, 386),
        (386, 385),
        (385, 384),
        (384, 398),
        (398, 362),
    )
)
FACEMESH_LEFT_EYE_LOWER = tuple(
    (
        (263, 249),
        (249, 390),
        (390, 373),
        (373, 374),
        (374, 380),
        (380, 381),
        (381, 382),
        (382, 362),
    )
)
FACEMESH_LEFT_EYE = join_upper_lower_connections(
    FACEMESH_LEFT_EYE_UPPER, FACEMESH_LEFT_EYE_LOWER
)

FACEMESH_LEFT_EYEBROW_UPPER = tuple(
    (
        (276, 300),
        (300, 293),
        (293, 334),
        (334, 296),
        (296, 336),
        (336, 285),
    )
)
FACEMESH_LEFT_EYEBROW_LOWER = tuple(
    (
        (276, 283),
        (283, 282),
        (282, 295),
        (295, 285),
    )
)
FACEMESH_LEFT_EYEBROW = join_upper_lower_connections(
    FACEMESH_LEFT_EYEBROW_UPPER, FACEMESH_LEFT_EYEBROW_LOWER
)

FACEMESH_RIGHT_EYE_UPPER = tuple(
    (
        (33, 246),
        (246, 161),
        (161, 160),
        (160, 159),
        (159, 158),
        (158, 157),
        (157, 173),
        (173, 133),
    )
)
FACEMESH_RIGHT_EYE_LOWER = tuple(
    (
        (33, 7),
        (7, 163),
        (163, 144),
        (144, 145),
        (145, 153),
        (153, 154),
        (154, 155),
        (155, 133),
    )
)
FACEMESH_RIGHT_EYE = join_upper_lower_connections(
    FACEMESH_RIGHT_EYE_UPPER, FACEMESH_RIGHT_EYE_LOWER
)

FACEMESH_RIGHT_EYEBROW_UPPER = tuple(
    (
        (46, 70),
        (70, 63),
        (63, 105),
        (105, 66),
        (66, 107),
    )
)
FACEMESH_RIGHT_EYEBROW_LOWER = tuple(
    (
        (46, 53),
        (53, 52),
        (52, 65),
        (65, 55),
        (55, 107),
    )
)
FACEMESH_RIGHT_EYEBROW = join_upper_lower_connections(
    FACEMESH_RIGHT_EYEBROW_UPPER, FACEMESH_RIGHT_EYEBROW_LOWER
)

FACEMESH_FACE_OVAL = tuple(
    (
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
        (284, 251),
        (251, 389),
        (389, 356),
        (356, 454),
        (454, 323),
        (323, 361),
        (361, 288),
        (288, 397),
        (397, 365),
        (365, 379),
        (379, 378),
        (378, 400),
        (400, 377),
        (377, 152),
        (152, 148),
        (148, 176),
        (176, 149),
        (149, 150),
        (150, 136),
        (136, 172),
        (172, 58),
        (58, 132),
        (132, 93),
        (93, 234),
        (234, 127),
        (127, 162),
        (162, 21),
        (21, 54),
        (54, 103),
        (103, 67),
        (67, 109),
        (109, 10),
    )
)

FACEMESH_CONTOURS = {
    "face_oval": FACEMESH_FACE_OVAL,
    "inner_lips": FACEMESH_INNER_LIPS,
    "outer_lips": FACEMESH_OUTER_LIPS,
    "left_eye": FACEMESH_LEFT_EYE,
    "right_eye": FACEMESH_RIGHT_EYE,
    "left_eyebrow": FACEMESH_LEFT_EYEBROW,
    "right_eyebrow": FACEMESH_RIGHT_EYEBROW,
}

FACEMESH_CONTOUR_PATHS = {
    k: get_path_from_contour(v) for k, v in FACEMESH_CONTOURS.items()
}

from mediapipe.python.solutions.pose import PoseLandmark as mp_pose_landmark

POSEMESH_OPEN = tuple(
    (
        (mp_pose_landmark.RIGHT_WRIST, mp_pose_landmark.RIGHT_ELBOW),
        (mp_pose_landmark.RIGHT_ELBOW, mp_pose_landmark.RIGHT_SHOULDER),
        (mp_pose_landmark.RIGHT_SHOULDER, mp_pose_landmark.LEFT_SHOULDER),
        (mp_pose_landmark.LEFT_SHOULDER, mp_pose_landmark.LEFT_ELBOW),
        (mp_pose_landmark.LEFT_ELBOW, mp_pose_landmark.LEFT_WRIST),
    )
)
