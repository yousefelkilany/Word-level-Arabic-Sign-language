import uuid

import streamlit as st

from core.mediapipe_utils import KP2SLICE


def get_visual_controls(total_samples, rnd_key):
    idx, draw_lines, draw_points, separate_view, active_slices = (None,) * 5

    col_sel, col_checks = st.columns([2, 3])
    with col_sel:
        idx = st.number_input(
            f"Sample Index (0 - {total_samples - 1})",
            min_value=0,
            max_value=total_samples - 1,
            value=0,
            help=f"Select a sample from 0 to {total_samples - 1}",
            key=f"sample_idx-{rnd_key}",
        )

    with col_checks:
        st.write("Visual Controls:")
        c_m, c = st.columns([2, 3])
        draw_lines = c_m.checkbox("Lines", True, key=f"draw_lines-{rnd_key}")
        draw_points = c_m.checkbox("Points", False, key=f"draw_points-{rnd_key}")
        single_view = c_m.checkbox("Only one", True, key=f"single_view-{rnd_key}")
        if not single_view:
            separate_view = c_m.checkbox(
                "Separated", True, key=f"separate_view-{rnd_key}"
            )

        if single_view:
            (c1,) = c.columns([1])
            show_part = c1.selectbox(
                "Body", ["pose", "face", "rh", "lh"], key=f"show_part-{rnd_key}"
            )
            show_pose, show_face, show_rh, show_lh = (False,) * 4
            match show_part:
                case "pose":
                    show_pose = True
                case "face":
                    show_face = True
                case "rh":
                    show_rh = True
                case "lh":
                    show_lh = True
        else:
            (c1, c2) = c.columns([1, 1])
            show_pose = c1.checkbox("Body", True, key=f"show_pose-{rnd_key}")
            show_face = c1.checkbox("Face", True, key=f"show_face-{rnd_key}")
            show_rh = c2.checkbox("RH", False, key=f"show_rh-{rnd_key}")
            show_lh = c2.checkbox("LH", False, key=f"show_lh-{rnd_key}")

    active_slices = {}
    if show_pose:
        active_slices["pose"] = KP2SLICE["pose"]
    if show_face:
        active_slices["face"] = KP2SLICE["face"]
    if show_rh:
        active_slices["rh"] = KP2SLICE["rh"]
    if show_lh:
        active_slices["lh"] = KP2SLICE["lh"]

    return (idx, draw_lines, draw_points, separate_view, active_slices)
