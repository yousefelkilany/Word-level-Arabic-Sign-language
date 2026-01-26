import numpy as np
import plotly.graph_objects as go

from core.constants import FEAT_NUM, SEQ_LEN, FEAT_DIM
from core.draw_kps import (
    get_face_lms_list,
    get_hand_lms_list,
    get_pose_lms_list,
)
from core.mediapipe_utils import (
    FACE_KPS_CONNECTIONS,
    HAND_KPS_CONNECTIONS,
    POSE_KPS_CONNECTIONS,
)


def get_face_camera_view(points_nx3):
    """
    Calculates the optimal Plotly camera dictionary to view a 3D point cloud frontally.
    """
    center = np.mean(points_nx3, axis=0)
    centered_data = points_nx3 - center

    try:
        _, _, Vt = np.linalg.svd(centered_data)
    except np.linalg.LinAlgError:
        return None

    vec_right, vec_up, vec_norm = Vt[:, [0, 1, 2]]
    if abs(vec_right[2]) > abs(vec_up[2]):
        vec_up, vec_right = vec_right, vec_up

    if np.dot(vec_up, np.array([0, 0, 1])) < 0:  # vec_up[2] < 0
        vec_up = -vec_up

    # projections = np.dot(centered_data, vec_norm)
    # med_proj = np.median(projections)
    # if med_proj - np.min(projections) > np.max(projections) - med_proj:
    #     vec_norm = -vec_norm

    eye_pos = vec_norm * 3

    return dict(
        up=dict(x=vec_up[0], y=vec_up[1], z=vec_up[2]),
        eye=dict(x=eye_pos[0], y=eye_pos[1], z=eye_pos[2]),
        center=dict(x=0, y=0, z=0),
    )


def _generate_frame_traces(
    frame_data, active_slices_map, draw_lines, draw_points, colors
):
    """Helper to generate traces for a specific frame data array."""
    traces = []

    def get_line_trace(coords, connections, color, name):
        x_lines, y_lines, z_lines = [], [], []
        for p1_idx, p2_idx in connections:
            if p1_idx >= len(coords) or p2_idx >= len(coords):
                continue
            # TODO: START ON AUGMENTATONS H-FLIP SCRIPT FOR FACE, START ALREADY !!!
            if coords[p1_idx, 0] != coords[p2_idx, 0]:
                x_lines.extend([coords[p1_idx, 0], coords[p2_idx, 0], None])
            if coords[p1_idx, 2] != coords[p2_idx, 2]:
                y_lines.extend([coords[p1_idx, 2], coords[p2_idx, 2], None])
            if coords[p1_idx, 1] != coords[p2_idx, 1]:
                z_lines.extend([-coords[p1_idx, 1], -coords[p2_idx, 1], None])

        return go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(color=color, width=2),
            name=f"{name} Edges",
            showlegend=False,
            hoverinfo="skip",
        )

    for part_name, sl in active_slices_map.items():
        part_data = frame_data[sl]
        if part_data.shape[0] == 0:
            continue

        if draw_points:
            traces.append(
                go.Scatter3d(
                    x=part_data[:, 0],
                    y=part_data[:, 2],
                    z=-part_data[:, 1],
                    mode="markers",
                    name=part_name.upper(),
                    marker=dict(
                        size=2 if part_name == "face" else 5,
                        color=colors.get(part_name, "gray"),
                        opacity=0.8,
                    ),
                )
            )

        if draw_lines:
            coords = None
            connections = None
            match part_name:
                case "pose":
                    coords = get_pose_lms_list(part_data, False)
                    connections = POSE_KPS_CONNECTIONS

                case "rh" | "lh":
                    coords = get_hand_lms_list(part_data, False)
                    connections = HAND_KPS_CONNECTIONS

                case "face":
                    coords: np.ndarray = get_face_lms_list(part_data, False)
                    connections = FACE_KPS_CONNECTIONS

            if connections and (coords is not None and coords.size > 0):
                traces.append(
                    get_line_trace(
                        coords, connections, colors.get(part_name, "gray"), part_name
                    )
                )

    return traces


def calculate_layout_ranges(sequence, active_slices_map):
    """Calculates global bounding box across the ENTIRE sequence to prevent jitter."""
    seq_reshaped = sequence.reshape(SEQ_LEN, FEAT_NUM, FEAT_DIM)
    all_x, all_y, all_z = [], [], []

    for random_idx in [0, SEQ_LEN // 2, SEQ_LEN - 1]:
        for _, sl in active_slices_map.items():
            part_data = seq_reshaped[random_idx][sl]
            if part_data.shape[0] > 0:
                all_x.append(part_data[:, 0])
                all_y.append(part_data[:, 2])
                all_z.append(-part_data[:, 1])

    if not all_x:
        return [-3, 3], [-3, 3], [-3, 3]

    xs, ys, zs = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_z)

    mid_x, mid_y, mid_z = (
        (np.max(xs) + np.min(xs)) / 2,
        (np.max(ys) + np.min(ys)) / 2,
        (np.max(zs) + np.min(zs)) / 2,
    )
    max_range = max(
        np.max(xs) - np.min(xs),
        np.max(ys) - np.min(ys),
        np.max(zs) - np.min(zs),
    )
    max_range = max_range / 2.0 * 1.1

    return (
        [mid_x - max_range, mid_x + max_range],
        [mid_y - max_range, mid_y + max_range],
        [mid_z - max_range, mid_z + max_range],
    )


def plot_3d_animation(
    sequence, active_slices_map, draw_lines=True, draw_points=True, title="3D Animation"
):
    """Generates a 3D animation Plotly Figure with Play/Pause and Slider."""
    seq_reshaped = sequence.reshape(SEQ_LEN, FEAT_NUM, FEAT_DIM)
    colors = {
        "pose": "#ef4444",
        "face": "#3b82f6",
        "rh": "#10b981",
        "lh": "#f59e0b",
    }

    x_range, y_range, z_range = calculate_layout_ranges(sequence, active_slices_map)

    camera_settings = None
    if len(active_slices_map) == 1:
        data = seq_reshaped[0][next(iter(active_slices_map.values()))]
        if data.shape[0] > 0:
            camera_settings = get_face_camera_view(
                np.column_stack((data[:, 0], data[:, 2], -data[:, 1]))
            )

    frames = []
    for k in range(SEQ_LEN):
        frame_traces = _generate_frame_traces(
            seq_reshaped[k], active_slices_map, draw_lines, draw_points, colors
        )
        frames.append(go.Frame(data=frame_traces, name=str(k)))
    initial_traces = _generate_frame_traces(
        seq_reshaped[0], active_slices_map, draw_lines, draw_points, colors
    )
    fig = go.Figure(data=initial_traces, frames=frames)

    play_btn = dict(
        label="▶ Play",
        method="animate",
        args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)],
    )
    pause_btn = dict(
        label="⏸ Pause",
        method="animate",
        args=[
            [None],
            dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate",
                transition=dict(duration=0),
            ),
        ],
    )
    update_menu = dict(
        type="buttons",
        showactive=False,
        x=0.1,
        y=0,
        xanchor="right",
        yanchor="top",
        pad=dict(t=0, r=10),
        buttons=[
            play_btn,
            pause_btn,
        ],
    )
    frames = [
        dict(
            method="animate",
            args=[
                [str(k)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(k),
        )
        for k in range(SEQ_LEN)
    ]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", range=x_range, showgrid=True),
            yaxis=dict(title="Z (Depth)", range=y_range, showgrid=True),
            zaxis=dict(title="-Y (Height)", range=z_range, showgrid=True),
            aspectmode="cube",
            camera=camera_settings,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
        updatemenus=[update_menu],
        sliders=[
            dict(
                steps=frames,
                active=0,
                currentvalue=dict(prefix="Frame: "),
                pad=dict(t=0),
            )
        ],
    )

    return fig
