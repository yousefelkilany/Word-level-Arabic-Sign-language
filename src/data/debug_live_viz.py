import cv2
import numpy as np


def visualize_debug_skeleton(kps, canvas_size=500):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    valid_mask = (np.abs(kps[:, 0]) > 1e-6) | (np.abs(kps[:, 1]) > 1e-6)
    valid_kps = kps[valid_mask]

    if len(valid_kps) == 0:
        cv2.putText(
            canvas,
            "NO VALID KPS",
            (10, canvas_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        return canvas, "Range: N/A"

    min_x, max_x = valid_kps[:, 0].min(), valid_kps[:, 0].max()
    min_y, max_y = valid_kps[:, 1].min(), valid_kps[:, 1].max()

    range_x = max_x - min_x
    range_y = max_y - min_y

    padding = 50
    available_size = canvas_size - (padding * 2)

    max_range = max(range_x, range_y, 1e-6)
    scale = available_size / max_range

    data_mid_x = (min_x + max_x) / 2
    data_mid_y = (min_y + max_y) / 2
    cx, cy = canvas_size // 2, canvas_size // 2

    for i, (x, y, z) in enumerate(kps):
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            continue

        px = int(cx + (x - data_mid_x) * scale)
        py = int(cy + (y - data_mid_y) * scale)

        # pose=6 -> face=136 -> rh=21 -> lh=21
        if i < 6:
            color = (255, 255, 255)
        elif i < 136 + 6:
            color = (255, 0, 0)
        elif i < 136 + 6 + 21:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(canvas, (px, py), 2, color, -1)

    report = (
        f"X Range: [{min_x:.4f}, {max_x:.4f}] (Span: {range_x:.4f})\n"
        f"Y Range: [{min_y:.4f}, {max_y:.4f}] (Span: {range_y:.4f})\n"
        f"Scale Applied: {scale:.2f}"
    )

    y_txt = 20
    for line in report.split("\n"):
        cv2.putText(
            canvas, line, (10, y_txt), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1
        )
        y_txt += 15

    return canvas, report


if __name__ == "__main__":
    kps = np.load("data/all_kps/01-train/0001.npz")["all_kps"]
    canvas, report = visualize_debug_skeleton(kps)
    cv2.imshow("Debug Skeleton", canvas)
    cv2.waitKey(0)
