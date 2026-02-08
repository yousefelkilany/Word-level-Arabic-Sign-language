import argparse
import glob
import os
import random
import sys
import subprocess

import cv2
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
from core.draw_kps import draw_all_kps_on_image
from core.mediapipe_utils import LandmarkerProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_image(path):
    return os.path.splitext(path.lower())[1] in IMAGE_EXTENSIONS


def is_video(path):
    return os.path.splitext(path.lower())[1] in VIDEO_EXTENSIONS


def process_image(input_path: str, output_path: str, processor: LandmarkerProcessor):
    """Process a single image."""
    print(f"Processing image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not read image file '{input_path}'.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Get both adjusted and raw (mp) keypoints
        _, mp_kps = processor.extract_frame_keypoints(
            frame_rgb, timestamp_ms=-1, adjusted=True, return_both=True
        )

        # Draw raw keypoints for visualization
        annotated_frame, *_ = draw_all_kps_on_image(frame_rgb, mp_kps)
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # If output_path is video-like, adjust it
        if is_video(output_path):
            output_path = os.path.splitext(output_path)[0] + ".jpg"

        cv2.imwrite(output_path, annotated_frame_bgr)
        print(f"Saved annotated image to: {output_path}")

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback

        traceback.print_exc()


def process_video(input_path: str, output_path: str, processor: LandmarkerProcessor):
    """Process a video file."""
    if not is_video(output_path):
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    ffmpeg_proc = None
    out = None
    try:
        ffmpeg_proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"Warning: ffmpeg failed ({e}). Falling back to cv2.VideoWriter...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for _ in tqdm(range(total_frames), desc="Frames"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # extract_frame_keypoints uses VIDEO mode internally if processor.video_mode is True
            mp_kps = processor.extract_frame_keypoints(frame_rgb, timestamp_ms=ts)
            annotated_frame, *_ = draw_all_kps_on_image(frame_rgb, mp_kps)
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            if ffmpeg_proc and ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.write(annotated_frame_bgr.tobytes())
            elif out:
                out.write(annotated_frame_bgr)

    finally:
        cap.release()
        if ffmpeg_proc:
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        elif out:
            out.release()
        print(f"Saved annotated video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone MediaPipe Processor Script"
    )
    parser.add_argument("--input", "-i", help="Path to input image or video file")
    parser.add_argument("--dir", "-d", help="Directory to pick a random file from")
    parser.add_argument("--output", "-o", help="Path to output file")
    args = parser.parse_args()

    input_file = args.input

    if not input_file and args.dir:
        files = []
        for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            files.extend(glob.glob(os.path.join(args.dir, f"*{ext}")))
            files.extend(glob.glob(os.path.join(args.dir, f"*{ext.upper()}")))

        if not files:
            print(f"Error: No supported files found in {args.dir}")
            return
        input_file = random.choice(files)
        print(f"Picked random file: {input_file}")

    if not input_file:
        parser.print_help()
        return

    is_vid = is_video(input_file)
    ext = os.path.splitext(input_file)[1]
    input_filename = os.path.basename(input_file)
    if not args.output:
        output_file = f"{input_filename}_annotated" + ext
    else:
        output_file = args.output
        if not os.path.splitext(output_file)[1]:
            output_file += ext

    print("Initializing LandmarkerProcessor...")
    processor = LandmarkerProcessor.create(video_mode=is_vid)

    try:
        if is_vid:
            process_video(input_file, output_file, processor)
        else:
            process_image(input_file, output_file, processor)
    finally:
        processor.close()


if __name__ == "__main__":
    main()
