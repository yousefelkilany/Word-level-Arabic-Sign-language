import os
import subprocess

split = ["train", "test"][0]
samples_file = f"{split}-samples.txt"
with open(samples_file, "r") as f:
    samples = f.readlines()
    for sample in samples:
        sample_dir = sample.strip("\n")
        sample_name = os.path.basename(sample_dir)
        input_path = os.path.join(sample_dir, f"{sample_name}_%04d.jpg")

        output_dir = os.path.join(f"{os.path.dirname(sample_dir)}-mp4s")
        output_path = os.path.join(output_dir, f"{sample_name}.mp4")

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            f"{input_path}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            f"{output_path}",
        ]

        output = None
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[DONE] File {sample_name}")
        except Exception as e:
            print(e)
        # break
