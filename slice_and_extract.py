import os
import shutil
import numpy as np
import pysubs2
import torchcodec
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import torch

def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    return [(int(np.round(seg_size * i)) + int(np.round(seg_size * (i + 1)))) // 2
            for i in range(desired_num_frames)]

def create_frame_output_dir(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def slice_frames(video_path, srt_path, num_frames, output_path):
    print(f"Extracting from video: {video_path}")
    
    frame_out_dir = os.path.join(output_path, "frames")
    create_frame_output_dir(frame_out_dir)

    # Считываем все кадры как тензоры [T, H, W, C] (uint8)
    frames, metadata = torchcodec.read_video(video_path, stream="video", return_metadata=True)
    fps = metadata["fps"]
    total_frames = frames.shape[0]

    # Выбираем кадры
    selected_ids = get_seq_frames(total_frames, num_frames)

    subtitles = []
    if srt_path and os.path.exists(srt_path):
        subs = pysubs2.load(srt_path, encoding="utf-8")

    for idx in selected_ids:
        frame_tensor = frames[idx].permute(2, 0, 1) / 255.0  # [C,H,W], float32

        time_seconds = idx / fps
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        out_name = f"{os.path.basename(video_path).replace('.', '_')}_frame_{time_str}.jpg"
        save_path = os.path.join(frame_out_dir, out_name)
        save_image(frame_tensor, save_path)

        # Найти субтитр по времени
        if srt_path and os.path.exists(srt_path):
            cur_time_ms = int(time_seconds * 1000)
            text = ""
            for sub in subs:
                if sub.start <= cur_time_ms <= sub.end:
                    text = sub.text.replace("\\N", " ")
                    break
            if text.strip():
                subtitles.append(text)

    if subtitles:
        with open(os.path.join(output_path, "subtitles.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(subtitles))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames + subtitles using TorchCodec.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--srt_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--output_path", type=str, default="output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    slice_frames(args.video_path, args.srt_path, args.num_frames, args.output_path)
