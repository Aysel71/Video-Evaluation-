import os
import shutil
import numpy as np
import pysubs2
import torch
import json
from torchcodec.decoders import VideoDecoder
from torchvision.utils import save_image


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    return [(int(np.round(seg_size * i)) + int(np.round(seg_size * (i + 1)))) // 2
            for i in range(desired_num_frames)]

def prepare_output_dirs(base_output_path):
    if os.path.exists(base_output_path):
        shutil.rmtree(base_output_path)
    os.makedirs(os.path.join(base_output_path, "frames"), exist_ok=True)

def slice_frames(video_path, srt_path, num_frames, output_root):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_root, video_id)
    frames_dir = os.path.join(output_path, "frames")

    print(f"\n🔍 Processing video: {video_id}")
    prepare_output_dirs(output_path)

    decoder = VideoDecoder(video_path, device="cpu")
    total_frames = decoder.metadata.num_frames
    fps = decoder.metadata.average_fps
    selected_ids = get_seq_frames(total_frames, num_frames)

    subs = pysubs2.load(srt_path, encoding="utf-8") if srt_path and os.path.exists(srt_path) else []

    subtitles = []
    json_output = {
        "video_id": video_id,
        "frames": []
    }

    for idx in selected_ids:
        frame_tensor = decoder[idx].float() / 255.0  # [C,H,W]
        time_seconds = idx / fps
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"

        frame_name = f"{video_id}_frame_{timestamp}.jpg"
        frame_path = os.path.join(frames_dir, frame_name)
        save_image(frame_tensor, frame_path)

        # Найдём субтитр
        cur_time_ms = int(time_seconds * 1000)
        matched_sub = ""
        for sub in subs:
            if sub.start <= cur_time_ms <= sub.end:
                matched_sub = sub.text.replace("\\N", " ").strip()
                break

        subtitles.append(matched_sub)
        json_output["frames"].append({
            "frame": frame_name,
            "timestamp": timestamp,
            "subtitle": matched_sub
        })

    # Сохраняем subtitles.txt
    with open(os.path.join(output_path, "subtitles.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(subtitles))

    # Сохраняем единый JSON
    json_path = os.path.join(output_path, f"{video_id}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_output, jf, indent=2, ensure_ascii=False)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--srt_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--output_path", type=str, default="output")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    slice_frames(args.video_path, args.srt_path, args.num_frames, args.output_path)
