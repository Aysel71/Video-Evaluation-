import os
from slice_and_extract import slice_frames

def batch_process(videos_dir, subtitles_dir, output_dir, num_frames=12):
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]

    print(f"Found {len(video_files)} video files.")
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        srt_path = os.path.join(subtitles_dir, f"{video_id}.srt")

        if not os.path.exists(srt_path):
            print(f" No subtitle file for {video_id}, skipping.")
            continue

        print(f"▶Processing {video_id}...")
        slice_frames(
            video_path=video_path,
            srt_path=srt_path,
            num_frames=num_frames,
            output_root=output_dir
        )

    print("Batch processing complete.")

if __name__ == "__main__":
    # Пример путей
    videos_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/videos"
    subtitles_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/subtitles"
    output_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/output"

    batch_process(videos_dir, subtitles_dir, output_dir, num_frames=12)
