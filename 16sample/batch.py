import os
import json
from datetime import datetime
from slice_and_extract import slice_frames

def batch_process(videos_dir, subtitles_dir, output_dir, step=16):
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]

    total_videos = 0
    total_skipped = 0
    total_frames = 0
    total_subtitle_misses = 0
    subtitle_word_counter = {}

    log_lines = []
    summary = {
        "total_videos_processed": 0,
        "total_videos_skipped": 0,
        "total_frames_extracted": 0,
        "total_frames_without_subtitles": 0,
        "videos": []
    }

    log_lines.append(f"# Batch started at {datetime.now()}\n")
    log_lines.append(f"Found {len(video_files)} video files\n")

    for video_file in sorted(video_files):
        video_path = os.path.join(videos_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        srt_path = os.path.join(subtitles_dir, f"{video_id}.srt")

        if not os.path.exists(srt_path):
            msg = f" No subtitle file for {video_id}, skipping."
            print(msg)
            log_lines.append(msg)
            total_skipped += 1
            continue

        print(f"\n▶️  Processing {video_id}...")
        log_lines.append(f"\n▶️  Processing {video_id}...")

        output_path = os.path.join(output_dir, video_id)

        # Вызываем обновленную функцию slice_frames с параметром step вместо num_frames
        slice_frames(
            video_path=video_path,
            srt_path=srt_path,
            step=step,
            output_root=output_dir
        )

        total_videos += 1

        # Проверим JSON и соберём инфу
        json_path = os.path.join(output_path, f"{video_id}.json")
        frames_count = 0
        missing_subs = 0
        
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                frames_count = len(data["frames"])
                total_frames += frames_count
                
                for frame in data["frames"]:
                    if not frame["subtitle"].strip():
                        missing_subs += 1
                    else:
                        # Посчитаем слова
                        words = frame["subtitle"].lower().split()
                        for w in words:
                            subtitle_word_counter[w] = subtitle_word_counter.get(w, 0) + 1

        total_subtitle_misses += missing_subs
        summary["videos"].append({
            "video_id": video_id,
            "frames_total": frames_count,
            "frames_with_subtitles": frames_count - missing_subs,
            "frames_without_subtitles": missing_subs
        })

    summary["total_videos_processed"] = total_videos
    summary["total_videos_skipped"] = total_skipped
    summary["total_frames_extracted"] = total_frames
    summary["total_frames_without_subtitles"] = total_subtitle_misses
    summary["most_common_words"] = sorted(subtitle_word_counter.items(), key=lambda x: -x[1])[:20]

    # Сохраняем log.txt
    with open(os.path.join(output_dir, "log.txt"), "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_lines))

    # Сохраняем summary.json
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print("\n Final Statistics:")
    print(f" Videos processed: {total_videos}")
    print(f" Total frames extracted: {total_frames}")
    print(f" Videos skipped (no subtitles): {total_skipped}")
    print(f" Frames with no subtitle match: {total_subtitle_misses}")
    print(f" Log saved to: {output_dir}/log.txt")
    print(f" Summary saved to: {output_dir}/summary.json")

if __name__ == "__main__":
    videos_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/videos"
    subtitles_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/subtitles"
    output_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/output"

    batch_process(videos_dir, subtitles_dir, output_dir, step=16)
