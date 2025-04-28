import os
import shutil
import av
import numpy as np
import pysubs2
import json
from PIL import Image
import torch
from torchvision.transforms import ToTensor

def get_seq_frames(total_num_frames, step=16):
    """
    Извлекает каждый step-й кадр начиная с первого (индексы: 0, step, 2*step, ...)
    пока не достигнет конца видео.
    """
    selected_frames = list(range(0, total_num_frames, step))
    return selected_frames

def prepare_output_dirs(base_output_path):
    if os.path.exists(base_output_path):
        shutil.rmtree(base_output_path)
    os.makedirs(os.path.join(base_output_path, "frames"), exist_ok=True)

def slice_frames_pyav(video_path, srt_path, step=16, output_root="output"):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_root, video_id)
    frames_dir = os.path.join(output_path, "frames")
    print(f"\n🔍 Processing video: {video_id}")
    prepare_output_dirs(output_path)
    
    # Открываем видео с PyAV
    try:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        total_frames = int(video_stream.frames)
        fps = float(video_stream.average_rate)
        
        print(f"📊 Всего кадров в видео: {total_frames}, FPS: {fps}")
        print(f"🔄 Извлекаю каждый {step}-й кадр")
        
        # Получаем индексы кадров с заданным шагом
        selected_ids = get_seq_frames(total_frames, step)
        print(f"🎞️ Будет извлечено {len(selected_ids)} кадров")
        
        # Загружаем субтитры
        subs = pysubs2.load(srt_path, encoding="utf-8") if srt_path and os.path.exists(srt_path) else []
        subtitles = []
        json_output = {
            "video_id": video_id,
            "frames": []
        }
        
        # Извлекаем кадры
        for i, frame_idx in enumerate(selected_ids):
            print(f"🔧 Обработка кадра {i+1}/{len(selected_ids)} (индекс {frame_idx})", end="\r")
            
            # Ускоренное перемещение по видео
            container.seek(frame_idx, stream=video_stream)
            
            # Извлекаем кадр
            for frame in container.decode(video=0):
                # Получаем numpy массив
                img_array = frame.to_ndarray(format='rgb24')
                
                # Здесь можно добавить дополнительную обработку изображения если нужно
                
                # Вычисляем временную метку
                time_seconds = frame_idx / fps
                minutes = int(time_seconds // 60)
                seconds = int(time_seconds % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                # Формируем имя файла
                frame_name = f"{video_id}_frame_{timestamp}.jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                
                # Сохраняем изображение
                img = Image.fromarray(img_array)
                img.save(frame_path, quality=95)
                
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
                
                break  # Берем только первый кадр из декодированных
        
        print(f"\n✅ Извлечение кадров завершено.")
        
        # Сохраняем subtitles.txt
        with open(os.path.join(output_path, "subtitles.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(subtitles))
        
        # Сохраняем единый JSON
        json_path = os.path.join(output_path, f"{video_id}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_output, jf, indent=2, ensure_ascii=False)
        
        print(f"📝 Metadata сохранена в {video_id}.json")
        
        return True
    
    except Exception as e:
        print(f"❌ Ошибка при обработке видео {video_id}: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--srt_path", type=str, default=None)
    parser.add_argument("--step", type=int, default=16, 
                      help="Шаг между кадрами (16 означает каждый 16-й кадр)")
    parser.add_argument("--output_path", type=str, default="output")
    
    args = parser.parse_args()
    slice_frames_pyav(args.video_path, args.srt_path, args.step, args.output_path)
