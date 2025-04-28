import os
import sys
import concurrent.futures
import json
import time
from datetime import datetime
from slice_with_pyav import slice_frames_pyav

def process_video(args):
    video_file, videos_dir, subtitles_dir, output_dir, step = args
    try:
        video_path = os.path.join(videos_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        srt_path = os.path.join(subtitles_dir, f"{video_id}.srt")
        
        if not os.path.exists(srt_path):
            print(f"No subtitle file for {video_id}, skipping.")
            return {"video_id": video_id, "status": "skipped", "reason": "no_subtitle"}
        
        print(f"\nProcessing {video_id}...")
        start_time = time.time()
        
        # Вызываем функцию slice_frames_pyav
        success = slice_frames_pyav(
            video_path=video_path,
            srt_path=srt_path,
            step=step,
            output_root=output_dir
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not success:
            return {"video_id": video_id, "status": "error", "error": "Processing failed"}
        
        # Собираем статистику
        json_path = os.path.join(output_dir, video_id, f"{video_id}.json")
        stats = {"video_id": video_id, "status": "success", "processing_time": processing_time}
        
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stats["frames_count"] = len(data["frames"])
                
                # Считаем кадры без субтитров
                frames_without_subs = sum(1 for frame in data["frames"] if not frame["subtitle"].strip())
                stats["frames_without_subs"] = frames_without_subs
        
        return stats
    
    except Exception as e:
        return {"video_id": video_id, "status": "error", "error": str(e)}

def parallel_batch_process(videos_dir, subtitles_dir, output_dir, step=16, max_workers=8):
    print(f"Starting parallel video processing with PyAV at {datetime.now()}")
    print(f"Configuration:")
    print(f"- Videos directory: {videos_dir}")
    print(f"- Subtitles directory: {subtitles_dir}")
    print(f"- Output directory: {output_dir}")
    print(f"- Step (every Nth frame): {step}")
    print(f"- Max workers: {max_workers}")
    
    # Создаем output_dir, если он не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех видео файлов
    video_files = [f for f in os.listdir(videos_dir) if f.endswith((".mp4", ".avi", ".mkv"))]
    print(f"Found {len(video_files)} video files")
    
    # Сортируем файлы для порядка обработки
    video_files.sort()
    
    # Журнал процесса
    log_path = os.path.join(output_dir, "pyav_processing_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"PyAV parallel processing started at {datetime.now()}\n")
        log_file.write(f"Total videos to process: {len(video_files)}\n\n")
    
    # Подготавливаем аргументы для каждого видео
    args_list = [(video_file, videos_dir, subtitles_dir, output_dir, step) 
                for video_file in video_files]
    
    # Статистика
    results = []
    processed = 0
    skipped = 0
    errors = 0
    
    # Запускаем параллельную обработку
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video, args) for args in args_list]
        
        # Обрабатываем результаты по мере их завершения
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            
            # Обновляем счетчики
            if result["status"] == "success":
                processed += 1
            elif result["status"] == "skipped":
                skipped += 1
            else:
                errors += 1
            
            # Записываем прогресс
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[{i+1}/{len(video_files)}] {result['video_id']}: {result['status']}\n")
                if "error" in result:
                    log_file.write(f"  Error: {result['error']}\n")
            
            # Выводим прогресс
            progress = (i + 1) / len(video_files) * 100
            print(f"Progress: {progress:.1f}% ({i+1}/{len(video_files)}) - Success: {processed}, Skipped: {skipped}, Errors: {errors}", 
                  file=sys.stderr)
    
    # Записываем итоговую статистику в JSON
    summary = {
        "start_time": str(datetime.now()),
        "total_videos": len(video_files),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "video_results": results
    }
    
    summary_path = os.path.join(output_dir, "pyav_processing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing completed!")
    print(f"- Successfully processed: {processed}")
    print(f"- Skipped: {skipped}")
    print(f"- Errors: {errors}")
    print(f"Log saved to: {log_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    videos_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/videos"
    subtitles_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/subtitles" 
    output_dir = "/mnt/public-datasets/a.mirzoeva/Video-MME/output2"  # Используем другую директорию для PyAV
    
    # Используем больше процессов для лучшей производительности
    max_workers = 8  # Можно увеличить, если сервер имеет много ядер
    
    # Шаг извлечения кадров (каждый 16-й кадр)
    step = 16
    
    parallel_batch_process(videos_dir, subtitles_dir, output_dir, step, max_workers)
