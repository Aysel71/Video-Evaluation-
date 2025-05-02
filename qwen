import os
import torch
import numpy as np
import traceback
import json
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from vllm import SamplingParams
from transformers import CLIPProcessor, CLIPModel
import glob
import subprocess
import vllm
import time
import logging
import sys
import argparse
import gc

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qwen_vllm_embedding.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Путь к директории с фреймами
frames_directory = '/mnt/public-datasets/a.mirzoeva/Video-MME/output2/skbELjWHyXA/frames'

# Токен доступа Hugging Face
HF_TOKEN = "token" 

def load_qwen_model_vllm():
    """Загрузка модели Qwen с использованием vLLM"""
    logging.info("Загрузка модели Qwen2-VL с использованием vLLM...")
    
    try:
        # Установка токена доступа
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        
        # Загрузка процессора отдельно (не через vLLM)
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", token=HF_TOKEN)
        logging.info("Процессор успешно загружен")
        
        # Загрузка модели через vLLM с оптимизированным использованием памяти
        model = LLM(
            model="Qwen/Qwen2-VL-7B-Instruct",
            trust_remote_code=True,
            tensor_parallel_size=1,  # Используем 1 GPU
            dtype="half",  # Половинная точность для экономии памяти
            max_model_len=2048,  # Ограничение длины входа
            enforce_eager=True,  # Использование eager режима для отладки
            gpu_memory_utilization=0.8,  # Использование 80% доступной памяти GPU
            tokenizer_mode="slow"  # Совместимость с мультимодальными моделями
        )
        
        logging.info("Модель успешно загружена через vLLM")
        
        # Информация о модели
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            logging.info(f"Использовано памяти GPU: {gpu_memory_allocated:.2f} ГБ")
        
        return model, processor
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели через vLLM: {e}")
        logging.error(traceback.format_exc())
        return None, None

def get_embeddings(model, processor, image_path):
    """Извлечение эмбеддингов из изображения с использованием vLLM"""
    try:
        # Загрузка изображения
        image = Image.open(image_path).convert('RGB')
        
        # Ограничиваем размер изображения
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # Обработка изображения процессором
        inputs = processor(
            images=[image],
            text=["<image>"],
            return_tensors="pt"
        )
        
        # Получение эмбеддингов изображения
        # Для vLLM нужно использовать API модели, который отличается от стандартного
        # Поэтому нам нужно получить прямой доступ к vision tower
        if hasattr(model, "model") and hasattr(model.model, "qwen2") and hasattr(model.model.qwen2, "vision_tower"):
            # Получаем доступ к vision tower через vLLM
            vision_tower = model.model.qwen2.vision_tower
            
            # Перемещаем тензоры на нужное устройство
            inputs = {k: v.to(vision_tower.device) for k, v in inputs.items()}
            
            # Получаем эмбеддинги
            with torch.no_grad():
                outputs = vision_tower(inputs.pixel_values)
                image_embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Нормализация
                image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
                
                # Преобразование в numpy
                embeddings = image_embeddings.cpu().numpy()
        else:
            # Альтернативный способ, если структура модели отличается
            logging.warning("Нестандартная структура модели, используем альтернативный метод")
            
            
            # Создаем промпт для извлечения эмбеддингов
            prompt = f"<image>\nGenerate embedding for this image."
            
            # Кодируем изображение в base64
            import base64
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Получаем ответ от модели с эмбеддингами
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=16,  # Маленькое значение, т.к. нам нужны только эмбеддинги
                return_embeddings=True  # Запрашиваем эмбеддинги
            )
            
            outputs = model.generate([prompt], sampling_params, img_str)
            
            # Извлекаем эмбеддинги из ответа
            embeddings = np.array(outputs[0].embeddings)
        
        return embeddings
    except Exception as e:
        logging.error(f"Ошибка при обработке {image_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def process_batch(model, processor, batch_files, batch_idx, total_batches):
    """Обработка пакета файлов"""
    batch_embeddings = {}
    failures = 0
    
    for img_path in tqdm(batch_files, desc=f"Пакет {batch_idx+1}/{total_batches}"):
        img_name = os.path.basename(img_path)
        
        try:
            # Извлечение эмбеддингов
            embedding = get_embeddings(model, processor, img_path)
            
            if embedding is not None:
                batch_embeddings[img_name] = embedding.tolist()
            else:
                failures += 1
                logging.warning(f"Не удалось получить эмбеддинг для {img_name}")
        except Exception as e:
            failures += 1
            logging.error(f"Ошибка при обработке {img_name}: {e}")
        
        # Очистка памяти после каждого изображения
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    success_rate = ((len(batch_files) - failures) / len(batch_files)) * 100 if batch_files else 0
    logging.info(f"Успешно обработано: {len(batch_files) - failures}/{len(batch_files)} изображений ({success_rate:.1f}%)")
    
    return batch_embeddings

def fallback_to_clip():
    """Резервный вариант - использование CLIP при проблемах с Qwen"""
    logging.info("Переключение на модель CLIP...")
    
    try:
        
        # Загрузка модели CLIP
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        logging.info("Успешно загружена модель CLIP как резервный вариант")
        return model, processor
    except Exception as e:
        logging.error(f"Ошибка при загрузке CLIP: {e}")
        return None, None

def main():
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Извлечение эмбеддингов из фреймов с помощью Qwen через vLLM")
    parser.add_argument("--batch_size", type=int, default=10, help="Размер пакета для обработки")
    parser.add_argument("--continue_from", type=int, default=0, help="Номер пакета, с которого продолжить обработку")
    parser.add_argument("--check_progress", action="store_true", help="Проверить текущий прогресс и выйти")
    parser.add_argument("--force_clip", action="store_true", help="Принудительно использовать CLIP вместо Qwen")
    
    args = parser.parse_args()
    
    # Проверка прогресса, если запрошено
    if args.check_progress:
        check_progress()
        return
    
    # Установка vLLM
    setup_vllm()
    
    # Загрузка модели
    if args.force_clip:
        logging.info("Принудительное использование CLIP по запросу пользователя")
        model, processor = fallback_to_clip()
        prefix = "clip"
    else:
        # Попытка загрузить Qwen через vLLM
        model, processor = load_qwen_model_vllm()
        
        # Если не удалось, используем CLIP как резервный вариант
        if model is None or processor is None:
            logging.warning("Не удалось загрузить Qwen, переключение на CLIP...")
            model, processor = fallback_to_clip()
            prefix = "clip"
        else:
            prefix = "qwen_vllm"
    
    if model is None or processor is None:
        logging.error("Не удалось загрузить ни одну модель. Завершение работы.")
        return
    
    # Получение списка файлов
    image_files = sorted(glob.glob(os.path.join(frames_directory, "*.jpg")) + 
                      glob.glob(os.path.join(frames_directory, "*.png")))
    
    if not image_files:
        logging.error(f"В директории {frames_directory} не найдено изображений.")
        return
    
    logging.info(f"Найдено {len(image_files)} изображений.")
    
    # Загрузка уже обработанных результатов, если указано продолжение
    all_embeddings = {}
    if args.continue_from > 0:
        for i in range(1, args.continue_from + 1):
            batch_file = os.path.join(os.path.dirname(frames_directory), f"{prefix}_embeddings_batch_{i}.json")
            if os.path.exists(batch_file):
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                        all_embeddings.update(batch_data)
                    logging.info(f"Загружены данные из {batch_file}: {len(batch_data)} эмбеддингов")
                except Exception as e:
                    logging.error(f"Ошибка при загрузке {batch_file}: {e}")
    
    batch_size = args.batch_size
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    logging.info(f"Разделено на {len(batches)} пакетов по {batch_size} изображений")
    
    if args.continue_from > 0:
        batches = batches[args.continue_from:]
        logging.info(f"Пропускаем {args.continue_from} уже обработанных пакетов")
    
    # Тестирование на одном изображении
    if len(image_files) > 0:
        test_image = image_files[0]
        logging.info(f"Тестирование на изображении: {os.path.basename(test_image)}")
        test_embedding = get_embeddings(model, processor, test_image)
        
        if test_embedding is not None:
            logging.info(f"Тест успешен. Размерность эмбеддинга: {test_embedding.shape}")
            logging.info(f"Пример значений эмбеддинга: {test_embedding[:5]}")
        else:
            logging.error("Тест не пройден. Переключение на CLIP...")
            model, processor = fallback_to_clip()
            prefix = "clip"
            
            if model is None or processor is None:
                logging.error("Не удалось загрузить CLIP. Завершение работы.")
                return
            
            # Повторное тестирование с CLIP
            test_embedding = get_embeddings(model, processor, test_image)
            if test_embedding is None:
                logging.error("Тест с CLIP также не пройден. Завершение работы.")
                return
    
    start_time = time.time()

    output_dir = os.path.dirname(frames_directory)
    for batch_idx, batch in enumerate(batches):
        actual_batch_idx = batch_idx + args.continue_from
        logging.info(f"Обработка пакета {actual_batch_idx+1}/{len(batches) + args.continue_from}")
        
        batch_embeddings = process_batch(model, processor, batch, actual_batch_idx+1, len(batches) + args.continue_from)
        
        # Сохранение промежуточных результатов
        batch_output_path = os.path.join(output_dir, f"{prefix}_embeddings_batch_{actual_batch_idx+1}.json")
        with open(batch_output_path, 'w') as f:
            json.dump(batch_embeddings, f)
        logging.info(f"Промежуточные результаты сохранены в {batch_output_path}")
        all_embeddings.update(batch_embeddings)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
      
        elapsed_time = time.time() - start_time
        progress_percent = (batch_idx + 1) / len(batches) * 100
        avg_time_per_batch = elapsed_time / (batch_idx + 1)
        estimated_remaining = avg_time_per_batch * (len(batches) - (batch_idx + 1))
        
        logging.info(f"Прогресс: {progress_percent:.1f}%. "
                   f"Ожидаемое оставшееся время: {estimated_remaining/60:.1f} минут")
    
    output_path = os.path.join(output_dir, f"{prefix}_embeddings_all.json")
    with open(output_path, 'w') as f:
        json.dump(all_embeddings, f)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Обработка завершена! Все эмбеддинги сохранены в {output_path}")
    logging.info(f"Всего извлечено эмбеддингов для {len(all_embeddings)}/{len(image_files)} изображений")
    logging.info(f"Затраченное время: {elapsed_time/60:.2f} минут")

def check_progress(prefix="qwen_vllm"):
    """Проверка текущего прогресса обработки"""
    output_dir = os.path.dirname(frames_directory)
    
    # Также проверяем CLIP файлы
    qwen_files = sorted(glob.glob(os.path.join(output_dir, f"{prefix}_embeddings_batch_*.json")))
    clip_files = sorted(glob.glob(os.path.join(output_dir, "clip_embeddings_batch_*.json")))
    
    # Получение списка всех фреймов
    image_files = glob.glob(os.path.join(frames_directory, "*.jpg")) + glob.glob(os.path.join(frames_directory, "*.png"))
    total_images = len(image_files)
    
    print(f"\nОбщее количество изображений: {total_images}")
    
    # Проверка Qwen файлов
    if qwen_files:
        print(f"\nФайлы {prefix}:")
        total_qwen_processed = 0
        for batch_file in qwen_files:
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    batch_count = len(batch_data)
                    total_qwen_processed += batch_count
                    batch_num = os.path.basename(batch_file).split('_')[-1].split('.')[0]
                    print(f"Пакет {batch_num}: {batch_count} изображений")
            except Exception as e:
                print(f"Ошибка при чтении {batch_file}: {e}")
        
        if total_images > 0:
            progress_percent = total_qwen_processed / total_images * 100
            print(f"Всего {prefix}: {total_qwen_processed}/{total_images} изображений ({progress_percent:.1f}%)")
    
    # Проверка CLIP файлов
    if clip_files:
        print(f"\nФайлы CLIP:")
        total_clip_processed = 0
        for batch_file in clip_files:
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    batch_count = len(batch_data)
                    total_clip_processed += batch_count
                    batch_num = os.path.basename(batch_file).split('_')[-1].split('.')[0]
                    print(f"Пакет {batch_num}: {batch_count} изображений")
            except Exception as e:
                print(f"Ошибка при чтении {batch_file}: {e}")
        
        if total_images > 0:
            progress_percent = total_clip_processed / total_images * 100
            print(f"Всего CLIP: {total_clip_processed}/{total_images} изображений ({progress_percent:.1f}%)")
    
    if not qwen_files and not clip_files:
        print("Не найдено файлов с результатами.")

if __name__ == "__main__":
    main()
