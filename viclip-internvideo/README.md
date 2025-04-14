# 🔍 Running ViCLIP on the Video-MME Benchmark (with Docker)

This guide provides **step-by-step instructions** on how to run the [ViCLIP](https://github.com/OpenGVLab/InternVideo) model on the [Video-MME benchmark](https://github.com/Video-Understanding/MME), including data formatting, model loading, inference, and evaluation — all inside a Docker container.

---

## 🐳 Docker Setup

### 1. **Build the Docker Image**

Make sure you clone the ViCLIP repo and build the image:

```bash
cd InternVideo
docker build -t viclip-internvideo -f docker/Dockerfile .
```

### 2. **Run Jupyter in Docker**

```bash
docker run --rm -it \
  --name a.mirzoeva.viclip-jupyter \
  -v /mnt/public-datasets/a.mirzoeva:/workspace/data \
  -v /home/a.mirzoeva/notebooks:/workspace/notebooks \
  -v /home/a.mirzoeva/InternVideo:/workspace/InternVideo \
  -p 8888:8888 \
  --gpus "device=1" \
  --shm-size=2g \
  --memory=32g \
  --cpuset-cpus=0-9 \
  --workdir /workspace/notebooks \
  viclip-internvideo \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

To open Jupyter from your browser:
```bash
ssh -L 8888:localhost:8888 a.mirzoeva@<SERVER_IP>
```

---

## 🧠 Model Checkpoint & Files

Download weights and config from Hugging Face:
```bash
cd InternVideo/Data/InternVid/viclip/

wget https://huggingface.co/OpenGVLab/ViCLIP/resolve/main/model.safetensors -O ViCLIP-L_InternVid-FLT-10M.pth
wget https://huggingface.co/OpenGVLab/ViCLIP/resolve/main/config.json
wget https://huggingface.co/OpenGVLab/ViCLIP/resolve/main/bpe_simple_vocab_16e6.txt.gz
```

---

## 📁 Data Format

Place your extracted frames, subtitles, and metadata here:

```
/mnt/public-datasets/a.mirzoeva/Video-MME/output/<video_id>/
├── frames/                # 12 uniformly sampled frames
├── subtitles.txt          # Subtitle file (plain text)
└── <video_id>.json        # Metadata file (optional)
```

---

## ⚙️ How Inference Works

ViCLIP is a **multi-modal zero-shot model**. Here's the key:

### What data is passed into the model:
- A tensor of 8 video frames (preprocessed via `frames2tensor`).
- A list of text options (MCQ choices or subtitle lines).

### How we encode it:
```python
frames_tensor = frames2tensor(frames)           # [1, 8, 3, 224, 224]
video_feat = model.get_vid_features(frames_tensor)
text_feats = model.get_text_features(options, tokenizer)
```

### How the model gives prediction:
```python
probs, indices = model.get_predict_label(video_feat, text_feats, top=1)
```
The model returns **probabilities for each answer**, and we take the top prediction.

---

## ✅ Batch Inference on MME

We provide a script to loop through all videos + their questions:
```python
from viclip import get_viclip, retrieve_text
import pandas as pd, json, cv2, os
from tqdm import tqdm

qa_df = pd.read_parquet("/workspace/data/Video-MME/videomme/test-00000-of-00001.parquet")
models = get_viclip(size='l')
clip = models['viclip'].cuda()

results = []

for video_id in tqdm(qa_df.videoID.unique()):
    # Load frames and subtitles
    frame_dir = f"/workspace/data/Video-MME/output/{video_id}/frames"
    subs_path = f"/workspace/data/Video-MME/output/{video_id}/subtitles.txt"
    
    if not os.path.exists(frame_dir) or not os.path.exists(subs_path):
        continue

    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    frames = [cv2.imread(p) for p in frame_paths[:8]]

    with open(subs_path) as f:
        subtitles = [line.strip() for line in f if line.strip()]

    video_questions = qa_df[qa_df.videoID == video_id]

    for _, row in video_questions.iterrows():
        question = row['question']
        options = eval(row['options']) if isinstance(row['options'], str) else row['options']
        answer = row['answer']
        qid = row['question_id']

        try:
            pred_texts, _ = retrieve_text(frames, options, models=models, topk=1)
            prediction = pred_texts[0]
        except Exception:
            prediction = "ERROR"

        results.append({
            "video_id": video_id,
            "question_id": qid,
            "question": question,
            "options": options,
            "answer": answer,
            "response": prediction
        })

with open("viclip_predictions.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 📊 Evaluation

We provide an evaluation script to get per-category performance:

```bash
python evaluate.py \
  --results_file viclip_output_test.json \
  --video_duration_type short \
  --return_categories_accuracy \
  --return_task_types_accuracy
```

You’ll get:
- ✅ Overall Accuracy
- 📂 Per-domain breakdown (e.g., Knowledge, Sports)
- 🎯 Task-wise breakdown (Counting, Reasoning...)

---

## 📎 Output Format

We support two output JSON formats:

### Flat:
```json
{
  "video_id": "fFjv93ACGo8",
  "question_id": "001-2",
  "answer": "A",
  "response": "D. Travel vlog"
}
```

### Grouped:
```json
{
  "video_id": "fFjv93ACGo8",
  "duration": "short",
  "domain": "Knowledge",
  "sub_category": "Humanity & History",
  "questions": [ ... ]
}
```

---

## 🧩 Summary

You now:
- ✅ Run ViCLIP in Docker
- 🎞️ Process any MME video with subtitles and frames
- 📈 Run zero-shot MCQ VQA
- 🧪 Evaluate with ground truth answers



