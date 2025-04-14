# Running ViCLIP on the Video-MME Benchmark


## 1. Environment Setup

```bash
conda create -n video-mme-env python=3.10 -y
conda activate video-mme-env

# Install required packages
pip install torch torchvision
pip install einops pandas tqdm pyarrow
pip install opencv-python-headless
pip install timm==0.6.12
pip install safetensors
pip install transformers
pip install huggingface_hub
```

---

## 2. Clone ViCLIP Repository

```bash
git clone https://github.com/OpenGVLab/InternVideo.git
cd InternVideo/Data/InternVid/viclip
```

---

## 3. Download ViCLIP Checkpoint

```bash
huggingface-cli login
huggingface-cli download OpenGVLab/ViCLIP ViCLIP-L_InternVid-FLT-10M.pth
mv ~/.cache/huggingface/hub/models--OpenGVLab--ViCLIP/snapshots/*/*.pth ./
```

---

## 4. Prepare Video-MME Dataset

Ensure you have the following directory structure:

```
/mnt/public-datasets/a.mirzoeva/Video-MME/
├── output/
│   └── <video_id>/
│       ├── frames/
│       ├── subtitles.txt
│       └── <video_id>.json
├── videomme/
│   └── test-00000-of-00001.parquet
```

---

## 5. Understanding the ViCLIP Inference Flow

### Inputs to the Model:
- **Frames**: Sequence of up to 32 RGB video frames.
- **Options**: List of 4 textual answer options (A-D).

### Data Transformation:
- `frames2tensor(frames)`:
  - Resizes to 224x224
  - Normalizes using mean/std
  - Converts to tensor of shape `[1, 8, 3, 224, 224]`

- `get_text_features(options)`:
  - Tokenizes each option
  - Encodes with text encoder
  - Returns normalized embeddings `[4, D]`

### Matching:
```python
score = torch.matmul(video_embedding, text_embeddings.T)  # [1, 4]
probs = softmax(score)
prediction = argmax(probs)
```

---

## 6. Batch Inference on Video-MME

```python
# Provided as `run_viclip.py`
import os, json, pandas as pd, torch, cv2, numpy as np
from tqdm import tqdm
from viclip import get_viclip, retrieve_text
import ast

models = get_viclip(size='l')
clip = models['viclip'].cuda()

video_root = "/workspace/data/Video-MME/output"
qa_path = "/workspace/data/Video-MME/videomme/test-00000-of-00001.parquet"
qa_df = pd.read_parquet(qa_path)

all_predictions, grouped_output = [], []

for video_id in tqdm(qa_df["videoID"].unique()):
    ...  # load frames and subtitles (see full script in repo)

    for _, row in video_df.iterrows():
        options = ast.literal_eval(row['options'].replace('\x00', ''))
        result, _ = retrieve_text(frames, options, models=models, topk=1)
        predicted_text = result[0]

        all_predictions.append({
            "video_id": video_id,
            "question_id": row['question_id'],
            "answer": row['answer'],
            "response": predicted_text,
            ... # other fields
        })

with open("viclip_predictions.json", "w") as f:
    json.dump(all_predictions, f, indent=2)

with open("viclip_output_test.json", "w") as f:
    json.dump(grouped_output, f, indent=2)
```

---

## 7. Evaluation

Use the official Video-MME evaluation script to assess performance:

```bash
python eval_viclip.py \
  --results_file viclip_output_test.json \
  --video_duration_type short \
  --return_categories_accuracy \
  --return_task_types_accuracy
```

This will output accuracy by domain, task type, and overall.

---

## ✅ Outputs
- `viclip_predictions.json`: Flat list of Q&A with responses.
- `viclip_output_test.json`: Structured output in the benchmark format.
- Console printout: Accuracy by category, subcategory, task type, and overall performance.

---


Feel free to contribute improvements or submit issues in the repo!

