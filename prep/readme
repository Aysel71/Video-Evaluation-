## 📽️ Video-MME Frame & Subtitle Extractor

This repository provides tools to extract frames and aligned subtitles from videos in the [Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME) dataset. It is designed for multimodal model evaluation (e.g., CLIP, ViCLIP) where precise alignment between video frames and textual information 
is required using optimized frame sampling https://github.com/pytorch/torchcodec  and also there is file slice_and_exctract.py that follows orginal slicing of video from video-mme that follwos the following pipeline the Python program slices multiple frames from a given video and 
extracts the corresponding subtitles from an SRT file. It uses OpenCV for video processing and the pysubs2 library for subtitle parsing.

---

## 📦 Features

- Extract evenly spaced frames from `.mp4` videos using **TorchCodec**
- Match frames with corresponding `.srt` subtitles based on timestamp
- Output:
  - `frames/` folder with JPEG images
  - `subtitles.txt` with matched subtitles
  - One `.json` per video with full metadata (frame name, timestamp, subtitle)
  - Optional: batch stats in `log.txt` and `summary.json`
- Fully compatible with PyTorch pipelines

---

## Requirements

- Python 3.9+
- `torch`, `torchvision`, `torchcodec`
- `pysubs2`
- `ffmpeg` (via `conda install -c conda-forge ffmpeg`)

---

## Usage

### 1. Extract frames and subtitles for a single video

```bash
python3 slice_and_extract.py \
  --video_path path/to/video.mp4 \
  --srt_path path/to/video.srt \
  --num_frames 12 \
  --output_path output/
```

### 2. Batch process the entire dataset

Make sure your files are organized:

```
videos/
└── video1.mp4
    video2.mp4
subtitles/
└── video1.srt
    video2.srt
```

Then run:

```bash
python3 batch_process.py
```

This will populate:

```
output/
└── video1/
    ├── frames/
    ├── subtitles.txt
    └── video1.json
```

Also saved:
- `log.txt`
- `summary.json` with dataset stats & top subtitle words

---

## Example summary.json snippet

```json
{
  "total_videos_processed": 853,
  "total_videos_skipped": 47,
  "total_frames_extracted": 10236,
  "total_frames_without_subtitles": 119,
  "most_common_words": [
    ["water", 48],
    ["earth", 43],
    ...
  ]
}
```

