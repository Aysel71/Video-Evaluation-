Sure! Below is a `README.md` template for your GitHub repository based on the provided details about the code and its changes:

---

# Video Frame Extraction (Every 16th Frame)

This repository contains a Python script designed for extracting frames from videos, specifically extracting every 16th frame starting from the first frame. The extraction logic has been modified to allow flexibility in selecting frames at a custom interval, with the default step set to 16 frames. This approach allows for easier manipulation and processing of video data by selecting frames at regular intervals.

## Features
- Extract every 16th frame (by default) from a video, starting from the first frame.
- Customizable frame extraction step via command-line argument (`--step`).
- Efficient extraction method using `range()` to generate frame indices.
- Seamlessly integrates with PyTorch for further processing.
- The extracted frames can be saved as images or used for other applications.

## Requirements
- Python 3.x
- `torch`
- `torchvision`
- `numpy`
- `pysubs2`
- `argparse`
- `json`
- `shutil`
- `os`
- `torchcodec` (for video decoding)

You can install the required dependencies with:

```bash
pip install torch torchvision numpy pysubs2 argparse json
```

Note: `torchcodec` may need to be installed separately or adapted depending on your environment.

## How It Works

### Core Logic

The main function responsible for extracting frames from a video is `get_seq_frames`. This function allows you to specify a custom frame extraction step. By default, it extracts every 16th frame, starting from the first frame. The logic has been modified from the original function, which evenly distributed the frame extraction across the video's entire length.

### Usage

1. **Command Line Arguments:**
   - `--video_path`: Path to the video file.
   - `--output_dir`: Directory where the extracted frames will be saved.
   - `--step`: Step for selecting frames. Default is 16.

2. **Example Command:**

```bash
python extract_frames.py --video_path path/to/video.mp4 --output_dir frames_output --step 16
```

This command will extract every 16th frame from the video located at `path/to/video.mp4` and save them to the `frames_output` directory.

### Key Functions
1. **`get_seq_frames(total_num_frames, step=16)`**:
   - This function calculates the indices of frames that will be extracted, with a default step of 16. 
   - For example, it will return frames 1, 17, 33, ..., etc., based on the total number of frames in the video.

2. **`slice_frames(video_path, step=16)`**:
   - This function uses the `get_seq_frames` logic to slice the video and extract the frames with the given step. 
   - The extracted frames are saved as individual image files in the specified output directory.

3. **`parse_args()`**:
   - This function parses command-line arguments, allowing the user to customize the `step` and specify the video and output directories.

## Example Code

Here's an example of how the frame extraction works:

```python
import os
import shutil
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
from torchvision.utils import save_image

def get_seq_frames(total_num_frames, step=16):
    """
    Returns a list of frame indices to extract, starting from the first frame
    with a given step interval. By default, it extracts every 16th frame.
    
    :param total_num_frames: Total number of frames in the video
    :param step: Step interval for frame extraction
    :return: List of frame indices
    """
    # Calculate indices of frames to be extracted (1, 17, 33, ...)
    return list(range(0, total_num_frames, step))

def slice_frames(video_path, output_dir, step=16):
    """
    Extracts frames from the video at the given step interval and saves them as images.
    
    :param video_path: Path to the video file
    :param output_dir: Directory to save extracted frames
    :param step: Step interval for frame extraction
    """
    # Initialize video decoder and read video
    decoder = VideoDecoder(video_path)
    total_num_frames = decoder.get_frame_count()

    # Get frame indices based on step
    frame_indices = get_seq_frames(total_num_frames, step)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for idx in frame_indices:
        frame = decoder.decode(idx)
        save_image(frame, os.path.join(output_dir, f"frame_{idx}.png"))

    print(f"Frames saved to {output_dir}")
```

## How to Customize

- **Step Interval**: You can change the interval for frame extraction by specifying the `--step` argument in the command line. For instance, `--step 10` will extract every 10th frame instead of 16.
- **Output Format**: Currently, frames are saved as PNG images using `save_image` from the `torchvision` library. You can easily change the output format to another image type if needed.

## Example Output

The frames will be saved in the specified output directory with filenames like:

```
frame_0.png
frame_16.png
frame_32.png
...
```

## Contributing

Feel free to fork the repository and submit pull requests for improvements or new features. Make sure to follow the code style and include tests for any new functionality you add.

## License

This project is licensed under the MIT License.

---

This `README.md` should give users a clear understanding of how to use the code and how the extraction logic works, as well as how to customize the frame extraction step through command-line arguments.
