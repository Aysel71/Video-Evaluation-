# 1. Create a new version of the script with GPU fixes
cat > caption_script_gpu.py << 'EOL'
#!/usr/bin/env python3
# Batch captioning script for all videos in the dataset
# Creates a separate "captions" folder for each video

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm  # Use regular tqdm instead of notebook version
import json
import torch

# Install the required packages if not already installed
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except ImportError:
    print("Installing required packages...")
    import os
    os.system('pip install transformers -q')
    os.system('pip install accelerate -q')
    from transformers import BlipProcessor, BlipForConditionalGeneration

def display_image_with_caption(image, caption, timestamp):
    """Save image with its caption instead of displaying"""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Time: {timestamp}\nCaption: {caption}", fontsize=14)
    plt.tight_layout()
    # Save instead of showing
    os.makedirs('caption_previews', exist_ok=True)
    plt.savefig(f"caption_previews/{timestamp}_preview.jpg")
    plt.close()

def process_single_video(video_id, root_dir, processor, model, device, max_frames=None, display_images=False):
    """Process all frames for a single video"""
    video_dir = os.path.join(root_dir, video_id)
    frames_dir = os.path.join(video_dir, "frames")
    
    # Create a dedicated captions directory
    captions_dir = os.path.join(video_dir, "captions")
    os.makedirs(captions_dir, exist_ok=True)
    
    print(f"Processing video: {video_id}")
    
    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found for {video_id}")
        return {}
    
    # List all jpg files
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    print(f"  Found {len(frames)} frames")
    
    # Limit number of frames if specified
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
        print(f"  Processing first {max_frames} frames")
    
    # Process each frame
    captions_data = {}
    
    for frame_file in tqdm(frames, desc=f"Captioning {video_id}"):
        frame_path = os.path.join(frames_dir, frame_file)
        
        # Load and process image
        raw_image = Image.open(frame_path).convert('RGB')
        
        # Extract timestamp from filename
        timestamp = frame_file.split('_frame_')[1].split('.jpg')[0]
        
        # Generate caption
        inputs = processor(raw_image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        
        # Display the result if requested
        if display_images:
            display_image_with_caption(raw_image, caption, timestamp)
        
        # Store caption
        captions_data[frame_file] = {
            "timestamp": timestamp,
            "caption": caption
        }
        
        # Save caption to file in the captions directory
        caption_file = os.path.join(captions_dir, f"{video_id}_frame_{timestamp}_caption.txt")
        with open(caption_file, 'w') as f:
            f.write(caption)
    
    # Save all captions to JSON
    json_path = os.path.join(video_dir, f"{video_id}_captions.json")
    with open(json_path, 'w') as f:
        json.dump(captions_data, f, indent=4)
      
    print(f"Completed captioning for {video_id}")
    return captions_data

def batch_process_videos(root_dir, specific_videos=None, max_frames=None, display_images=False):
    """Process multiple videos in batch"""
    print(f"Processing videos from: {root_dir}")
    
    # Set device with explicit check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Try to optimize CUDA performance
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: GPU not available, using CPU. This will be much slower.")
    
    # Load BLIP model (only once for efficiency)
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    print("Model loaded successfully")
    
    # Get list of video directories
    if specific_videos:
        # Process only the specified videos
        video_ids = [vid for vid in specific_videos if os.path.exists(os.path.join(root_dir, vid))]
    else:
        # Process all directories in the root directory
        video_ids = [d for d in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir, d)) and 
                    os.path.exists(os.path.join(root_dir, d, "frames"))]
    
    print(f"Found {len(video_ids)} videos to process")
    
    # Process each video
    results = {}
    for video_id in video_ids:
        try:
            video_captions = process_single_video(
                video_id, 
                root_dir, 
                processor, 
                model, 
                device, 
                max_frames, 
                display_images
            )
            results[video_id] = len(video_captions)
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")
    
    # Print summary
    print("\n=== Captioning Summary ===")
    for video_id, count in results.items():
        print(f"Video {video_id}: {count} frames captioned")
    
    print("\nCaptioning completed successfully!")
    return results

# Parameters for processing videos
ROOT_DIR = "/workspace/data/Video-MME/output"
SPECIFIC_VIDEOS = None  # Set to None to process all videos
MAX_FRAMES = None  # Set to None to process all frames or a number to limit
DISPLAY_IMAGES = False  # Set to True to display images (not recommended for batch processing)

# Main execution
if __name__ == "__main__":
    print("Starting batch captioning process...")
    results = batch_process_videos(ROOT_DIR, SPECIFIC_VIDEOS, MAX_FRAMES, DISPLAY_IMAGES)
    print("Process complete!")
EOL
