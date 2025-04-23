
cat > eval.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import numpy as np
import requests
import re
import time
import glob
from tqdm import tqdm

def load_video_captions(video_id, captions_base_path):
    """
    Load all captions for a specific video ID from the captions folder
    """
    potential_paths = [
        os.path.join(captions_base_path, video_id, "captions"),
        os.path.join(captions_base_path, video_id),
        captions_base_path
    ]
    
    captions = {}
    
    for path in potential_paths:
        # Find caption files
        caption_files = glob.glob(os.path.join(path, f"{video_id}_frame_*_caption.txt"))
        
        if not caption_files:
            caption_files = glob.glob(os.path.join(path, "*_frame_*_caption.txt"))
        
        # Load captions from found files
        for caption_file in caption_files:
            try:
                # Extract timestamp
                timestamp_match = re.search(r'frame_([^_]+)_caption', os.path.basename(caption_file))
                timestamp = timestamp_match.group(1) if timestamp_match else os.path.basename(caption_file)
                
                # Read caption
                with open(caption_file, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
                
                if caption_text:
                    captions[timestamp] = caption_text
            except Exception as e:
                print(f"Error loading caption from {caption_file}: {e}")
        
        # Stop searching if captions found
        if captions:
            break
    
    print(f"Video {video_id}: Loaded {len(captions)} captions")
    return captions

def format_captions_context(captions):
    """
    Format video captions as context for the model
    """
    if not captions:
        return ""
    
    # Sort captions by timestamp
    try:
        sorted_timestamps = sorted(
            captions.keys(), 
            key=lambda x: [int(t) if t.isdigit() else t for t in x.split(':')]
        )
    except Exception:
        sorted_timestamps = list(captions.keys())
    
    # Create context string
    context = "Video scene descriptions:\n"
    for timestamp in sorted_timestamps:
        context += f"Time {timestamp}: {captions[timestamp]}\n"
    
    return context

def create_qa_prompt_with_captions(question, options, captions):
    """
    Create a prompt that includes video captions as context
    """
    context = format_captions_context(captions)
    
    prompt = f"{context}\n\nBased on the video scenes described above, please answer the following question:\n\nQ: {question}\n"
    
    options_letters = ['A', 'B', 'C', 'D']
    for i, option_text in enumerate(options):
        if i < len(options_letters):
            letter = options_letters[i]
            prompt += f"{letter}. {option_text}\n"
    
    prompt += "\nThe correct answer is:"
    
    return prompt

def query_model_with_logprobs(prompt, api_key, retry_count=3, retry_delay=1):
    url = "https://api.fireworks.ai/inference/v1/completions"

    payload = {
        "model": "accounts/fireworks/models/deepseek-v3",
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": 40,
        "logprobs": 5
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for attempt in range(retry_count):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error for logprobs (attempt {attempt+1}/{retry_count}): {e}")
            if hasattr(response, 'text'):
                print(f"Server response: {response.text}")
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
            else:
                print("All API request attempts failed")
                return None

def extract_answer_probs(response):
    """
    Extract probabilities for answer options A, B, C, D
    """
    if not response or 'choices' not in response or not response['choices']:
        return None, None, None

    try:
        # Get top tokens and their probabilities
        logprobs_obj = response['choices'][0].get('logprobs', {})
        if not logprobs_obj or 'top_logprobs' not in logprobs_obj or not logprobs_obj['top_logprobs']:
            return None, None, None

        top_tokens = logprobs_obj['top_logprobs'][0]  # First (and only) token

        # Extract probabilities for answer options (A, B, C, D)
        answer_probs = {}
        for letter in ['A', 'B', 'C', 'D']:
            # Search for the letter among top tokens
            prob = None
            for token, logprob in top_tokens.items():
                token_clean = token.strip().upper()
                if token_clean == letter or token_clean.startswith(letter + "."):
                    prob = np.exp(logprob)  # Convert log probability to probability
                    break
            answer_probs[letter] = prob

        # Find the answer with the highest probability
        valid_probs = {k: v for k, v in answer_probs.items() if v is not None}
        if valid_probs:
            max_prob_letter = max(valid_probs.items(), key=lambda x: x[1])[0]
            max_prob = valid_probs[max_prob_letter]
            return max_prob_letter, max_prob, answer_probs
        else:
            return None, None, answer_probs
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error extracting probabilities: {e}")
        return None, None, None

def load_data(file_path):
    """
    Load data from a Parquet file
    """
    print(f"Loading data from: {file_path}")
    try:
        data = pd.read_parquet(file_path)
        print(f"Loaded {len(data)} data rows")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def process_single_video(parquet_file, captions_base_path, api_key, video_id=None):
    """
    Process a single video to see inputs and outputs
    """
    # Load data
    data = load_data(parquet_file)
    if data is None:
        return None
    
    # If no video_id specified, take the first one
    if video_id is None:
        video_id = data['videoID'].iloc[0]
    
    # Filter data for this video
    video_data = data[data['videoID'] == video_id]
    print(f"\nProcessing video: {video_id}")
    print(f"Number of questions: {len(video_data)}")
    
    # Get video info
    first_row = video_data.iloc[0]
    print(f"Duration: {first_row.get('duration', '')}")
    print(f"Domain: {first_row.get('domain', '')}")
    print(f"Sub Category: {first_row.get('sub_category', '')}")
    print(f"URL: {first_row.get('url', '')}")
    
    # Load captions
    captions = load_video_captions(video_id, captions_base_path)
    print(f"\nLoaded {len(captions)} captions for this video")
    
    # Print first few captions as example
    print("\nExample captions:")
    caption_keys = list(captions.keys())[:3]
    for key in caption_keys:
        print(f"Time {key}: {captions[key][:100]}...")
    
    print("\n" + "="*50)
    print("Processing each question:")
    print("="*50)
    
    # Process each question for this video
    for idx, row in video_data.iterrows():
        print(f"\nQuestion {idx + 1}:")
        print(f"Question ID: {row.get('question_id', '')}")
        print(f"Question: {row.get('question', '')}")
        print(f"Options: {row.get('options', [])}")
        print(f"Correct Answer: {row.get('answer', '')}")
        print(f"Task Type: {row.get('task_type', '')}")
        
        # Create prompt
        prompt = create_qa_prompt_with_captions(
            row.get('question', ''),
            row.get('options', []),
            captions
        )
        
        print("\n--- FULL PROMPT SENT TO DEEPSEEK ---")
        print(prompt)
        print("--- END OF PROMPT ---\n")
        
        # Query the model
        api_response = query_model_with_logprobs(prompt, api_key)
        
        # Extract answer
        predicted_letter, max_prob, answer_probs = extract_answer_probs(api_response)
        
        print("\n--- MODEL RESPONSE ---")
        print(f"Full API Response: {json.dumps(api_response, indent=2)}")
        print(f"Predicted Answer: {predicted_letter}")
        print(f"Confidence: {max_prob}")
        print(f"All probabilities: {answer_probs}")
        print("--- END OF RESPONSE ---\n")
        
        print("="*50)

# Main execution
if __name__ == "__main__":
    # Potential Parquet file paths
    PARQUET_FILE_PATHS = [
        "/workspace/data/Video-MME/videomme/test-00000-of-00001.parquet",
        "/mnt/public-datasets/a.mirzoeva/Video-MME/videomme/test-00000-of-00001.parquet",
        "/workspace/InternVideo/Data/InternVid/test-00000-of-00001.parquet"
    ]

    # Potential caption base paths
    CAPTIONS_BASE_PATHS = [
        "/workspace/data/Video-MME/output",
        "/mnt/public-datasets/a.mirzoeva/Video-MME/captions",
        "/workspace/InternVideo/Data/captions"
    ]

    # API key
    API_KEY = "fw_3ZTYfDqnj6VXpbMAutpHmzVG"

    # Try each Parquet file path
    parquet_file = None
    for path in PARQUET_FILE_PATHS:
        if os.path.exists(path):
            parquet_file = path
            break

    if not parquet_file:
        print("No Parquet file found. Exiting.")
        sys.exit(1)

    # Try each caption base path
    captions_base_path = None
    for path in CAPTIONS_BASE_PATHS:
        if os.path.exists(path):
            captions_base_path = path
            break
    
    if not captions_base_path:
        print("No captions base path found. Exiting.")
        sys.exit(1)

    # Process a single video
    process_single_video(
        parquet_file=parquet_file,
        captions_base_path=captions_base_path,
        api_key=API_KEY,
        video_id=None  # Set to None to use the first video, or specify a specific video ID
    )
PYTHON_SCRIPT
