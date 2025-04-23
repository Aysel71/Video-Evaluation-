cat > run_evaluation_logprobs.py << 'PYTHON_SCRIPT'
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union

# Install required packages if not available
try:
    import pyarrow
except ImportError:
    print("Installing pyarrow...")
    os.system("pip install pyarrow")
    import pyarrow

# Categories and constants remain the same as in previous scripts
CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

def load_video_captions(video_id, captions_base_path):
    """
    Load all captions for a specific video ID from the captions folder
    """
    # Potential base paths
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

def process_question_with_captions(question_data, video_id, captions, api_key):
    """
    Process a single question with video captions as context using logprobs
    """
    question_id = question_data.get('question_id', '')
    question_text = question_data.get('question', '')
    options = question_data.get('options', [])
    correct_answer = question_data.get('answer', '')
    task_type = question_data.get('task_type', '')
    
    # Create prompt with captions
    prompt = create_qa_prompt_with_captions(question_text, options, captions)
    
    # Query the model with logprobs
    api_response = query_model_with_logprobs(prompt, api_key)
    
    # Extract probabilities
    predicted_letter, max_prob, answer_probs = extract_answer_probs(api_response)
    
    # Format model response
    if predicted_letter:
        model_response = f"The correct answer is: {predicted_letter} (Probability: {max_prob:.4f})"
    else:
        model_response = ""
    
    # Create result
    result_question = {
        'question_id': question_id,
        'question': question_text,
        'options': options.tolist() if isinstance(options, np.ndarray) else options,
        'answer': correct_answer,
        'task_type': task_type,
        'response': model_response,
        'predicted_letter': predicted_letter,
        'answer_probabilities': answer_probs,
        'max_probability': max_prob,
        'prompt': prompt  # Include prompt for debugging
    }
    
    return result_question

def process_dataset_with_captions(data, captions_base_path, api_key, output_file, max_samples=None, workers=1):
    """
    Process the entire dataset with video captions as context
    """
    # Create structure for results
    results = []
    
    # Group data by videoID
    video_groups = data.groupby('videoID')
    
    # Limit samples if specified
    if max_samples and max_samples > 0:
        video_ids = list(video_groups.groups.keys())[:max_samples]
    else:
        video_ids = list(video_groups.groups.keys())
    
    print(f"Processing {len(video_ids)} videos with captions...")
    
    # Process videos
    for video_idx, video_id in enumerate(tqdm(video_ids, desc="Processing videos")):
        video_data = video_groups.get_group(video_id)
        
        # Get video info from first row
        first_row = video_data.iloc[0]
        
        # Create video result object
        video_result = {
            'video_id': first_row.get('videoID', ''),
            'duration': first_row.get('duration', ''),
            'domain': first_row.get('domain', ''),
            'sub_category': first_row.get('sub_category', ''),
            'url': first_row.get('url', ''),
            'videoID': video_id,
            'missing': False,
            'questions': []
        }
        
        # Load captions for this video
        captions = load_video_captions(video_id, captions_base_path)
        caption_count = len(captions)
        print(f"Video {video_id}: Loaded {caption_count} captions")
        
        # Collect questions for this video
        questions_data = []
        for _, row in video_data.iterrows():
            question_data = {
                'question_id': row.get('question_id', ''),
                'question': row.get('question', ''),
                'options': row.get('options', []),
                'answer': row.get('answer', ''),
                'task_type': row.get('task_type', '')
            }
            questions_data.append(question_data)
        
        # Process questions (parallel or sequential)
        if workers > 1 and len(questions_data) > 1:
            processed_questions = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(
                    process_question_with_captions, q, video_id, captions, api_key
                ): q for q in questions_data}
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc=f"Processing questions for video {video_idx+1}/{len(video_ids)}"):
                    result = future.result()
                    if result:
                        processed_questions.append(result)
        else:
            processed_questions = []
            for q in tqdm(questions_data, 
                         desc=f"Processing questions for video {video_idx+1}/{len(video_ids)}"):
                result = process_question_with_captions(q, video_id, captions, api_key)
                if result:
                    processed_questions.append(result)
                # Add delay to avoid API rate limits
                time.sleep(1.0)
        
        # Format questions for evaluation
        formatted_questions = []
        for q in processed_questions:
            formatted_q = {
                'question_id': q['question_id'],
                'question': q['question'],
                'options': q['options'],
                'answer': q['answer'],
                'task_type': q['task_type'],
                'response': q['response'] if 'response' in q and q['response'] else ''
            }
            formatted_questions.append(formatted_q)
        
        # Add processed questions to video result
        video_result['questions'] = formatted_questions
        
        # Add video result to overall results
        results.append(video_result)
        
        # Save intermediate results
        if (video_idx + 1) % 5 == 0 or video_idx == len(video_ids) - 1:
            with open(output_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Intermediate results saved to {output_file}")
    
    # Save final results
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")
    
    return results

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

def run_video_qa_with_captions(
    parquet_file,
    captions_base_path,
    api_key,
    output_file="model_results_with_captions.json",
    max_samples=None,
    workers=1
):
    """
    Run the video QA evaluation with captions integration
    """
    # Load data
    data = load_data(parquet_file)
    if data is None:
        return None
    
    # Process data with captions
    print("=== Starting data processing with captions ===")
    results = process_dataset_with_captions(
        data=data,
        captions_base_path=captions_base_path,
        api_key=api_key,
        output_file=output_file,
        max_samples=max_samples,
        workers=workers
    )
    
    return results

# Function to evaluate results
def eval_your_results(
        your_results_path,
        video_types,
        skip_missing=False,
        return_categories_accuracy=True,
        return_sub_categories_accuracy=False,
        return_task_types_accuracy=False,
        gt_answer_key="answer",
        your_answer_key="response"
    ):
    """
    Evaluate your results against the ground truth
    """
    # Check if results file exists
    if not os.path.exists(your_results_path):
        print(f"Results file not found: {your_results_path}")
        return None
    
    # Load your results
    with open(your_results_path, 'r') as f:
        your_results = json.load(f)

    if isinstance(video_types, str):
        video_types = video_types.split(",")

    q_type_dict = {}
    v_type_dict = {}
    v_sub_type_dict = {}

    for video_type in video_types:
        # Filter your results based on video types
        your_results_video_type = [item for item in your_results if item["duration"] == video_type]

        # Task Categories
        q_type_dict[video_type] = {}
        for q_type in TASK_CATEGORIES:
            q_type_dict[video_type][q_type] = {"correct": 0, "answered": 0}

        # Video categories
        v_type_dict[video_type] = {}
        for v_type in CATEGORIES:
            v_type_dict[video_type][v_type] = {"correct": 0, "answered": 0}

        v_sub_type_dict[video_type] = {}
        for v_sub_type in SUB_CATEGORIES:
            v_sub_type_dict[video_type][v_sub_type] = {"correct": 0, "answered": 0}

        if not skip_missing:
            # Check if the number of files in your results and ground truth are the same
            if len(your_results_video_type) != 300:
                print(f"Warning: Number of videos of type {video_type} is {len(your_results_video_type)}, expected 300.")
                if not skip_missing:
                    print(f"Set skip_missing=True to ignore this warning.")

        for item in your_results_video_type:
            if skip_missing and item["missing"]:
                continue

            # Get the video category, sub category and question category
            video_category = item["domain"]
            video_sub_category = item["sub_category"]

            questions = item["questions"]

            for question in questions:
                q_type = question["task_type"]

                # Get the ground truth and your response
                gt_answer = question[gt_answer_key]
                response = question[your_answer_key]

                # Extract the answer from the response
                extracted = extract_characters_regex(response)

                if extracted != "":
                    q_type_dict[video_type][q_type]["answered"] += 1
                    q_type_dict[video_type][q_type]["correct"] += extracted == gt_answer

                    v_type_dict[video_type][video_category]["answered"] += 1
                    v_type_dict[video_type][video_category]["correct"] += extracted == gt_answer

                    v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
                    v_sub_type_dict[video_type][video_sub_category]["correct"] += extracted == gt_answer

    # Print the results for each video type
    for video_type in video_types:
        print("=====================================")
        print(f"Evaluation on video Type: {video_type}")
        print("=====================================")
        if return_categories_accuracy:
            print("-------------------------------------")
            print("Video Categories")
            print("-------------------------------------")
            for v_type in v_type_dict[video_type]:
                print(f"{v_type}: {100 * v_type_dict[video_type][v_type]['correct'] / v_type_dict[video_type][v_type]['answered'] if v_type_dict[video_type][v_type]['answered'] > 0 else 0 : .1f}%")
        if return_sub_categories_accuracy:
            print("-------------------------------------")
            print("Video Sub Categories")
            print("-------------------------------------")
            for v_sub_type in v_sub_type_dict[video_type]:
                print(f"{v_sub_type}: {100 * v_sub_type_dict[video_type][v_sub_type]['correct'] / v_sub_type_dict[video_type][v_sub_type]['answered'] if v_sub_type_dict[video_type][v_sub_type]['answered'] > 0 else 0 : .1f}%")
        if return_task_types_accuracy:
            print("-------------------------------------")
            print("Task Categories")
            print("-------------------------------------")
            for q_type in q_type_dict[video_type]:
                print(f"{q_type}: {100 * q_type_dict[video_type][q_type]['correct'] / q_type_dict[video_type][q_type]['answered'] if q_type_dict[video_type][q_type]['answered'] > 0 else 0 : .1f}%")

        print("-------------------------------------")
        print("Overall Performance")
        print("-------------------------------------")
        total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES])
        total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES])
        print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")
        print("\n")

    # Print the results for the entire dataset
    print("=====================================")
    print("Evaluation on the entire dataset")
    print("=====================================")

    if return_categories_accuracy:
        print("-------------------------------------")
        print("Video Domains")
        print("-------------------------------------")
        for v_type in CATEGORIES:
            total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
            total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
            print(f"{v_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    if return_sub_categories_accuracy:
        print("-------------------------------------")
        print("Video Sub Categories")
        print("-------------------------------------")
        for v_sub_type in SUB_CATEGORIES:
            total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
            total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
            print(f"{v_sub_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    if return_task_types_accuracy:
        print("-------------------------------------")
        print("Task Categories")
        print("-------------------------------------")
        for q_type in TASK_CATEGORIES:
            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
            print(f"{q_type}: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    print("-------------------------------------")
    print("Overall Performance")
    print("-------------------------------------")
    total_correct = sum([sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
    total_answered = sum([sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
    print(f"Overall: {100 * total_correct / total_answered if total_answered > 0 else 0 : .1f}%")

    # Return metrics for further use
    return {
        "total_correct": total_correct,
        "total_answered": total_answered,
        "accuracy": total_correct / total_answered if total_answered > 0 else 0
    }

# Run evaluation
if __name__ == "__main__":
    print("Starting evaluation...")
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

    # API key (ensure this is your actual API key)
    API_KEY = "API_KEY"
    
    # Output file for results
    OUTPUT_FILE = "video_qa_results_with_logprobs.json"

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
    results = None
    for captions_base_path in CAPTIONS_BASE_PATHS:
        try:
            print(f"Attempting to process with captions base path: {captions_base_path}")
            
            # Run QA with captions
            results = run_video_qa_with_captions(
                parquet_file=parquet_file,
                captions_base_path=captions_base_path,
                api_key=API_KEY,
                output_file=OUTPUT_FILE,
                max_samples=None,  # Set to a number like 10 for testing
                workers=1  # Adjust based on your system
            )
            
            # If results are successful, break the loop
            if results:
                break
        except Exception as e:
            print(f"Error processing with base path {captions_base_path}: {e}")

    # Evaluate results
    if results:
        try:
            metrics = eval_your_results(
                your_results_path=OUTPUT_FILE,
                video_types="short,medium,long",
                skip_missing=True,
                return_categories_accuracy=True,
                return_sub_categories_accuracy=True,
                return_task_types_accuracy=True
            )
            
            # Print detailed metrics
            print("\n=== Evaluation Metrics ===")
            print(f"Total Correct: {metrics['total_correct']}")
            print(f"Total Answered: {metrics['total_answered']}")
            print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
    else:
        print("Processing failed, cannot evaluate results.")

    print("Script completed!")
PYTHON_SCRIPT
