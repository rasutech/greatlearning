import os
import sys
import moviepy.editor as mp
import requests
import cv2
import whisper
import numpy as np
from datetime import timedelta

def main(input_path):
    if os.path.isfile(input_path):
        process_video(input_path)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(input_path, filename)
                process_video(video_path)
    else:
        print("Invalid input path")

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video: {video_name}")

    # Create output directory for screenshots
    screenshots_dir = f"{video_name}_screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)

    # Extract audio and generate transcript
    audio_path = extract_audio(video_path, video_name)
    transcript = generate_transcript(audio_path)
    enhanced_transcript = enhance_transcript(transcript)
    save_transcript(enhanced_transcript, video_name)
    
    # Capture smart screenshots
    capture_screenshots(video_path, video_name, screenshots_dir)

def extract_audio(video_path, video_name):
    video = mp.VideoFileClip(video_path)
    audio_path = f"{video_name}_audio.wav"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

def generate_transcript(audio_path):
    model = whisper.load_model("base")
    print("Transcribing audio with Whisper...")
    result = model.transcribe(audio_path)
    return result["text"]

def enhance_transcript(transcript):
    api_endpoint = "https://api.yourorganization.com/llm"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"
    }
    data = {
        "prompt": f"Enhance the following transcript for sentence formation and accuracy:\n\n{transcript}"
    }
    try:
        response = requests.post(api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('enhanced_text', '')
    except requests.exceptions.RequestException as e:
        print("Error enhancing transcript:", e)
        return transcript

def save_transcript(enhanced_transcript, video_name):
    transcript_path = f"{video_name}_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(enhanced_transcript)
    print(f"Transcript saved to {transcript_path}")

def calculate_frame_similarity(frame1, frame2):
    """Calculate structural similarity between two frames."""
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((gray1 - gray2) ** 2)
    
    # Calculate similarity score (inverse of MSE, normalized)
    similarity = 1 / (1 + mse)
    return similarity

def detect_slide_change(current_frame, prev_frame, min_similarity_threshold=0.95):
    """Detect if there's a significant change between frames (slide change)."""
    if prev_frame is None:
        return True
        
    similarity = calculate_frame_similarity(current_frame, prev_frame)
    return similarity < min_similarity_threshold

def is_valid_content_frame(frame, min_content_threshold=0.1):
    """Check if frame has enough content (not blank or nearly blank)."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the standard deviation of pixel values
    std_dev = np.std(gray)
    
    # Calculate the mean of pixel values
    mean_value = np.mean(gray)
    
    # Check if frame has enough variation and isn't too bright or dark
    return std_dev > 20 and 20 < mean_value < 235

def capture_screenshots(video_path, video_name, output_dir):
    print(f"Capturing screenshots from {video_name}...")
    
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Parameters
    min_time_between_captures = 2  # Minimum seconds between captures
    frame_interval = int(fps * 0.5)  # Check every half second
    min_similarity_threshold = 0.95  # Threshold for detecting changes
    
    frame_count = 0
    image_count = 0
    prev_frame = None
    last_capture_time = -min_time_between_captures
    
    while True:
        success, frame = vidcap.read()
        if not success:
            break
            
        current_time = frame_count / fps
        
        # Only process frames at the specified interval
        if frame_count % frame_interval == 0:
            # Check if enough time has passed since last capture
            if current_time - last_capture_time >= min_time_between_captures:
                # Detect if frame is a valid content frame and represents a slide change
                if (is_valid_content_frame(frame) and 
                    detect_slide_change(frame, prev_frame, min_similarity_threshold)):
                    
                    # Format timestamp for filename
                    timestamp = str(timedelta(seconds=int(current_time)))
                    image_path = os.path.join(
                        output_dir, 
                        f"{video_name}_slide_{image_count:03d}_{timestamp}.png"
                    )
                    
                    cv2.imwrite(image_path, frame)
                    print(f"Captured slide {image_count} at {timestamp}")
                    
                    image_count += 1
                    last_capture_time = current_time
                    prev_frame = frame.copy()
        
        frame_count += 1
        
        # Print progress every 1000 frames
        if frame_count % 1000 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Processing: {progress:.1f}% complete")
    
    vidcap.release()
    print(f"Captured {image_count} unique slides from {video_name}")
    print(f"Screenshots saved in: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_file_or_directory>")
    else:
        main(sys.argv[1])
