import os
import sys
import cv2
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

    screenshots_dir = f"{video_name}_screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    capture_screenshots(video_path, video_name, screenshots_dir)

def detect_content_region(frame):
    """Detect the main content region in a Webex recording."""
    frame_height, frame_width = frame.shape[:2]
    
    # Default to using most of the frame if detection fails
    default_region = (
        int(frame_width * 0.1),  # x: 10% from left
        int(frame_height * 0.1), # y: 10% from top
        int(frame_width * 0.8),  # width: 80% of frame width
        int(frame_height * 0.8)  # height: 80% of frame height
    )
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found, using default region")
            return default_region
        
        # Find the largest contour
        max_area = 0
        content_region = default_region
        min_area = (frame_width * frame_height) * 0.1  # Reduced to 10% minimum
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > max_area and area > min_area:
                aspect_ratio = w / h
                # Relaxed aspect ratio constraints
                if 0.5 <= aspect_ratio <= 3.0:
                    max_area = area
                    content_region = (x, y, w, h)
        
        return content_region
    except Exception as e:
        print(f"Error in detect_content_region: {e}")
        return default_region

def calculate_frame_similarity(frame1, frame2, content_region):
    """Calculate structural similarity between two frames within content region."""
    try:
        x, y, w, h = content_region
        roi1 = frame1[y:y+h, x:x+w]
        roi2 = frame2[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((gray1 - gray2) ** 2)
        
        # Calculate similarity score (inverse of MSE, normalized)
        similarity = 1 / (1 + mse)
        return similarity
    except Exception as e:
        print(f"Error in calculate_frame_similarity: {e}")
        return 1.0

def is_valid_content_frame(frame, content_region, min_content_threshold=0.05):
    """Check if frame has enough content in the content region."""
    try:
        x, y, w, h = content_region
        roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate the standard deviation of pixel values
        std_dev = np.std(gray)
        
        # Calculate the mean of pixel values
        mean_value = np.mean(gray)
        
        # Relaxed criteria
        return (std_dev > 15 and  # Reduced from 25
                10 < mean_value < 245)  # Wider range
    except Exception as e:
        print(f"Error in is_valid_content_frame: {e}")
        return True

def capture_screenshots(video_path, video_name, output_dir):
    print(f"Capturing screenshots from {video_name}...")
    
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Parameters - more relaxed now
    min_time_between_captures = 2  # Reduced from 3
    frame_interval = int(fps)  # Check every second
    min_similarity_threshold = 0.95  # Relaxed threshold
    
    frame_count = 0
    image_count = 0
    prev_frame = None
    content_region = None
    last_capture_time = -min_time_between_captures
    
    while True:
        success, frame = vidcap.read()
        if not success:
            break
            
        current_time = frame_count / fps
        
        # Only process frames at the specified interval
        if frame_count % frame_interval == 0:
            # Update content region periodically
            if frame_count % (frame_interval * 30) == 0 or content_region is None:
                content_region = detect_content_region(frame)
                print(f"Content region updated: {content_region}")
            
            # Check if enough time has passed since last capture
            if current_time - last_capture_time >= min_time_between_captures:
                # Check frame validity
                valid_frame = is_valid_content_frame(frame, content_region)
                print(f"Frame {frame_count} valid: {valid_frame}")
                
                if valid_frame:
                    # Check similarity with previous frame
                    should_capture = True
                    if prev_frame is not None:
                        similarity = calculate_frame_similarity(frame, prev_frame, content_region)
                        print(f"Frame {frame_count} similarity: {similarity:.3f}")
                        should_capture = similarity < min_similarity_threshold
                    
                    if should_capture:
                        # Format timestamp
                        timestamp = str(timedelta(seconds=int(current_time)))
                        image_path = os.path.join(
                            output_dir, 
                            f"{video_name}_slide_{image_count:03d}_{timestamp}.png"
                        )
                        
                        # Save frame
                        try:
                            cv2.imwrite(image_path, frame)
                            print(f"Successfully captured slide {image_count} at {timestamp}")
                            image_count += 1
                            last_capture_time = current_time
                            prev_frame = frame.copy()
                        except Exception as e:
                            print(f"Error saving image: {e}")
        
        frame_count += 1
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
