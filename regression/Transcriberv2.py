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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest rectangular contour (likely the content pane)
    max_area = 0
    content_region = None
    frame_height, frame_width = frame.shape[:2]
    min_area = (frame_width * frame_height) * 0.2  # Minimum 20% of frame
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area and area > min_area:
            aspect_ratio = w / h
            # Check if it's reasonably rectangular (typical aspect ratios for content)
            if 1.0 <= aspect_ratio <= 2.0:
                max_area = area
                content_region = (x, y, w, h)
    
    return content_region

def calculate_frame_similarity(frame1, frame2, content_region):
    """Calculate structural similarity between two frames within content region."""
    if content_region is None:
        return 1.0  # Return high similarity if no content region detected
    
    x, y, w, h = content_region
    roi1 = frame1[y:y+h, x:x+w]
    roi2 = frame2[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # Compare histograms using correlation
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Also calculate structural similarity
    mse = np.mean((gray1 - gray2) ** 2)
    structural_sim = 1 / (1 + mse)
    
    # Combine both metrics
    combined_similarity = (similarity + structural_sim) / 2
    return combined_similarity

def is_valid_content_frame(frame, content_region, min_content_threshold=0.15):
    """Check if frame has enough content in the content region."""
    if content_region is None:
        return False
        
    x, y, w, h = content_region
    roi = frame[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate the standard deviation of pixel values
    std_dev = np.std(gray)
    
    # Calculate the mean of pixel values
    mean_value = np.mean(gray)
    
    # Calculate edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / (w * h)
    
    # Check multiple criteria
    return (std_dev > 25 and  # Has variation in pixel values
            20 < mean_value < 235 and  # Not too bright or dark
            edge_density > min_content_threshold)  # Has sufficient edge content

def capture_screenshots(video_path, video_name, output_dir):
    print(f"Capturing screenshots from {video_name}...")
    
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Parameters
    min_time_between_captures = 3  # Minimum seconds between captures
    frame_interval = int(fps)  # Check every second
    min_similarity_threshold = 0.85  # Increased threshold for detecting changes
    
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
            # Detect content region periodically
            if frame_count % (frame_interval * 10) == 0:
                content_region = detect_content_region(frame)
            
            # Check if enough time has passed since last capture
            if current_time - last_capture_time >= min_time_between_captures:
                if content_region and is_valid_content_frame(frame, content_region):
                    if prev_frame is None or calculate_frame_similarity(
                        frame, prev_frame, content_region) < min_similarity_threshold:
                        
                        # Format timestamp
                        timestamp = str(timedelta(seconds=int(current_time)))
                        image_path = os.path.join(
                            output_dir, 
                            f"{video_name}_slide_{image_count:03d}_{timestamp}.png"
                        )
                        
                        # Save both full frame and content region
                        cv2.imwrite(image_path, frame)
                        
                        print(f"Captured slide {image_count} at {timestamp}")
                        image_count += 1
                        last_capture_time = current_time
                        prev_frame = frame.copy()
        
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
