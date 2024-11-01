import os
import sys
import moviepy.editor as mp
from vosk import Model, KaldiRecognizer
import wave
import json
import requests
import cv2

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

    # Step 1: Extract audio from video
    audio_path = extract_audio(video_path, video_name)

    # Step 2: Generate transcript using Vosk
    transcript = generate_transcript(audio_path)

    # Step 3: Enhance transcript using API
    enhanced_transcript = enhance_transcript(transcript)

    # Step 4: Save enhanced transcript
    save_transcript(enhanced_transcript, video_name)

    # Step 5: Capture screenshots
    capture_screenshots(video_path, video_name)

def extract_audio(video_path, video_name):
    video = mp.VideoFileClip(video_path)
    audio_path = f"{video_name}_audio.wav"
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return audio_path

def generate_transcript(audio_path):
    # Ensure the Vosk model is downloaded and placed in the 'model' directory
    if not os.path.exists("model"):
        print("Please download the Vosk model and place it in the 'model' directory.")
        sys.exit(1)
    model = Model("model")
    wf = wave.open(audio_path, "rb")

    if wf.getnchannels() != 1:
        print("Converting audio to mono channel...")
        mono_audio_path = audio_path.replace(".wav", "_mono.wav")
        os.system(f"ffmpeg -i {audio_path} -ac 1 {mono_audio_path}")
        wf = wave.open(mono_audio_path, "rb")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    transcript = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            transcript += res.get('text', '') + ' '
    res = json.loads(rec.FinalResult())
    transcript += res.get('text', '')
    wf.close()
    return transcript

def enhance_transcript(transcript):
    # Replace with your actual API endpoint and parameters
    api_endpoint = "https://api.yourorganization.com/llm"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"  # Replace with your actual API key
    }
    data = {
        "prompt": f"Enhance the following transcript for sentence formation and accuracy:\n\n{transcript}"
    }
    try:
        response = requests.post(api_endpoint, headers=headers, json=data)
        response.raise_for_status()
        enhanced_transcript = response.json().get('enhanced_text', '')
        return enhanced_transcript
    except requests.exceptions.RequestException as e:
        print("Error enhancing transcript:", e)
        return transcript  # Return original if enhancement fails

def save_transcript(enhanced_transcript, video_name):
    transcript_path = f"{video_name}_transcript.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(enhanced_transcript)
    print(f"Transcript saved to {transcript_path}")

def capture_screenshots(video_path, video_name):
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 5)  # Capture a frame every 5 seconds
    frame_count = 0
    image_count = 0
    success, image = vidcap.read()

    while success:
        if frame_count % frame_interval == 0:
            image_path = f"{video_name}_frame_{image_count}.png"
            cv2.imwrite(image_path, image)
            image_count += 1
        success, image = vidcap.read()
        frame_count += 1
    vidcap.release()
    print(f"Captured {image_count} screenshots from {video_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_file_or_directory>")
    else:
        main(sys.argv[1])
