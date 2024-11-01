import os
import whisper
import torch
import subprocess
from pathlib import Path
from typing import Optional
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm

class OptimizedTranscriber:
    def __init__(self, model_name: str = "medium", device: str = None):
        """
        Initialize the transcriber with optimized settings.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Force specific device ('cuda', 'cpu', or None for auto-detection)
        """
        self.logger = logging.getLogger(__name__)
        
        # Automatically detect best available device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Load model with optimal settings
        self.model = whisper.load_model(model_name).to(self.device)
        
        # Set optimal torch settings
        if self.device == 'cuda':
            torch.set_float32_matmul_precision('medium')
            torch.backends.cudnn.benchmark = True
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video using FFmpeg for faster processing.
        """
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.mp3'))
            
        # Use FFmpeg to extract audio with optimized settings
        command = [
            'ffmpeg', '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'libmp3lame',
            '-ar', '16000',  # Sample rate Whisper expects
            '-ac', '1',  # Mono channel
            '-q:a', '0',  # Highest quality
            '-y',  # Overwrite output file
            output_path
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path

    def transcribe_chunk(self, audio_path: str, start_time: float, duration: float) -> dict:
        """
        Transcribe a specific chunk of audio.
        """
        options = {
            "language": "en",
            "task": "transcribe",
            "start_time": start_time,
            "duration": duration,
            "fp16": self.device == 'cuda'  # Use FP16 for CUDA
        }
        
        return self.model.transcribe(audio_path, **options)

    def parallel_transcribe(self, audio_path: str, chunk_duration: int = 600) -> str:
        """
        Transcribe audio in parallel chunks.
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds (default 10 minutes)
        """
        # Get audio duration using FFprobe
        duration_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ]
        duration = float(subprocess.check_output(duration_cmd).decode().strip())
        
        # Calculate chunks
        chunks = []
        current_time = 0
        while current_time < duration:
            chunk_length = min(chunk_duration, duration - current_time)
            chunks.append((current_time, chunk_length))
            current_time += chunk_length
        
        # Process chunks in parallel
        transcripts = []
        with ThreadPoolExecutor(max_workers=min(len(chunks), multiprocessing.cpu_count())) as executor:
            futures = []
            for start_time, chunk_length in chunks:
                future = executor.submit(
                    self.transcribe_chunk, 
                    audio_path, 
                    start_time, 
                    chunk_length
                )
                futures.append(future)
            
            # Show progress bar
            for future in tqdm(futures, desc="Transcribing chunks"):
                result = future.result()
                transcripts.append(result["text"])
        
        return ' '.join(transcripts)

    def generate_optimized_transcript(self, video_path: str, use_chunks: bool = True) -> str:
        """
        Generate transcript with optimized performance.
        
        Args:
            video_path: Path to video file
            use_chunks: Whether to use parallel chunk processing
        """
        try:
            # Extract audio first
            self.logger.info("Extracting audio from video...")
            audio_path = self.extract_audio(video_path)
            
            # Transcribe
            self.logger.info("Transcribing audio...")
            if use_chunks:
                transcript = self.parallel_transcribe(audio_path)
            else:
                transcript = self.model.transcribe(audio_path)["text"]
            
            # Clean up
            os.remove(audio_path)
            
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            raise

def main():
    # Example usage with performance tracking
    import time
    
    video_path = "path/to/your/video.mp4"
    
    # Initialize transcriber
    transcriber = OptimizedTranscriber(
        model_name="medium",  # You can try 'small' or 'base' for faster processing
        device=None  # Auto-detect best device
    )
    
    # Time the transcription
    start_time = time.time()
    transcript = transcriber.generate_optimized_transcript(video_path)
    end_time = time.time()
    
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    print(f"Transcript length: {len(transcript)} characters")

if __name__ == "__main__":
    main()
