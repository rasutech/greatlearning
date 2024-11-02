import os
import whisper
import torch
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
import moviepy.editor as mp
import numpy as np
import gc
import psutil
from dataclasses import dataclass
from threading import Lock
import json
from datetime import datetime
import argparse

@dataclass
class GPUStats:
    gpu_utilization: float
    memory_used: float
    memory_total: float

@dataclass
class ProcessingResult:
    video_name: str
    transcript: str
    duration: float
    output_dir: Path
    status: str
    error: Optional[str] = None

class GPUOptimizedTranscriber:
    def __init__(self, 
                 model_name: str = "medium", 
                 device: str = None,
                 batch_size: int = 16,
                 chunk_duration: int = 30,
                 overlap_duration: int = 2):
        """Initialize transcriber with GPU optimizations."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize device and model
        self.device = self._initialize_device(device)
        self.batch_size = batch_size
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        # Load model
        self.model = self._initialize_model(model_name)
        self.gpu_lock = Lock()
        
        # Performance monitoring
        self.performance_stats = {
            'gpu_utilization': [],
            'memory_usage': [],
            'processing_times': []
        }

    def _initialize_device(self, device: Optional[str] = None) -> str:
        """Initialize and return the appropriate device."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            
            self.logger.info(f"Using GPU: {gpu_name}")
            self.logger.info(f"Initial GPU Memory: {memory_allocated/1024**3:.2f}GB / {memory_total/1024**3:.2f}GB")
        else:
            self.logger.warning("GPU not available, using CPU")
            
        return device

    def _initialize_model(self, model_name: str) -> whisper.Whisper:
        """Initialize and optimize the Whisper model."""
        self.logger.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        if self.device == 'cuda':
            model = model.to(self.device)
            model = model.half()  # Use half precision
            model.eval()  # Set to evaluation mode
            
            # Warmup run
            self.logger.info("Performing warmup run...")
            with torch.cuda.amp.autocast():
                dummy_input = torch.randn(1, 80, 3000).half().cuda()
                with torch.no_grad():
                    _ = model.encode(dummy_input)
        
        return model

    def get_gpu_stats(self) -> GPUStats:
        """Get current GPU statistics."""
        if self.device == 'cuda':
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_used = torch.cuda.memory_allocated(0)
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except:
                gpu_util = 0
                
            return GPUStats(
                gpu_utilization=gpu_util,
                memory_used=memory_used,
                memory_total=memory_total
            )
        return GPUStats(0, 0, 0)

    def process_audio_chunk(self, audio_data: np.ndarray, start_time: float) -> Dict:
        """Process a single audio chunk with GPU optimization."""
        try:
            with torch.cuda.amp.autocast(), torch.no_grad():
                result = self.model.transcribe(
                    audio_data,
                    fp16=True if self.device == 'cuda' else False,
                    language="en",
                    task="transcribe",
                    start_time=start_time
                )
            return result
        except Exception as e:
            self.logger.error(f"Error processing chunk at {start_time}: {str(e)}")
            return {"text": ""}

    def parallel_transcribe(self, audio_path: str) -> str:
        """Transcribe audio with parallel GPU processing."""
        self.logger.info(f"Starting parallel transcription of {audio_path}")
        
        # Load audio
        audio = mp.AudioFileClip(audio_path)
        duration = audio.duration
        
        # Calculate chunks with overlap
        chunks = []
        current_time = 0
        while current_time < duration:
            chunk_end = min(current_time + self.chunk_duration, duration)
            chunk = {
                'start': max(0, current_time - self.overlap_duration),
                'end': chunk_end
            }
            chunks.append(chunk)
            current_time = chunk_end
        
        # Process chunks
        transcripts = []
        
        with ThreadPoolExecutor(max_workers=min(len(chunks), multiprocessing.cpu_count())) as executor:
            futures = []
            
            for chunk in chunks:
                # Extract audio chunk
                chunk_audio = audio.subclip(chunk['start'], chunk['end'])
                chunk_data = np.array(chunk_audio.to_soundarray(), dtype=np.float32)
                
                future = executor.submit(
                    self.process_audio_chunk,
                    chunk_data,
                    chunk['start']
                )
                futures.append(future)
            
            # Process results
            for future in tqdm(futures, desc="Processing audio chunks"):
                result = future.result()
                transcripts.append(result["text"])
        
        # Clean up
        audio.close()
        
        # Merge transcripts
        return ' '.join(transcripts)

    def generate_optimized_transcript(self, video_path: str) -> str:
        """Generate transcript with GPU optimization."""
        try:
            # Extract audio
            self.logger.info("Extracting audio...")
            audio_path = str(Path(video_path).with_suffix('.mp3'))
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, fps=16000)
            video.close()
            
            # Transcribe
            self.logger.info("Transcribing with GPU optimization...")
            transcript = self.parallel_transcribe(audio_path)
            
            # Clean up
            os.remove(audio_path)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Log performance stats
            self._log_performance_stats()
            
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            raise

    def _log_performance_stats(self):
        """Log GPU performance statistics."""
        if self.performance_stats['gpu_utilization']:
            avg_gpu_util = np.mean(self.performance_stats['gpu_utilization'])
            max_memory = max(self.performance_stats['memory_usage'])
            
            self.logger.info(f"Average GPU Utilization: {avg_gpu_util:.2f}%")
            self.logger.info(f"Peak GPU Memory Usage: {max_memory/1024**3:.2f}GB")

    def process_video(self, video_path: Path, output_dir: Path) -> ProcessingResult:
        """Process a single video file."""
        start_time = time.time()
        
        try:
            # Generate transcript
            transcript = self.generate_optimized_transcript(str(video_path))
            
            # Calculate duration
            duration = time.time() - start_time
            
            return ProcessingResult(
                video_name=video_path.stem,
                transcript=transcript,
                duration=duration,
                output_dir=output_dir,
                status="success"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {video_path.name}: {str(e)}")
            return ProcessingResult(
                video_name=video_path.stem,
                transcript="",
                duration=time.time() - start_time,
                output_dir=output_dir,
                status="error",
                error=str(e)
            )

class DirectoryProcessor:
    def __init__(self, input_dir: str, output_base_dir: str):
        """Initialize directory processor."""
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Setup logging for directory processing."""
        log_dir = self.output_base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        
    def get_video_files(self) -> List[Path]:
        """Get all MP4 files in the input directory."""
        return list(self.input_dir.glob("**/*.mp4"))
    
    def create_output_directory(self, video_path: Path) -> Path:
        """Create output directory for a video file."""
        output_dir = self.output_base_dir / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def save_processing_result(self, result: ProcessingResult):
        """Save processing results to files."""
        # Save transcript
        transcript_file = result.output_dir / f"{result.video_name}_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(result.transcript)
        
        # Save metadata
        metadata = {
            'video_name': result.video_name,
            'processing_duration': result.duration,
            'status': result.status,
            'error': result.error,
            'processed_at': datetime.now().isoformat(),
        }
        
        metadata_file = result.output_dir / f"{result.video_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process videos in directory')
    parser.add_argument('input_dir', help='Input directory containing MP4 files')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--model', default='medium', help='Whisper model size')
    parser.add_argument('--batch-size', type=int, default=16, help='GPU batch size')
    args = parser.parse_args()

    # Initialize processors
    directory_processor = DirectoryProcessor(args.input_dir, args.output_dir)
    transcriber = GPUOptimizedTranscriber(
        model_name=args.model,
        batch_size=args.batch_size
    )

    # Get video files
    video_files = directory_processor.get_video_files()
    if not video_files:
        print(f"No MP4 files found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} MP4 files to process")

    # Process each video
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Create output directory for this video
        output_dir = directory_processor.create_output_directory(video_path)
        
        # Process video
        print(f"\nProcessing: {video_path.name}")
        result = transcriber.process_video(video_path, output_dir)
        
        # Save results
        directory_processor.save_processing_result(result)
        
        # Log result
        status = "✓" if result.status == "success" else "✗"
        print(f"{status} Completed in {result.duration:.2f} seconds")
        
        # Clear GPU cache between videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary
    print("\nProcessing Summary:")
    print(f"Total videos processed: {len(video_files)}")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    import time
    main()
