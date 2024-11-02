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

class DirectoryProcessor:
    def __init__(self, input_dir: str, output_base_dir: str):
        """
        Initialize directory processor.
        
        Args:
            input_dir: Directory containing video files
            output_base_dir: Base directory for outputs
        """
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
        # Create a directory with the same name as the video (without extension)
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

class GPUOptimizedTranscriber:
    def __init__(self, 
                 model_name: str = "medium", 
                 device: str = None,
                 batch_size: int = 16,
                 chunk_duration: int = 30,
                 overlap_duration: int = 2):
        """Initialize transcriber with GPU optimizations."""
        self.logger = logging.getLogger(__name__)
        
        # GPU setup and verification
        self.device = self._setup_gpu(device)
        self.batch_size = batch_size
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        # Load model with optimized settings
        self.model = self._load_optimized_model(model_name)
        self.gpu_lock = Lock()
        
        # Performance monitoring
        self.performance_stats = {
            'gpu_utilization': [],
            'memory_usage': [],
            'processing_times': []
        }

    # [Previous GPU optimization methods remain the same...]
    # Note: Keep all the methods from the previous version, just adding new functionality

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

def main():
    # Parse command line arguments
    import argparse
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
