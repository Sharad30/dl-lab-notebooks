"""
Video processing utilities for facial aging.

This module provides frame-by-frame video processing capabilities
for applying facial aging effects to video content.
"""

import cv2
import numpy as np
import os
import tempfile
from typing import Optional, Tuple
from tqdm import tqdm
from face_utils import get_face_mask
from inpaint_utils import inpaint_face


def process_video(
    input_path: str, 
    output_path: str, 
    prompt: str = "A realistic photo of the same person aged 10 years"
) -> None:
    """
    Process video frame-by-frame to apply facial aging.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        prompt: Aging prompt for inpainting
    """
    print(f"Processing video: {input_path}")
    print(f"Output: {output_path}")
    print(f"Prompt: {prompt}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")
    
    frames_processed = 0
    frames_with_faces = 0
    frames_skipped = 0
    
    try:
        # Process frames with progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Try to detect face and apply aging
                processed_frame = process_single_frame(frame, prompt)
                
                if processed_frame is not None:
                    out.write(processed_frame)
                    frames_with_faces += 1
                else:
                    # No face detected, write original frame
                    out.write(frame)
                    frames_skipped += 1
                
                frames_processed += 1
                pbar.update(1)
                
                # Update progress description
                if frames_processed % 10 == 0:
                    pbar.set_description(f"Faces: {frames_with_faces}, Skipped: {frames_skipped}")
    
    finally:
        # Clean up
        cap.release()
        out.release()
    
    print(f"\nVideo processing completed!")
    print(f"Total frames: {frames_processed}")
    print(f"Frames with faces processed: {frames_with_faces}")
    print(f"Frames passed through: {frames_skipped}")
    print(f"Output saved to: {output_path}")


def process_single_frame(frame: np.ndarray, prompt: str) -> Optional[np.ndarray]:
    """
    Process a single video frame for facial aging.
    
    Args:
        frame: Input frame as numpy array
        prompt: Aging prompt for inpainting
        
    Returns:
        Processed frame or None if no face detected
    """
    try:
        # Get face mask
        mask = get_face_mask(frame)
        if mask is None:
            return None
        
        # Apply inpainting
        aged_frame = inpaint_face(
            image=frame,
            mask=mask,
            prompt=prompt,
            num_inference_steps=30,  # Lower for speed
            guidance_scale=7.5,
            strength=0.85
        )
        
        return aged_frame
        
    except Exception as e:
        print(f"Warning: Frame processing failed: {e}")
        return None


def process_video_batch(
    input_path: str,
    output_path: str,
    prompts: list[str],
    batch_size: int = 4
) -> None:
    """
    Process video with multiple prompts in batches for efficiency.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        prompts: List of prompts to cycle through
        batch_size: Number of frames to process together
    """
    print(f"Batch processing video with {len(prompts)} prompts")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_buffer = []
    prompt_index = 0
    
    try:
        with tqdm(total=total_frames, desc="Batch processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in buffer
                    if frame_buffer:
                        process_frame_batch(frame_buffer, prompts, out)
                    break
                
                frame_buffer.append(frame)
                
                # Process batch when buffer is full
                if len(frame_buffer) >= batch_size:
                    process_frame_batch(frame_buffer, prompts, out)
                    frame_buffer = []
                
                pbar.update(1)
    
    finally:
        cap.release()
        out.release()
    
    print(f"Batch processing completed: {output_path}")


def process_frame_batch(frames: list[np.ndarray], prompts: list[str], writer: cv2.VideoWriter) -> None:
    """Process a batch of frames efficiently."""
    for i, frame in enumerate(frames):
        prompt = prompts[i % len(prompts)]
        processed = process_single_frame(frame, prompt)
        writer.write(processed if processed is not None else frame)


def extract_video_info(video_path: str) -> dict:
    """
    Extract detailed information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
    }
    
    cap.release()
    return info


def create_aging_progression_video(
    image_path: str,
    output_path: str,
    age_increments: list[int] = [5, 10, 15, 20, 25],
    duration_per_age: float = 2.0,
    fps: int = 30
) -> None:
    """
    Create a video showing aging progression from a single image.
    
    Args:
        image_path: Path to input image
        output_path: Path to output video
        age_increments: List of aging years
        duration_per_age: Seconds to show each age
        fps: Frames per second
    """
    from enhanced_pipeline import EnhancedFacialAgingPipeline
    
    print(f"Creating aging progression video from: {image_path}")
    
    # Initialize pipeline
    pipeline = EnhancedFacialAgingPipeline()
    
    # Generate aged images
    results = pipeline.process_age_series_enhanced(
        image_path=image_path,
        age_increments=age_increments,
        use_gradual=True
    )
    
    # Load original image to get dimensions
    original = cv2.imread(image_path)
    height, width = original.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frames_per_age = int(duration_per_age * fps)
    
    try:
        # Add frames for each age
        for age in ["original"] + age_increments:
            if age in results:
                image = results[age]
                for _ in range(frames_per_age):
                    out.write(image)
                print(f"Added {frames_per_age} frames for age: {age}")
    
    finally:
        out.release()
    
    print(f"Aging progression video created: {output_path}")


# Example usage functions
def demo_video_processing():
    """Demonstrate video processing capabilities."""
    print("=== Video Processing Demo ===")
    
    # Example video info
    sample_video = "path/to/sample.mp4"
    if os.path.exists(sample_video):
        info = extract_video_info(sample_video)
        print(f"Video info: {info}")
        
        # Process video
        output_video = "aged_output.mp4"
        process_video(
            input_path=sample_video,
            output_path=output_video,
            prompt="A realistic photo of the same person aged 15 years"
        )
    else:
        print("No sample video found for demo")


if __name__ == "__main__":
    demo_video_processing() 