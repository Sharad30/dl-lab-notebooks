"""
Stable Diffusion Facial Aging Pipeline

A modular pipeline for facial aging using Stable Diffusion inpainting.
This package provides utilities for:
- Face mask extraction using MediaPipe
- Stable Diffusion inpainting for facial aging
- Video processing capabilities
- Reusable utilities for Jupyter notebooks

Usage:
    from face_utils import get_face_mask, visualize_face_mask
    from inpaint_utils import inpaint_face, create_age_variants
    
    # Extract face mask
    mask = get_face_mask(image)
    
    # Age the face
    aged_image = inpaint_face(image, mask)
    
    # Create multiple age variants
    age_variants = create_age_variants(image, mask)
"""

__version__ = "0.1.0"
__author__ = "DL Lab"

# Import face utilities
try:
    from .face_utils import (
        get_face_mask,
        visualize_face_mask,
        test_face_mask_extraction,
        FaceMaskExtractor
    )
    
    face_utils_available = True
    print("Face utilities loaded successfully")
    
except ImportError as e:
    print(f"Warning: Could not import face utilities: {e}")
    print("Make sure MediaPipe is installed: pip install mediapipe")
    face_utils_available = False

# Import inpainting utilities
try:
    from .inpaint_utils import (
        inpaint_face,
        create_age_variants,
        batch_inpaint_faces,
        test_inpainting,
        opencv_to_pil,
        pil_to_opencv,
        prepare_mask,
        StableDiffusionInpainter
    )
    
    inpaint_utils_available = True
    print("Inpainting utilities loaded successfully")
    
except ImportError as e:
    print(f"Warning: Could not import inpainting utilities: {e}")
    print("Make sure diffusers and torch are installed: pip install diffusers torch torchvision")
    inpaint_utils_available = False

# Build __all__ list based on what's available
__all__ = []

if face_utils_available:
    __all__.extend([
        "get_face_mask",
        "visualize_face_mask", 
        "test_face_mask_extraction",
        "FaceMaskExtractor"
    ])

if inpaint_utils_available:
    __all__.extend([
        "inpaint_face",
        "create_age_variants",
        "batch_inpaint_faces",
        "test_inpainting",
        "opencv_to_pil",
        "pil_to_opencv",
        "prepare_mask",
        "StableDiffusionInpainter"
    ])

# Utility function to check system readiness
def check_system_requirements():
    """
    Check if all required dependencies are available.
    
    Returns:
        dict: Status of each component
    """
    status = {
        "face_extraction": face_utils_available,
        "inpainting": inpaint_utils_available,
        "cuda": False,
        "ready_for_full_pipeline": False
    }
    
    # Check CUDA availability
    try:
        import torch
        status["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Check if ready for full pipeline
    status["ready_for_full_pipeline"] = (
        status["face_extraction"] and 
        status["inpainting"]
    )
    
    return status

def print_system_status():
    """Print a detailed system status report."""
    status = check_system_requirements()
    
    print("=== Facial Aging Pipeline System Status ===")
    print(f"Face Extraction:     {'✅' if status['face_extraction'] else '❌'}")
    print(f"Stable Diffusion:    {'✅' if status['inpainting'] else '❌'}")
    print(f"CUDA Support:        {'✅' if status['cuda'] else '❌'}")
    print(f"Full Pipeline Ready: {'✅' if status['ready_for_full_pipeline'] else '❌'}")
    
    if not status["ready_for_full_pipeline"]:
        print("\n=== Installation Instructions ===")
        if not status["face_extraction"]:
            print("Install face utilities: pip install mediapipe opencv-python")
        if not status["inpainting"]:
            print("Install ML dependencies: pip install diffusers transformers torch torchvision")
        if not status["cuda"]:
            print("Note: CUDA not available. Pipeline will run on CPU (slower).")

# Print status on import
if __name__ != "__main__":
    print_system_status() 