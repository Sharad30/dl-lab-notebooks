"""
Stable Diffusion inpainting utilities for facial aging pipeline.

This module provides utilities for:
- Loading and managing Stable Diffusion inpainting models
- Converting between OpenCV and PIL image formats
- Performing facial aging through inpainting
- Managing GPU memory and model caching
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional, Tuple, Union, Dict, Any, TYPE_CHECKING
import warnings
import gc

try:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from diffusers.utils import logging as diffusers_logging
    # Suppress some diffusers warnings
    diffusers_logging.set_verbosity_error()
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Diffusers not installed. Run: pip install diffusers transformers")
    DIFFUSERS_AVAILABLE = False
    # Create a placeholder for type checking
    if TYPE_CHECKING:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    else:
        StableDiffusionInpaintPipeline = None

# Suppress some torch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class StableDiffusionInpainter:
    """Manages Stable Diffusion inpainting pipeline for facial aging."""
    
    def __init__(
        self, 
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        enable_memory_efficient_attention: bool = True
    ):
        """
        Initialize the Stable Diffusion inpainting pipeline.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            dtype: Data type for model weights
            enable_memory_efficient_attention: Enable memory efficient attention if available
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers package is required but not installed")
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
            
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.model_id = model_id
        self.pipeline: Optional[Any] = None  # Use Any to avoid type issues
        
        # Memory optimization settings
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        
        print(f"Initializing Stable Diffusion pipeline on {self.device} with {self.dtype}")
        
    def load_pipeline(self) -> Any:
        """Load the Stable Diffusion inpainting pipeline."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers package is required but not installed")
            
        if self.pipeline is not None:
            return self.pipeline
            
        print(f"Loading model: {self.model_id}")
        
        # Import here to ensure it's available
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
        
        # Load pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            safety_checker=None,  # Disable safety checker for faster loading
            requires_safety_checker=False,
        )
        
        # Move to device
        self.pipeline = self.pipeline.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            # Enable memory efficient attention if available
            if self.enable_memory_efficient_attention:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("Enabled xformers memory efficient attention")
                except Exception:
                    try:
                        self.pipeline.enable_attention_slicing()
                        print("Enabled attention slicing")
                    except Exception:
                        print("No memory optimizations available")
            
            # Enable CPU offload for large models if memory is limited
            try:
                self.pipeline.enable_model_cpu_offload()
                print("Enabled model CPU offload")
            except Exception:
                pass
        
        print("Pipeline loaded successfully")
        return self.pipeline
    
    def unload_pipeline(self):
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            print("Pipeline unloaded")


def opencv_to_pil(image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image (BGR) to PIL image (RGB).
    
    Args:
        image: OpenCV image in BGR format
        
    Returns:
        PIL Image in RGB format
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL image (RGB) to OpenCV image (BGR).
    
    Args:
        pil_image: PIL Image in RGB format
        
    Returns:
        OpenCV image in BGR format
    """
    # Convert to numpy array
    rgb_array = np.array(pil_image)
    # Convert RGB to BGR
    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    return bgr_image


def prepare_mask(mask: np.ndarray) -> Image.Image:
    """
    Prepare mask for Stable Diffusion inpainting.
    
    Args:
        mask: Binary mask (255 = inpaint area, 0 = keep original)
        
    Returns:
        PIL Image mask ready for inpainting
    """
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.uint8) * 255
    
    # Convert to PIL
    mask_pil = Image.fromarray(mask_binary, mode='L')
    
    return mask_pil


def inpaint_face(
    image: np.ndarray, 
    mask: np.ndarray, 
    prompt: str = "A realistic photo of the same person aged 10 years",
    negative_prompt: str = "blurry, distorted, unrealistic, bad quality, artifacts",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
    seed: Optional[int] = None,
    model_id: str = "runwayml/stable-diffusion-inpainting"
) -> np.ndarray:
    """
    Inpaint face using Stable Diffusion for aging effects.
    
    Args:
        image: Input image as OpenCV array (BGR format)
        mask: Binary mask (255 = inpaint area, 0 = keep original)
        prompt: Text prompt for inpainting
        negative_prompt: Negative prompt to avoid unwanted features
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        strength: Strength of the inpainting effect (0.0 to 1.0)
        seed: Random seed for reproducibility
        model_id: HuggingFace model identifier
        
    Returns:
        Inpainted image as OpenCV array (BGR format)
        
    Example:
        >>> import cv2
        >>> from face_utils import get_face_mask
        >>> from inpaint_utils import inpaint_face
        >>> 
        >>> image = cv2.imread("face.jpg")
        >>> mask = get_face_mask(image)
        >>> aged_image = inpaint_face(image, mask)
        >>> cv2.imwrite("aged_face.jpg", aged_image)
    """
    # Create inpainter instance
    inpainter = StableDiffusionInpainter(model_id=model_id)
    
    try:
        # Load pipeline
        pipeline = inpainter.load_pipeline()
        
        # Convert image formats
        pil_image = opencv_to_pil(image)
        pil_mask = prepare_mask(mask)
        
        # Ensure images are the same size
        if pil_image.size != pil_mask.size:
            pil_mask = pil_mask.resize(pil_image.size, Image.Resampling.LANCZOS)
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=inpainter.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"Starting inpainting with prompt: '{prompt}'")
        print(f"Image size: {pil_image.size}, Steps: {num_inference_steps}")
        
        # Perform inpainting
        with torch.autocast(inpainter.device):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator
            )
        
        # Get the result image
        result_image = result.images[0]
        
        # Convert back to OpenCV format
        opencv_result = pil_to_opencv(result_image)
        
        print("Inpainting completed successfully")
        return opencv_result
        
    except Exception as e:
        print(f"Error during inpainting: {e}")
        raise
    finally:
        # Clean up memory
        if inpainter.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def batch_inpaint_faces(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    prompts: Union[str, list[str]] = "A realistic photo of the same person aged 10 years",
    **kwargs
) -> list[np.ndarray]:
    """
    Batch inpaint multiple faces for efficiency.
    
    Args:
        images: List of input images as OpenCV arrays
        masks: List of corresponding masks
        prompts: Single prompt or list of prompts for each image
        **kwargs: Additional arguments passed to inpaint_face
        
    Returns:
        List of inpainted images as OpenCV arrays
    """
    if len(images) != len(masks):
        raise ValueError("Number of images and masks must match")
    
    # Handle prompts
    if isinstance(prompts, str):
        prompt_list = [prompts] * len(images)
    else:
        if len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")
        prompt_list = prompts
    
    # Create single inpainter instance for efficiency
    model_id = kwargs.get('model_id', "runwayml/stable-diffusion-inpainting")
    inpainter = StableDiffusionInpainter(model_id=model_id)
    pipeline = inpainter.load_pipeline()
    
    results = []
    
    try:
        for i, (image, mask, prompt) in enumerate(zip(images, masks, prompt_list)):
            print(f"Processing image {i+1}/{len(images)}")
            
            # Use the existing pipeline instead of creating new one
            kwargs_copy = kwargs.copy()
            kwargs_copy['model_id'] = model_id
            
            result = inpaint_face(image, mask, prompt, **kwargs_copy)
            results.append(result)
            
    finally:
        # Clean up
        inpainter.unload_pipeline()
    
    return results


def create_age_variants(
    image: np.ndarray,
    mask: np.ndarray,
    age_increments: list[int] = [5, 10, 15, 20],
    base_prompt: str = "A realistic photo of the same person aged {} years",
    **kwargs
) -> Dict[int, np.ndarray]:
    """
    Create multiple age variants of the same face.
    
    Args:
        image: Input image as OpenCV array
        mask: Binary face mask
        age_increments: List of years to age the person
        base_prompt: Base prompt with {} placeholder for age
        **kwargs: Additional arguments passed to inpaint_face
        
    Returns:
        Dictionary mapping age increment to result image
    """
    results = {}
    
    # Create prompts for each age
    prompts = [base_prompt.format(age) for age in age_increments]
    images = [image] * len(age_increments)
    masks = [mask] * len(age_increments)
    
    # Batch process
    inpainted_images = batch_inpaint_faces(images, masks, prompts, **kwargs)
    
    # Create result dictionary
    for age, result_image in zip(age_increments, inpainted_images):
        results[age] = result_image
    
    return results


# Convenience function for testing
def test_inpainting(image_path: str, output_dir: str = "./output"):
    """
    Test the inpainting pipeline on a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    import os
    from face_utils import get_face_mask
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print("Extracting face mask...")
    mask = get_face_mask(image)
    
    if mask is None:
        print("No face detected in image")
        return
    
    # Save original and mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), image)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.jpg"), mask)
    
    print("Starting inpainting...")
    
    # Test different age increments
    age_variants = create_age_variants(
        image, mask, 
        age_increments=[5, 10, 15, 20],
        num_inference_steps=30,  # Faster for testing
        seed=42  # Reproducible results
    )
    
    # Save results
    for age, result_image in age_variants.items():
        output_path = os.path.join(output_dir, f"{base_name}_aged_{age}years.jpg")
        cv2.imwrite(output_path, result_image)
        print(f"Saved: {output_path}")
    
    print("Testing completed!") 