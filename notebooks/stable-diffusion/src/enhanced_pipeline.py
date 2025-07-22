"""
Enhanced facial aging pipeline with improved techniques for realistic results.

This module provides advanced techniques for better aging accuracy:
- Optimized prompting strategies
- Better parameter tuning
- Multi-pass aging for gradual effects
- Quality enhancement techniques
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Union, Tuple, Optional, Any
from face_utils import get_face_mask, visualize_face_mask
from inpaint_utils import inpaint_face, create_age_variants
from aging_prompts import get_premium_prompt, AgingPromptGenerator


class EnhancedFacialAgingPipeline:
    """
    Enhanced facial aging pipeline with improved realism and accuracy.
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-inpainting"):
        """
        Initialize the enhanced aging pipeline.
        
        Args:
            model_id: Stable Diffusion model to use
        """
        self.model_id = model_id
        self.prompt_generator = AgingPromptGenerator()
        
        # Optimized parameters for realistic aging
        self.default_params = {
            "num_inference_steps": 50,  # Higher for better quality
            "guidance_scale": 7.5,      # Balanced prompt adherence
            "strength": 0.85,           # Preserve more of original
        }
        
        print(f"Enhanced Facial Aging Pipeline initialized with model: {model_id}")
    
    def process_gradual_aging(
        self,
        image_path: str,
        target_age: int,
        steps: int = 2,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, int]]:
        """
        Apply aging gradually in multiple steps for more realistic results.
        
        Args:
            image_path: Path to input image
            target_age: Total years to age
            steps: Number of gradual steps (2-4 recommended)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with original and final aged result
        """
        print(f"Gradual aging: {target_age} years in {steps} steps")
        
        # Load initial image
        current_image = cv2.imread(image_path)
        if current_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = current_image.copy()
        
        # Calculate age increments
        age_per_step = target_age / steps
        
        for step in range(steps):
            step_age = int(age_per_step * (step + 1))
            print(f"  → Step {step + 1}/{steps}: aging by {age_per_step:.1f} years (total: {step_age})")
            
            # Get face mask for current image
            mask = get_face_mask(current_image)
            if mask is None:
                raise ValueError(f"No face detected at step {step + 1}")
            
            # Generate optimized prompt
            positive_prompt, negative_prompt = get_premium_prompt(int(age_per_step))
            
            # Apply aging with enhanced parameters
            enhanced_params = {
                **self.default_params,
                "strength": max(0.7, 0.9 - step * 0.1),  # Reduce strength in later steps
                **kwargs
            }
            
            current_image = inpaint_face(
                image=current_image,
                mask=mask,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                model_id=self.model_id,
                **enhanced_params
            )
        
        return {
            "original": original_image,
            "aged": current_image,
            "target_age": target_age,
            "steps_used": steps
        }
    
    def process_high_quality_aging(
        self,
        image_path: str,
        age_years: int,
        gender_hint: str = "general",
        style: str = "natural",
        **kwargs
    ) -> Dict[str, Union[np.ndarray, int, str]]:
        """
        Process single image with high-quality aging settings.
        
        Args:
            image_path: Path to input image
            age_years: Years to age
            gender_hint: "general", "masculine", or "feminine"
            style: "natural", "graceful", or "distinguished"
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processing results
        """
        print(f"High-quality aging: {age_years} years ({style} style)")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract high-quality face mask
        mask = get_face_mask(image, mask_type="facial_region")  # More precise mask
        if mask is None:
            raise ValueError("No face detected in image")
        
        # Generate sophisticated prompt
        positive_prompt = self.prompt_generator.generate_aging_prompt(
            age_years=age_years,
            gender_hint=gender_hint,
            style=style,
            include_quality=True
        )
        negative_prompt = self.prompt_generator.generate_negative_prompt(detailed=True)
        
        print(f"  → Using prompt: {positive_prompt[:80]}...")
        
        # High-quality parameters
        hq_params = {
            "num_inference_steps": 60,  # Even higher quality
            "guidance_scale": 8.0,      # Slightly stronger prompt following
            "strength": 0.8,            # Preserve original features
            **kwargs
        }
        
        # Apply aging
        aged_image = inpaint_face(
            image=image,
            mask=mask,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            model_id=self.model_id,
            **hq_params
        )
        
        return {
            "original": image,
            "mask": mask,
            "aged": aged_image,
            "age_years": age_years,
            "prompt_used": positive_prompt
        }
    
    def process_age_series_enhanced(
        self,
        image_path: str,
        age_increments: List[int] = [5, 10, 15, 20],
        use_gradual: bool = True,
        **kwargs
    ) -> Dict[Union[str, int], np.ndarray]:
        """
        Create enhanced age series with better realism.
        
        Args:
            image_path: Path to input image
            age_increments: List of years to age
            use_gradual: Whether to use gradual aging for better results
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with all aging results
        """
        print(f"Enhanced age series: {age_increments}")
        
        # Load original
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results = {"original": original_image}
        
        for age in age_increments:
            print(f"\nProcessing {age} years...")
            
            if use_gradual and age >= 15:
                # Use gradual aging for larger age gaps
                steps = min(3, max(2, age // 8))
                age_result = self.process_gradual_aging(
                    image_path=image_path,
                    target_age=age,
                    steps=steps,
                    **kwargs
                )
                results[age] = age_result["aged"]
            else:
                # Use high-quality single-step aging
                age_result = self.process_high_quality_aging(
                    image_path=image_path,
                    age_years=age,
                    **kwargs
                )
                results[age] = age_result["aged"]
        
        return results
    
    def compare_aging_techniques(
        self,
        image_path: str,
        age_years: int = 15
    ) -> Dict[str, np.ndarray]:
        """
        Compare different aging techniques side by side.
        
        Args:
            image_path: Path to input image
            age_years: Years to age for comparison
            
        Returns:
            Dictionary with different technique results
        """
        print(f"Comparing aging techniques for {age_years} years")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results = {"original": image}
        
        # Technique 1: Basic aging
        print("  → Basic aging...")
        mask = get_face_mask(image)
        if mask is None:
            raise ValueError("No face detected for basic aging")
            
        basic_result = inpaint_face(
            image=image,
            mask=mask,
            prompt=f"A realistic photo of the same person aged {age_years} years",
            num_inference_steps=30,
            guidance_scale=7.5
        )
        results["basic"] = basic_result
        
        # Technique 2: Enhanced prompting
        print("  → Enhanced prompting...")
        positive, negative = get_premium_prompt(age_years)
        enhanced_result = inpaint_face(
            image=image,
            mask=mask,
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=0.85
        )
        results["enhanced_prompt"] = enhanced_result
        
        # Technique 3: Gradual aging
        print("  → Gradual aging...")
        gradual_result = self.process_gradual_aging(
            image_path=image_path,
            target_age=age_years,
            steps=2
        )
        results["gradual"] = gradual_result["aged"]
        
        return results
    
    def visualize_comparison(self, results: Dict, title: str = "Aging Technique Comparison"):
        """Enhanced visualization with technique labels."""
        plot_items = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
        
        num_items = len(plot_items)
        if num_items == 0:
            print("No images to display")
            return
        
        fig, axes = plt.subplots(1, num_items, figsize=(4*num_items, 6))
        if num_items == 1:
            axes = [axes]
        
        for idx, (key, image) in enumerate(plot_items.items()):
            axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Enhanced labels
            labels = {
                "original": "Original",
                "basic": "Basic Aging",
                "enhanced_prompt": "Enhanced Prompts",
                "gradual": "Gradual Aging"
            }
            
            if key in labels:
                axes[idx].set_title(labels[key], fontsize=12, fontweight='bold')
            elif isinstance(key, int):
                axes[idx].set_title(f"Aged +{key} Years", fontsize=12)
            else:
                axes[idx].set_title(str(key).replace("_", " ").title(), fontsize=12)
            
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Convenience functions for quick testing
def test_enhanced_aging(image_path: str, output_dir: str = "./enhanced_output"):
    """Test the enhanced aging pipeline."""
    pipeline = EnhancedFacialAgingPipeline()
    
    # Test comparison
    comparison = pipeline.compare_aging_techniques(image_path, age_years=15)
    pipeline.visualize_comparison(comparison, "Aging Technique Comparison")
    
    # Test enhanced age series
    age_series = pipeline.process_age_series_enhanced(
        image_path=image_path,
        age_increments=[10, 20, 30],
        use_gradual=True
    )
    pipeline.visualize_comparison(age_series, "Enhanced Age Series")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    for key, image in {**comparison, **age_series}.items():
        if isinstance(image, np.ndarray):
            filename = f"{key}_result.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), image)
    
    print(f"Results saved to {output_dir}")


def get_aging_recommendations() -> Dict[str, str]:
    """Get recommendations for better aging results."""
    return {
        "image_quality": "Use high-resolution images (512x512 or larger) with good lighting",
        "face_positioning": "Ensure face is centered and clearly visible",
        "prompt_engineering": "Use detailed, age-specific prompts with quality modifiers",
        "parameter_tuning": "Higher inference steps (50+) and strength 0.8-0.9 for better quality",
        "gradual_aging": "For large age gaps (20+ years), use gradual multi-step aging",
        "mask_quality": "Use 'facial_region' mask type for more precise aging areas",
        "model_choice": "Consider specialized aging models if available",
        "post_processing": "Apply subtle image enhancement after aging if needed"
    } 