"""
Advanced prompting strategies for realistic facial aging.

This module provides better prompts and techniques for more accurate
facial aging results using Stable Diffusion inpainting.
"""

from typing import Dict, List, Tuple
import random


class AgingPromptGenerator:
    """Generate sophisticated prompts for realistic facial aging."""
    
    def __init__(self):
        # Age-specific characteristics
        self.aging_features = {
            "early": {
                "years": (5, 15),
                "features": [
                    "slight laugh lines",
                    "subtle facial maturity",
                    "natural aging",
                    "gentle wrinkles around eyes",
                    "mature expression"
                ]
            },
            "middle": {
                "years": (15, 30),
                "features": [
                    "visible wrinkles",
                    "crow's feet",
                    "forehead lines",
                    "nasolabial folds",
                    "gray hair starting",
                    "mature facial structure"
                ]
            },
            "advanced": {
                "years": (30, 50),
                "features": [
                    "deep wrinkles",
                    "significant gray hair",
                    "age spots",
                    "sagging skin",
                    "pronounced facial lines",
                    "elderly appearance"
                ]
            }
        }
        
        # Gender-specific aging patterns
        self.gender_features = {
            "general": [
                "natural aging progression",
                "realistic wrinkles",
                "age-appropriate features"
            ],
            "masculine": [
                "receding hairline",
                "beard gray",
                "strong jaw aging"
            ],
            "feminine": [
                "graceful aging",
                "elegant maturity",
                "refined features"
            ]
        }
        
        # Quality enhancers
        self.quality_terms = [
            "photorealistic",
            "high quality",
            "professional photography",
            "studio lighting",
            "detailed",
            "sharp focus",
            "natural looking"
        ]
    
    def generate_aging_prompt(
        self, 
        age_years: int, 
        gender_hint: str = "general",
        style: str = "natural",
        include_quality: bool = True
    ) -> str:
        """
        Generate a sophisticated aging prompt.
        
        Args:
            age_years: Number of years to age
            gender_hint: "general", "masculine", or "feminine"
            style: "natural", "graceful", or "distinguished"
            include_quality: Whether to include quality enhancement terms
            
        Returns:
            Optimized prompt for aging
        """
        # Determine aging category
        if age_years <= 15:
            category = "early"
        elif age_years <= 30:
            category = "middle"
        else:
            category = "advanced"
        
        # Build prompt components
        base = f"The same person aged {age_years} years older"
        
        # Add age-specific features
        features = self.aging_features[category]["features"]
        selected_features = random.sample(features, min(2, len(features)))
        
        # Add gender-specific features if appropriate
        if gender_hint != "general" and age_years > 10:
            gender_features = self.gender_features.get(gender_hint, [])
            if gender_features:
                selected_features.extend(random.sample(gender_features, 1))
        
        # Add style modifiers
        style_modifiers = {
            "natural": "with natural aging progression",
            "graceful": "aging gracefully and elegantly", 
            "distinguished": "with distinguished mature appearance"
        }
        
        # Combine elements
        prompt_parts = [base]
        
        if selected_features:
            prompt_parts.append(f"showing {', '.join(selected_features)}")
            
        if style in style_modifiers:
            prompt_parts.append(style_modifiers[style])
        
        if include_quality:
            quality = random.sample(self.quality_terms, 2)
            prompt_parts.extend(quality)
        
        return ", ".join(prompt_parts)
    
    def generate_negative_prompt(self, detailed: bool = True) -> str:
        """Generate negative prompt to avoid common aging artifacts."""
        base_negative = [
            "blurry", "distorted", "unrealistic", "bad quality", "artifacts",
            "deformed", "disfigured", "mutation", "extra limbs"
        ]
        
        aging_specific_negative = [
            "cartoon", "anime", "painting", "drawing", "illustration",
            "unrealistic aging", "exaggerated wrinkles", "zombie-like",
            "horror", "scary", "grotesque", "unnatural skin texture"
        ]
        
        if detailed:
            return ", ".join(base_negative + aging_specific_negative)
        else:
            return ", ".join(base_negative)


def get_optimized_aging_prompts(age_increments: List[int]) -> Dict[int, Tuple[str, str]]:
    """
    Get optimized prompts for a list of age increments.
    
    Args:
        age_increments: List of years to age
        
    Returns:
        Dictionary mapping age to (positive_prompt, negative_prompt)
    """
    generator = AgingPromptGenerator()
    prompts = {}
    
    for age in age_increments:
        positive = generator.generate_aging_prompt(age)
        negative = generator.generate_negative_prompt()
        prompts[age] = (positive, negative)
    
    return prompts


# Predefined high-quality prompts for common age increments
PREMIUM_AGING_PROMPTS = {
    5: (
        "The same person 5 years older, subtle facial maturity, gentle aging, natural progression, photorealistic, high quality",
        "blurry, distorted, unrealistic, cartoon, bad quality, artifacts"
    ),
    10: (
        "The same person aged 10 years, showing natural aging progression with slight wrinkles around eyes, mature expression, professional photography, detailed, sharp focus",
        "blurry, distorted, unrealistic, bad quality, artifacts, cartoon, anime, exaggerated features"
    ),
    15: (
        "The same person 15 years older, displaying visible wrinkles, crow's feet, natural facial maturity, graceful aging, photorealistic, studio lighting",
        "blurry, distorted, unrealistic, bad quality, artifacts, cartoon, horror, grotesque aging"
    ),
    20: (
        "The same person aged 20 years, showing pronounced facial lines, gray hair beginning, mature features, distinguished appearance, high quality photography",
        "blurry, distorted, unrealistic, bad quality, artifacts, cartoon, anime, zombie-like, unnatural"
    ),
    25: (
        "The same person 25 years older, significant aging with deep wrinkles, gray hair, age-appropriate features, elegant maturity, photorealistic, detailed",
        "blurry, distorted, unrealistic, bad quality, artifacts, cartoon, painting, exaggerated wrinkles"
    )
}


def get_premium_prompt(age_years: int) -> Tuple[str, str]:
    """
    Get a premium hand-crafted prompt for specific age.
    
    Args:
        age_years: Years to age
        
    Returns:
        (positive_prompt, negative_prompt)
    """
    if age_years in PREMIUM_AGING_PROMPTS:
        return PREMIUM_AGING_PROMPTS[age_years]
    
    # Generate dynamic prompt for other ages
    generator = AgingPromptGenerator()
    positive = generator.generate_aging_prompt(age_years)
    negative = generator.generate_negative_prompt()
    
    return positive, negative 