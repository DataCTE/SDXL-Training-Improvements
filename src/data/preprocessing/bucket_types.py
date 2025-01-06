"""Shared type definitions for bucket handling."""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class BucketDimensions:
    """Explicit storage of all dimension-related information."""
    width: int
    height: int
    width_latent: int
    height_latent: int
    aspect_ratio: float
    aspect_ratio_inverse: float
    total_pixels: int
    total_latents: int
    
    @classmethod
    def from_pixels(cls, width: int, height: int) -> 'BucketDimensions':
        """Create dimensions from pixel values with validation."""
        try:
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")
            
            if width % 8 != 0 or height % 8 != 0:
                raise ValueError(f"Dimensions must be divisible by 8: {width}x{height}")
                
            return cls(
                width=width,
                height=height,
                width_latent=width // 8,
                height_latent=height // 8,
                aspect_ratio=width / height,
                aspect_ratio_inverse=height / width,
                total_pixels=width * height,
                total_latents=(width // 8) * (height // 8)
            )
        except Exception as e:
            raise ValueError(f"Failed to create bucket dimensions: {str(e)}")
    
    def validate(self) -> bool:
        """Validate internal consistency of dimensions."""
        checks = [
            self.width > 0,
            self.height > 0,
            self.width_latent == self.width // 8,
            self.height_latent == self.height // 8,
            np.isclose(self.aspect_ratio, self.width / self.height),
            np.isclose(self.aspect_ratio_inverse, 1 / self.aspect_ratio),
            self.total_pixels == self.width * self.height,
            self.total_latents == self.width_latent * self.height_latent
        ]
        return all(checks)
    
    def validate_with_details(self) -> Tuple[bool, Optional[str]]:
        """Validate dimensions with detailed error reporting."""
        checks = [
            (self.width > 0, "Width must be positive"),
            (self.height > 0, "Height must be positive"),
            (self.width_latent == self.width // 8, "Invalid latent width"),
            (self.height_latent == self.height // 8, "Invalid latent height"),
            (np.isclose(self.aspect_ratio, self.width / self.height), "Invalid aspect ratio"),
            (self.total_pixels == self.width * self.height, "Invalid pixel count"),
            (self.total_latents == self.width_latent * self.height_latent, "Invalid latent count")
        ]
        
        for check, message in checks:
            if not check:
                return False, message
        return True, None

@dataclass
class BucketInfo:
    """Comprehensive bucket information with redundant storage."""
    dimensions: BucketDimensions    # All dimension-related information
    pixel_dims: Tuple[int, int]    # Redundant pixel dimensions (w, h)
    latent_dims: Tuple[int, int]   # Redundant latent dimensions (w//8, h//8)
    bucket_index: int              # Index in bucket list
    size_class: str               # Size classification
    aspect_class: str            # Aspect ratio classification
    
    @property
    def total_pixels(self) -> int:
        """Get total pixel count."""
        return self.dimensions.total_pixels
        
    @property
    def total_latents(self) -> int:
        """Get total latent count."""
        return self.dimensions.total_latents
        
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self.dimensions.aspect_ratio
    
    @classmethod
    def from_dims(cls, width: int, height: int, bucket_index: int) -> 'BucketInfo':
        """Create BucketInfo with full validation."""
        dimensions = BucketDimensions.from_pixels(width, height)
        
        # Validate dimension consistency
        if not dimensions.validate():
            raise ValueError(f"Invalid dimensions for bucket: {width}x{height}")
        
        # Classify bucket
        size_class = cls._classify_size(dimensions.total_pixels)
        aspect_class = cls._classify_aspect(dimensions.aspect_ratio)
        
        return cls(
            dimensions=dimensions,
            pixel_dims=(width, height),
            latent_dims=(width // 8, height // 8),
            bucket_index=bucket_index,
            size_class=size_class,
            aspect_class=aspect_class
        )
    
    @staticmethod
    def _classify_size(total_pixels: int) -> str:
        """Classify bucket by total pixels."""
        if total_pixels < 512 * 512:
            return "small"
        elif total_pixels < 1024 * 1024:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def _classify_aspect(ratio: float) -> str:
        """Classify bucket by aspect ratio."""
        if np.isclose(ratio, 1.0, atol=0.1):
            return "square"
        elif ratio > 1.0:
            return "landscape"
        else:
            return "portrait"
    
    def validate(self) -> bool:
        """Comprehensive validation of all stored information."""
        try:
            # Validate dimensions object
            if not self.dimensions.validate():
                return False
            
            # Validate consistency between redundant storage
            checks = [
                self.pixel_dims == (self.dimensions.width, self.dimensions.height),
                self.latent_dims == (self.dimensions.width_latent, self.dimensions.height_latent),
                self.dimensions.aspect_ratio == self.pixel_dims[0] / self.pixel_dims[1],
                self._classify_size(self.dimensions.total_pixels) == self.size_class,
                self._classify_aspect(self.dimensions.aspect_ratio) == self.aspect_class
            ]
            return all(checks)
            
        except Exception:
            return False 