"""Path handling utilities for cross-platform dataset access."""
import os
import glob
from pathlib import Path
from typing import Union, List, Optional, Tuple
from src.core.utils.paths import convert_windows_path, is_windows_path, is_wsl

def convert_path_to_pathlib(path: Union[str, Path], make_absolute: bool = False) -> Path:
    """Convert a path string to a pathlib.Path object."""
    path_str = convert_windows_path(path, make_absolute)
    path = Path(path_str)
    return path.resolve() if make_absolute else path

def convert_paths(paths: Union[str, List[str], Path, List[Path]]) -> Union[Path, List[Path]]:
    """Convert single path or list of paths to proper format."""
    if isinstance(paths, (list, tuple)):
        return [convert_path_to_pathlib(p) for p in paths]
    return convert_path_to_pathlib(paths)

def load_data_from_directory(data_dir: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Load image paths and captions from data directory."""
    # Handle single directory or list of directories
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    
    # Create new lists for storing paths and captions    
    image_paths = []
    captions = []
    
    for directory in data_dir:
        print(f"Processing directory: {directory}")
        
        # Convert Windows paths if needed
        directory = convert_path_to_pathlib(directory)
        
        # Collect all image files for this directory
        dir_image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            pattern = directory / ext
            found_paths = glob.glob(str(pattern))
            dir_image_paths.extend([Path(p) for p in found_paths])
            print(f"Found {len(found_paths)} files with extension {ext}")
        
        print(f"Found {len(dir_image_paths)} total images in {directory}")
        
        # Process each image in this directory
        for img_path in dir_image_paths:
            txt_path = img_path.with_suffix('.txt')
            try:
                caption = txt_path.read_text(encoding='utf-8').strip()
                # Add to main lists
                image_paths.append(str(img_path))
                captions.append(caption)
            except Exception as e:
                print(f"Failed to load caption for {img_path}: {e}")
                continue
    
    # Validate final results
    if not image_paths:
        raise ValueError(f"No valid image-caption pairs found in directories: {data_dir}")
    
    print(f"Successfully loaded {len(image_paths)} image-caption pairs from {len(data_dir)} directories")
    
    return image_paths, captions
