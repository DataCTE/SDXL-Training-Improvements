"""Path handling utilities for cross-platform dataset access."""
import os
import platform
import re
from pathlib import Path, PureWindowsPath
from typing import Union, List, Optional, Tuple
from src.core.logging import get_logger, LogConfig
import glob

logger = get_logger(__name__)

def is_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux."""
    return "microsoft-standard-WSL" in platform.uname().release

def is_windows_path(path: Union[str, Path]) -> bool:
    """Check if path is a Windows-style path."""
    path_str = str(path)
    # Match patterns like C:, D:\, \\server\share, relative Windows paths, and dot paths
    return bool(re.match(r'^[a-zA-Z]:|^\\\\|\\|^[\w.]+$', path_str))

def convert_windows_path(path: Union[str, Path], make_absolute: bool = False) -> Path:
    """Convert Windows path to proper format for current system."""
    try:
        # Convert to string if Path object
        path_str = str(path)
        
        # Skip if already proper format
        if not is_windows_path(path_str):
            path = Path(path_str)
            return path.resolve() if make_absolute else path
            
        # Handle Windows-style paths
        if "\\" in path_str:
            path_str = path_str.replace("\\", "/")
            
        # Handle drive letters in WSL
        if ":" in path_str and is_wsl():
            drive_letter = path_str[0].lower()
            path_without_drive = path_str[2:]  # Remove "C:"
            mount_point = get_wsl_drive_mount() or "/mnt"
            path_str = f"{mount_point}/{drive_letter}{path_without_drive}"
            
        path = Path(path_str)
        return path.resolve() if make_absolute else path
        
    except Exception as e:
        logger.error(f"Path conversion failed: {str(e)}")
        return Path(str(path))  # Return original if conversion fails

def convert_paths(paths: Union[str, List[str], Path, List[Path]]) -> Union[Path, List[Path]]:
    """Convert single path or list of paths to proper format."""
    if isinstance(paths, (list, tuple)):
        return [convert_windows_path(p) for p in paths]
    return convert_windows_path(paths)

def get_wsl_drive_mount() -> Optional[str]:
    """Get the WSL drive mount point.
    
    Returns:
        Mount point (e.g. "/mnt") or None if not in WSL
    """
    if not is_wsl():
        return None
        
    # Check common mount points
    mount_points = ["/mnt"]
    for mp in mount_points:
        if os.path.isdir(mp):
            return mp
    return None

def load_data_from_directory(data_dir: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Load image paths and captions from data directory.
    
    Returns copies of paths and captions to prevent dataset corruption.
    """
    # Handle single directory or list of directories
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    
    # Create new lists for storing paths and captions    
    image_paths = []
    captions = []
    
    for directory in data_dir:
        dir_image_paths = []
        # Collect image files for this directory
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            # Use str() to create new string objects
            dir_image_paths.extend([str(p) for p in glob.glob(os.path.join(directory, ext))])
        
        # Load corresponding captions for this directory's images
        dir_captions = []
        for img_path in dir_image_paths:
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    # Create new string object for caption
                    caption = str(f.read().strip())
                    dir_captions.append(caption)
                # Only add copies of paths if caption was successfully loaded
                image_paths.append(str(img_path))
                captions.append(caption)
            except Exception as e:
                logger.warning(f"Failed to load caption for {img_path}: {e}")
                continue
    
    if not image_paths:
        logger.error("No valid image-caption pairs found in directories: %s", data_dir)
        raise ValueError("No valid image-caption pairs found")
    
    # Create final copies of lists to ensure complete isolation
    image_paths = image_paths.copy()
    captions = captions.copy()
    
    logger.info(f"Loaded {len(image_paths)} image-caption pairs from {len(data_dir)} directories")
    
    # Return copies of everything to ensure complete isolation
    return image_paths, captions
