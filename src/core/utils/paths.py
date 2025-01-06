"""Path utilities for handling Windows and WSL paths."""
import os
import platform
import re
from pathlib import Path
from typing import Union, Optional

def is_wsl() -> bool:
    """Check if running under Windows Subsystem for Linux."""
    return 'microsoft-standard' in platform.uname().release.lower()

def is_windows_path(path: Union[str, Path]) -> bool:
    """Check if a path is a Windows-style path."""
    path_str = str(path)
    return bool(re.match(r'^[a-zA-Z]:\\', path_str) or path_str.startswith('\\\\'))

def convert_windows_path(
    path: Union[str, Path],
    make_absolute: bool = False
) -> str:
    """Convert Windows path to WSL path if needed.
    
    Args:
        path: Path to convert
        make_absolute: If True, convert relative paths to absolute
        
    Returns:
        Converted path as string
    """
    path_str = str(path)
    
    # Handle absolute paths
    if is_windows_path(path_str):
        # Convert Windows path to WSL path
        if path_str.startswith('\\\\'):
            # Network path
            path_str = '/mnt/wsl' + path_str.replace('\\', '/')
        else:
            # Local path
            drive = path_str[0].lower()
            path_str = f'/mnt/{drive}' + path_str[2:].replace('\\', '/')
    elif make_absolute and not os.path.isabs(path_str):
        # Convert relative path to absolute
        path_str = os.path.abspath(path_str)
        
    return path_str