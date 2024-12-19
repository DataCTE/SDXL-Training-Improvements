"""Path handling utilities for cross-platform dataset access."""
import os
import platform
from pathlib import Path, PureWindowsPath
from typing import Union, List

def convert_windows_path(path: Union[str, Path]) -> Path:
    """Convert Windows path to WSL path if running in WSL."""
    if not isinstance(path, (str, Path)):
        return path
        
    path_str = str(path)
    
    # Check if running in WSL
    if "microsoft-standard-WSL" in platform.uname().release:
        if isinstance(path, str) and ":" in path:
            # Convert Windows path (e.g. D:\Dataset) to WSL path (/mnt/d/Dataset)
            drive = path_str[0].lower()
            rest = path_str[2:].replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
            
    return Path(path_str)

def convert_path_list(paths: Union[str, List[str], Path, List[Path]]) -> List[Path]:
    """Convert list of Windows/Unix paths to appropriate format."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
        
    return [convert_windows_path(p) for p in paths]
