"""Path handling utilities for cross-platform dataset access."""
import os
import platform
import re
from pathlib import Path, PureWindowsPath
from typing import Union, List

def is_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux."""
    return "microsoft-standard-WSL" in platform.uname().release

def is_windows_path(path: Union[str, Path]) -> bool:
    """Check if path is a Windows-style path."""
    path_str = str(path)
    # Match patterns like C:, D:\, \\server\share
    return bool(re.match(r'^[a-zA-Z]:|^\\\\', path_str))

def convert_windows_path(path: Union[str, Path]) -> Path:
    """Convert Windows path to WSL path if running in WSL."""
    if not isinstance(path, (str, Path)):
        return path
        
    path_str = str(path)
    
    # Only convert if in WSL and path is Windows-style
    if is_wsl() and is_windows_path(path_str):
        # Handle UNC paths (\\server\share)
        if path_str.startswith('\\\\'):
            path_str = path_str.replace('\\', '/')
            return Path(f"/mnt/wsl/wslg{path_str}")
            
        # Handle regular Windows paths (C:\path\to\file)
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return Path(f"/mnt/{drive}/{rest}")
            
    return Path(path_str)

def convert_path_list(paths: Union[str, List[str], Path, List[Path]]) -> List[Path]:
    """Convert list of Windows/Unix paths to appropriate format."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
        
    return [convert_windows_path(p) for p in paths]
