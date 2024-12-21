"""Path handling utilities for cross-platform dataset access."""
import os
import platform
import re
from pathlib import Path, PureWindowsPath
from typing import Union, List, Optional

def is_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux."""
    return "microsoft-standard-WSL" in platform.uname().release

def is_windows_path(path: Union[str, Path]) -> bool:
    """Check if path is a Windows-style path."""
    path_str = str(path)
    # Match patterns like C:, D:\, \\server\share, relative Windows paths, and dot paths
    return bool(re.match(r'^[a-zA-Z]:|^\\\\|\\|^[\w.]+$', path_str))

def convert_windows_path(path: Union[str, Path, List[Union[str, Path]]], make_absolute: bool = True) -> Union[Path, List[Path]]:
    """Convert Windows path(s) to WSL path if running in WSL.
    
    Args:
        path: Windows, Unix or dot-separated path(s)
        make_absolute: Whether to convert to absolute path
        
    Returns:
        Converted Path object(s)
        
    Raises:
        ValueError: If path is invalid or cannot be converted
    """
    # Handle list/tuple input recursively
    if isinstance(path, (list, tuple)):
        converted = []
        for p in path:
            try:
                result = convert_windows_path(p, make_absolute)
                if isinstance(result, list):
                    converted.extend(result)
                else:
                    converted.append(result)
            except ValueError as e:
                logger.warning(f"Skipping invalid path: {str(e)}")
                continue
        return converted if converted else [Path()]
        
    if not isinstance(path, (str, Path)):
        raise ValueError(f"Path must be string, Path, or list of paths, not {type(path)}")
        
    # Convert to string and normalize
    try:
        path_str = os.path.normpath(str(path))
    except Exception as e:
        raise ValueError(f"Failed to normalize path {path}: {str(e)}")
    
    # Handle dot-separated paths
    if re.match(r'^[\w.]+$', path_str):
        try:
            path_str = path_str.replace('.', os.path.sep)
        except Exception as e:
            raise ValueError(f"Failed to convert dot path {path_str}: {str(e)}")
    
    # Skip WSL conversion if not needed
    if not is_wsl() or not is_windows_path(path_str):
        try:
            return Path(os.path.normpath(path_str))
        except Exception as e:
            raise ValueError(f"Failed to create Path from {path_str}: {str(e)}")
        
    try:
        # Handle UNC paths (\\server\share)
        if path_str.startswith('\\\\'):
            path_str = path_str.replace('\\', '/')
            return Path(f"/mnt/wsl/wslg{path_str}")
                
        # Handle absolute Windows paths (C:\path\to\file)
        if re.match(r'^[a-zA-Z]:', path_str):
            drive = path_str[0].lower()
            rest = path_str[2:].replace('\\', '/')
            return Path(f"/mnt/{drive}/{rest}")
                
        # Handle relative Windows paths if make_absolute=True
        if make_absolute and '\\' in path_str:
            try:
                # Get current working directory and normalize
                cwd = os.getcwd()
                if cwd.startswith('/mnt/'):
                    base_path = cwd
                else:
                    # If not in a mounted drive, use outputs/wslref as base
                    base_path = os.path.join(os.getcwd(), 'outputs', 'wslref')
                    os.makedirs(base_path, exist_ok=True)
                        
                # Convert relative path using normalized base
                rel_path = path_str.replace('\\', '/')
                return Path(os.path.normpath(os.path.join(base_path, rel_path)))
            except Exception as e:
                raise ValueError(f"Failed to handle relative path {path_str}: {str(e)}")
                    
        # Handle remaining cases
        try:
            return Path(os.path.normpath(path_str.replace('\\', '/')))
        except Exception as e:
            raise ValueError(f"Failed to create final Path from {path_str}: {str(e)}")
                
    except Exception as e:
        if not isinstance(e, ValueError):
            raise ValueError(f"Error converting Windows path {path_str}: {str(e)}")
        raise

def convert_path_list(paths: Union[str, List[str], Path, List[Path]], make_absolute: bool = True) -> List[Path]:
    """Convert list of Windows/Unix paths to appropriate format.
    
    Args:
        paths: Single path or list of paths
        make_absolute: Whether to convert to absolute paths
        
    Returns:
        List of converted Path objects
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
        
    return [convert_windows_path(p, make_absolute) for p in paths]

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
