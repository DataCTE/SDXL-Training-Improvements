"""Core logging functionality."""
from typing import Optional, Dict, Any, List, Union
import logging
import sys
import threading
from pathlib import Path
import torch
import wandb
from datetime import datetime
from functools import wraps
import colorama
from colorama import Fore, Style

from .config import LogConfig
from .wandb import WandbLogger

colorama.init(autoreset=True)

class LogManager:
    """Centralized logging manager."""
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._loggers = {}
        self._config = None
        self._wandb_logger = None
        
    @classmethod
    def get_instance(cls) -> 'LogManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def get_logger(self, name: str, config: Optional[LogConfig] = None) -> 'Logger':
        with self._lock:
            if name not in self._loggers:
                self._loggers[name] = Logger(name, config or self.config)
            return self._loggers[name]
            
    @property
    def config(self) -> LogConfig:
        if self._config is None:
            self._config = LogConfig()
        return self._config

class Logger:
    """Unified logger combining console, file, and metrics tracking."""
    def __init__(self, name: str, config: LogConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(name)
        self._setup_logging()
        
    def _setup_logging(self):
        """Initialize logging with handlers."""
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(ColoredFormatter())
            console_handler.setLevel(getattr(logging, self.config.console_level.upper()))
            self.logger.addHandler(console_handler)
            
        if self.config.file_output:
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_path / self.config.filename,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, self.config.file_level.upper()))
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            ))
            self.logger.addHandler(file_handler)

    # Delegate basic logging methods
    def debug(self, msg: str, *args, **kwargs): self.logger.debug(msg, *args, **kwargs)
    def info(self, msg: str, *args, **kwargs): self.logger.info(msg, *args, **kwargs)
    def warning(self, msg: str, *args, **kwargs): self.logger.warning(msg, *args, **kwargs)
    def error(self, msg: str, *args, **kwargs): self.logger.error(msg, *args, **kwargs)

