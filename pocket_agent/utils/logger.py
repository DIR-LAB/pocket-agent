import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union

class PocketAgentLogger:
    """Professional logging configuration for PocketAgent framework"""
    
    def __init__(self, 
                 name: str = "pocket_agent",
                 level: Union[str, int] = logging.INFO,
                 log_dir: Optional[Path] = None,
                 console: bool = True,
                 file_logging: bool = True,
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5):
        
        self.name = name
        self.level = self._parse_level(level)
        self.log_dir = log_dir or Path.cwd()
        self.console = console
        self.file_logging = file_logging
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create the main logger (don't modify root!)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _parse_level(self, level: Union[str, int]) -> int:
        if isinstance(level, str):
            return getattr(logging, level.upper(), logging.INFO)
        return level
    
    def _setup_handlers(self):
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.level)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.file_logging:
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.level)
            self.logger.addHandler(file_handler)
    
    def get_logger(self, component: str = "") -> logging.Logger:
        """Get a component-specific logger"""
        if component:
            logger_name = f"{self.name}.{component}"
        else:
            logger_name = self.name
        
        return logging.getLogger(logger_name)
    
    def configure_third_party_loggers(self):
        """Configure third-party library loggers without interfering"""
        third_party_configs = {
            'fastmcp': logging.WARNING,
            'mcp': logging.WARNING, 
            'litellm': logging.WARNING,
            'urllib3': logging.WARNING,  # Often noisy
            'httpx': logging.WARNING,
        }
        
        for logger_name, level in third_party_configs.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)

def _detect_environment():
    """Detect if we're in development vs production"""
    # Common development indicators
    dev_indicators = [
        os.getenv("DEBUG"),
        os.getenv("DEVELOPMENT"), 
        os.path.exists(".git"),  # In a git repo
        "pytest" in sys.modules,  # Running tests
        "jupyter" in sys.modules,  # In Jupyter
    ]
    return any(dev_indicators)

def configure_logger(log_level: str = "INFO", console_level: str = "WARNING"):
    """
    Configure hierarchical logging for PocketAgent framework
    
    Args:
        log_level: Overall logging level (affects file output)
        console_level: Console output level (default: WARNING, so only warnings/errors show)
    """
    log_level = os.getenv("POCKET_AGENT_LOG_LEVEL", log_level)
    console_level = os.getenv("POCKET_AGENT_CONSOLE_LEVEL", console_level)
    
    log_level = log_level.upper()
    console_level = console_level.upper()
    
    log_level_int = logging.getLevelNamesMapping().get(log_level, logging.INFO)
    console_level_int = logging.getLevelNamesMapping().get(console_level, logging.WARNING)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get or create the main logger
    logger = logging.getLogger("pocket_agent")
    logger.setLevel(log_level_int)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Add console handler - WARNING+ by default (errors/warnings visible immediately)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level_int)  # Only warnings/errors by default
    logger.addHandler(console_handler)
    
    # Add file handler - full detail based on log_level
    file_handler = logging.handlers.RotatingFileHandler(
        'pocket-agent.log', 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level_int)  # Full detail in file
    logger.addHandler(file_handler)
    
    # Configure third-party loggers (keep them quiet)
    for lib_name in ['fastmcp', 'mcp', 'litellm', 'urllib3', 'httpx']:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)
    
    return logger