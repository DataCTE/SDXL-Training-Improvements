"""Logging formatters with color support."""
import logging
from colorama import Fore, Style
import colorama

# Initialize colorama
colorama.init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Enhanced formatter with colored output and context tracking."""
    
    COLORS = {
        'DEBUG': Fore.CYAN + Style.DIM,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW + Style.BRIGHT,
        'ERROR': Fore.RED + Style.BRIGHT,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT + Style.DIM
    }

    HIGHLIGHT_COLORS = {
        'file_path': Fore.BLUE,
        'line_number': Fore.CYAN,
        'function': Fore.MAGENTA,
        'error': Fore.RED + Style.BRIGHT,
        'success': Fore.GREEN + Style.BRIGHT,
        'warning': Fore.YELLOW + Style.BRIGHT
    }

    KEYWORDS = {
        'start': (Fore.CYAN, ['Starting', 'Initializing', 'Beginning']),
        'success': (Fore.GREEN, ['Complete', 'Finished', 'Saved', 'Success']),
        'error': (Fore.RED, ['Error', 'Failed', 'Exception']),
        'warning': (Fore.YELLOW, ['Warning', 'Caution']),
        'progress': (Fore.BLUE, ['Processing', 'Loading', 'Computing'])
    }

    def __init__(self, fmt=None, datefmt=None, colored=True):
        super().__init__(fmt or '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                        datefmt or '%Y-%m-%d %H:%M:%S')
        self.colored = colored

    def format(self, record):
        if not self.colored:
            return super().format(record)
            
        # Create a copy of the record
        filtered_record = logging.makeLogRecord(record.__dict__)
        
        # Get base color for level
        base_color = self.COLORS.get(record.levelname, '')
        
        # Format with parent
        formatted_message = super().format(filtered_record)
        
        # Apply keyword highlighting
        for keyword, (color, words) in self.KEYWORDS.items():
            for word in words:
                if word in formatted_message:
                    formatted_message = formatted_message.replace(
                        word, f"{color}{word}{Style.RESET_ALL}")
                    setattr(record, 'keyword', keyword)
        
        # Apply context highlighting
        for context, color in self.HIGHLIGHT_COLORS.items():
            if hasattr(record, context):
                value = getattr(record, context)
                formatted_message = formatted_message.replace(
                    str(value), f"{color}{value}{Style.RESET_ALL}")
        
        return f"{base_color}{formatted_message}{Style.RESET_ALL}"