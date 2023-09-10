import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter to include method name and add color"""

    COLORS = {
        'WARNING': '\033[93m',
        'INFO': '\033[94m',
        'DEBUG': '\033[92m',
        'CRITICAL': '\033[91m',
        'ERROR': '\033[91m'
    }

    RESET = '\033[0m'

    def format(self, record):
        log_format = f"[training_script][{record.funcName}] {record.msg}"
        log_color = self.COLORS.get(record.levelname, self.RESET)
        return f"{log_color}{log_format}{self.RESET}"


# Initialize the logger
training_logger = logging.getLogger("my_logger")
training_logger.setLevel(logging.DEBUG)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Set the custom formatter to the console handler
ch.setFormatter(CustomFormatter())

# Add console handler to the logger
training_logger.addHandler(ch)
