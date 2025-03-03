import os
import logging

# Set up the directory for log files
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create file handlers for different log levels
info_handler = logging.FileHandler(os.path.join(log_dir, "info.log"))
warn_handler = logging.FileHandler(os.path.join(log_dir, "warn.log"))
error_handler = logging.FileHandler(os.path.join(log_dir, "error.log"))

# Set log level for each handler
info_handler.setLevel(logging.INFO)
warn_handler.setLevel(logging.WARNING)
error_handler.setLevel(logging.ERROR)

# Set the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Apply formatter to handlers
info_handler.setFormatter(formatter)
warn_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # This allows all levels of logs

# Add handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(warn_handler)
logger.addHandler(error_handler)

# Optionally: Disable console logging
logger.propagate = False  # This prevents logs from appearing in the console

# You can also add a stream handler if you need console logs in the future.
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.DEBUG)
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)


