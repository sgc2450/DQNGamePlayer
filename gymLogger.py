import logging
import os
import sys

# Create or get a logger
logger = logging.getLogger('model_training')

# Disable propagation to avoid duplication
logger.propagate = False

# Check if handlers are already present to avoid duplication
if not logger.hasHandlers():
    # Clear any existing handlers
    logger.handlers.clear()

    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create handlers for file and console
    file_handler = logging.FileHandler('training.log')
    console_handler = logging.StreamHandler(sys.stdout)

    # Set the level for handlers
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)