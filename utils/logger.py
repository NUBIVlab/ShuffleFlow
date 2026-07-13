import logging
import sys


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    return logger