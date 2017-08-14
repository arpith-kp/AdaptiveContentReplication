import logging
import os

from src.customlogger import ColoredLogger

logging.setLoggerClass(ColoredLogger)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))


for _ in range(10):
    logging.debug("PRints")
    logging.info("23")
