"""
Simple producer which streams random view count to topic.
This could be replaced to fetch statistics from YouTube API and stream it to topic.
"""

import random
import logging
import os

from confluent_kafka import Producer

from src.customlogger import ColoredLogger

logging.setLoggerClass(ColoredLogger)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


config = {'bootstrap.servers': '172.18.0.1:32770'}
producer = Producer(**config)
data_source = [str(random.randint(10, 100)) for _ in range(21)]
topic = 'test'

for data in data_source:
    logging.debug("Sending view count to to topic %s", topic)
    producer.produce(topic, data.encode('utf-8'))

producer.flush()
