"""
Simple consumer which polls from kafka topic and analyses on input data.
Currently the stream is simple continuous string.
This could also be replaced to handle different metrics to provide better result.
"""
import logging
import os
import subprocess

from confluent_kafka import Consumer, KafkaError
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.prediction import TrainingModel
from src.customlogger import ColoredLogger

logging.setLoggerClass(ColoredLogger)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

config = {'bootstrap.servers': '172.18.0.1:32770', 'group.id': 'test-consumer-group',
          'default.topic.config': {'auto.offset.reset': 'smallest'}}

consumer = Consumer(**config)
consumer.subscribe(['test'])

pool = ThreadPoolExecutor(10)
running = True
batch_messages = []


def get_predictive_result(batch_messages):
    logging.debug("Processing request for analysis of predictive popularity")
    abc = TrainingModel()
    return abc.is_video_going_to_viral(batch_messages)

try:
    while running:
        msg = consumer.poll()

        if not msg.error():
            batch_messages.append(msg.value().decode('utf-8'))
            if len(batch_messages) > 20:
                logging.debug("Processing on stream of messages %s", batch_messages)
                futures = [pool.submit(get_predictive_result, batch_messages)]
                results = [r.result() for r in as_completed(futures)]
                # Auto scale is handled by docker
                if results and results[0]:
                    logging.debug("Prediction algo. result on popularity is True\n")
                    logging.debug("Scaling up Kafka nodes by 4 %s", subprocess.Popen("./kafka-cluster.sh scale 4", shell=True, stdout=subprocess.PIPE).stdout.read())
                    logging.debug("Successfully scaled Kafka Nodes")
                    logging.debug("Verifying total Kafka Nodes %s", subprocess.Popen("docker ps", shell=True, stdout=subprocess.PIPE).stdout.read())
                del batch_messages[:]
        elif msg.error().code() != KafkaError._PARTITION_EOF:
            print(msg.error())
            running = False
except KeyboardInterrupt, SystemExit:
    consumer.close()