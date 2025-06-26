from kafka import KafkaConsumer, KafkaProducer
import json
import numpy as np
import cv2
import base64
import os
from dotenv import load_dotenv

load_dotenv()
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW_IMAGES = os.getenv("KAFKA_TOPIC_RAW_IMAGES", "raw_wafer_images")
KAFKA_TOPIC_PREPROC = os.getenv("KAFKA_TOPIC_PREPROC", "preprocessed_images")

consumer = KafkaConsumer(KAFKA_TOPIC_RAW_IMAGES, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)

def preprocess(img_b64):
    img_np = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img_np, (64, 64))
    return img_resized.astype(np.float32) / 255.0

for msg in consumer:
    data = json.loads(msg.value.decode('utf-8'))
    img_processed = preprocess(data['img'])
    payload = {
        "wafer_id": data["wafer_id"],
        "tensor": img_processed.tolist()
    }
    producer.send(KAFKA_TOPIC_PREPROC, json.dumps(payload).encode('utf-8'))
