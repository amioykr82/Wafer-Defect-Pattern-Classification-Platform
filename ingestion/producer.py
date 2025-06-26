from kafka import KafkaProducer
import json
import base64
import cv2
import os
from dotenv import load_dotenv

load_dotenv()
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_RAW_IMAGES = os.getenv("KAFKA_TOPIC_RAW_IMAGES", "raw_wafer_images")

producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)

def encode_image(img_path):
    img = cv2.imread(img_path)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def stream_images(folder, topic=KAFKA_TOPIC_RAW_IMAGES):
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            image_b64 = encode_image(os.path.join(folder, fname))
            metadata = {'wafer_id': fname.split('_')[0], 'img': image_b64}
            producer.send(topic, json.dumps(metadata).encode('utf-8'))

if __name__ == "__main__":
    stream_images('./data/raw')
