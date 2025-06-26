from kafka import KafkaConsumer, KafkaProducer
import json
import os
from dotenv import load_dotenv

load_dotenv()
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", 0.95))
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

consumer = KafkaConsumer('inference_results', bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)

for msg in consumer:
    result = json.loads(msg.value.decode('utf-8'))
    if result['confidence'] < REVIEW_THRESHOLD:
        producer.send("manual_review_queue", msg.value)
    else:
        producer.send("defect_map_results", msg.value)
