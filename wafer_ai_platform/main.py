import multiprocessing
import os

def run_producer():
    os.system("python ingestion/producer.py")

def run_preproc():
    os.system("python ingestion/consumer_preproc.py")

def run_postproc():
    os.system("python postprocessing/postprocessor.py")

def run_review_queue():
    os.system("uvicorn postprocessing.review_queue:app --reload --port 8008")

if __name__ == "__main__":
    procs = [
        multiprocessing.Process(target=run_producer),
        multiprocessing.Process(target=run_preproc),
        multiprocessing.Process(target=run_postproc),
        multiprocessing.Process(target=run_review_queue),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
