import time
import os
from pathlib import Path
import pandas as pd
from retry import retry
from circuitbreaker import circuit
from elasticsearch import Elasticsearch
import json

elastic_password = os.environ.get("ELASTIC_PASSWORD")
url = "http://elasticsearch:9200"

# @retry(tries=3)
# @circuit(failure_threshold=1, recovery_timeout=5)
def init_client(url, password):
    try:
        print("Creating elasticsearch")
        return Elasticsearch(url, basic_auth=("elastic", password)) 
    except Exception as e:
        print(f"Error creating Elasticsearch client: {e}")

# @retry(tries=3)
# @circuit(failure_threshold=1, recovery_timeout=10)
def safe_import_json(file_path, database_client, url, password, directory, filename):
    try:
        send_json_to_db(file_path, database_client, directory, filename)
    except Exception as e:
        print(f"Error importing JSON: {e}")

        database_client = init_client(url, password)
        send_json_to_db(file_path, database_client, directory, filename)


# @retry(tries=10)
# @circuit(failure_threshold=1, recovery_timeout=10)
def send_json_to_db(json_file_path, database_client, directory, filename):
    try:
        print("OPEN")
        with open(json_file_path, 'r') as file:
            file_contents = file.read()

            symbol = Path(directory).stem.lower()
            timestamp = filename
            indexes = [symbol, timestamp]
            lines = file_contents.split('\n')
            print(f"Indexing tweet made at {timestamp}")
            index_tweets(lines, indexes, database_client)
            
    except Exception as e:
        print(f"Error loading JSON file: {e}")

def index_tweets(lines, indexes, database_client):
    for line in lines:
        if not line.strip():
            continue

        tweet = json.loads(line)
        for index in indexes:
            print(f"Indexing tweet in: {index}")
            response = database_client.index(index=index, document=tweet)

    

time.sleep(30)

database_client = init_client(url, elastic_password)
directory = "./stocknet-dataset/tweet/preprocessed"
for tier1 in os.listdir(directory):
    path_tier1 = os.path.join(directory, tier1)
    for filename in os.listdir(path_tier1):
        json_path = os.path.join(path_tier1, filename)
        print(json_path)
        if os.path.isfile(json_path):
            safe_import_json(json_path, database_client, url, elastic_password, tier1, filename)
        
        break
    break
