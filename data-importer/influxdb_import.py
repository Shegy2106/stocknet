import time
import os
from pathlib import Path
import pandas as pd
from retry import retry
from circuitbreaker import circuit
import influxdb_client
from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = os.environ.get("INFLUXDB_TOKEN")
org = os.environ.get("INFLUXDB_ORG")
bucket = os.environ.get("INFLUXDB_BUCKET")
url = "http://influxdb:8086"

# Initialize the client
@retry(tries=3)
@circuit(failure_threshold=1, recovery_timeout=5)
def init_client(url, token, org):
    try:
        client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
        return client.write_api(write_options=SYNCHRONOUS)
    except Exception as e:
        print(f"Error creating InfluxDB client: {e}")
        raise

# Decorator for circuit breaker pattern
@retry(tries=3)
@circuit(failure_threshold=1, recovery_timeout=10)
def safe_import_csv(file_path, write_api, url, token, org):
    try:
        import_csv(file_path, write_api)
    except Exception as e:
        print(f"Error importing CSV: {e}")
        # Reinitialize client and retry
        new_write_api = init_client(url, token, org)
        import_csv(file_path, new_write_api)

# Function to import CSV data
@retry(tries=10)
@circuit(failure_threshold=1, recovery_timeout=10)
def import_csv(file_path, write_api):
    try:
        df = pd.read_csv(file_path)
        symbol = Path(file_path).stem

        for index, row in df.iterrows():
            point = Point("stock_data") \
                .tag("symbol", symbol) \
                .field("open", float(row['Open'])) \
                .field("high", float(row['High'])) \
                .field("low", float(row['Low'])) \
                .field("close", float(row['Close'])) \
                .field("adj_close", float(row['Adj Close'])) \
                .field("volume", float(row['Volume'])) \
                .time(pd.to_datetime(row['Date']), WritePrecision.NS)

            write_api.write(bucket=bucket, org=org, record=point)
            time.sleep(1)
    except Exception as e:
        print(f"Error importing CSV: {e}")

time.sleep(10)
# Main execution
write_api = init_client(url, token, org)
directory = "./stocknet-dataset/price/raw"
for filename in os.listdir(directory):
    csv_file = os.path.join(directory, filename)
    if os.path.isfile(csv_file):
        safe_import_csv(csv_file, write_api, url, token, org)
