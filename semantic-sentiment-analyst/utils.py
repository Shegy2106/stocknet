import json
import requests
def load_dictionary_to_json(dictionary, filename=''):
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file)

def load_json_to_dictionary(filename):
    with open(filename, 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data

def get_earnings(ticker):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey=B0RLXVWXQMHH0LQD'
    r = requests.get(url)
    data = r.json()
    print(data)
    return data['quarterlyEarnings']