from setfit import AbsaModel
from pprint import pprint
import os
from random import random
import json
import re
import yfinance as yf
import numpy as np
from numpy.linalg import norm
import logging

import networkx as nx
from itertools import combinations

from sentence_transformers import SentenceTransformer


class SemanticSentimentAnalyst:
    def __init__(self):
        self.suppressed_loggers = ["yfinance"]
        self.similarity_threshold = None
        self.company_or_symbol = None
        self.company_symbol_pairs = []
        self.sentiment_directory = None
        self.polarities_symbol_company_list = []
        self.text_comparison_model = None
        self.polarity_model = None
        self.graph = None
        self.suppress_errors()

    def init(self, sentiment_directory, company_or_symbol, similarity_threshold):
        self.sentiment_directory = sentiment_directory
        self.company_or_symbol = company_or_symbol
        self.similarity_threshold = similarity_threshold
        self.polarity_model = AbsaModel.from_pretrained(
            "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
            "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
            spacy_model="en_core_web_lg")
        self.text_comparison_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.graph = nx.Graph()

    def get_embedding(self, text):
        sentences = [text]
        embeddings = self.text_comparison_model.encode(sentences)
        return embeddings

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

    def vector_comparison(self, first_entity, second_entity):
        first_vector_entity = self.get_embedding(first_entity)[0]
        second_vector_entity = self.get_embedding(second_entity)[0]

        return self.cosine_similarity(first_vector_entity, second_vector_entity)

    def encode_polarity_string2numeric(self, data):
        polarity_map = {'negative': -1, 'neutral': 0, 'positive': 1}

        for item_list in data:
            for item in item_list:
                if item['polarity'] in polarity_map:
                    item['polarity'] = polarity_map[item['polarity']]

        return data

    # Function to find the best match for a span
    def get_best_matching_span(self, span, comparison_data):
        if span in comparison_data:
            # Sort the dictionary by values and return the key with the highest value
            return max(comparison_data[span], key=comparison_data[span].get)
        return span  # Return the original span if no match is found

    def map_entities_to_companies_by_comparison(self, polarity_predictions, companies):
        entity_company_matches = {}

        for pred in polarity_predictions:
            for prediction in pred:
                entity_company_matches[f"{prediction['span']}"] = []
                for company in companies:
                    for attribute in ["company_name", "symbol"]:
                        company_value = company[f"{attribute}"]
                        entity_company_matches[f"{prediction['span']}"].append(
                            {f"{company_value}": self.vector_comparison(company_value, prediction['span'])})

        return entity_company_matches

    def filter_entity_company_matches(self, entity_company_matches, similarity_threshold):
        entity_company_thresholded = {}

        for key in entity_company_matches.keys():
            entity_company_thresholded[f"{key}"] = []
            for item in entity_company_matches[f"{key}"]:
                for k, v in item.items():
                    if v > similarity_threshold:
                        entity_company_thresholded[f"{key}"].append({f"{k}": v})
        return entity_company_thresholded

    def replace_entity_with_best_matching_company(self, entity_company_pairs):
        entity_company_maxscore = {}

        for key, value_list in entity_company_pairs.items():
            if value_list:  # Check if the list is not empty
                max_score = max(value_list, key=lambda x: list(x.values())[0])
                entity_company_maxscore[key] = max_score

        return entity_company_maxscore

    def replace_polarity_spans_with_companies(self, polarity_predictions, entity_company_matches):
        new_predictions = []

        # Iterate over the original predictions and create new modified ones
        for prediction in polarity_predictions:
            new_prediction = []
            for item in prediction:
                # Create a new dictionary with the same keys and updated span
                new_item = {key: value for key, value in item.items()}
                new_item['span'] = self.get_best_matching_span(new_item['span'], entity_company_matches)
                new_prediction.append(new_item)
            new_predictions.append(new_prediction)

        return new_predictions

    def tweet_line_company_average_polarity_prediction(self, data):
        # Initialize a dictionary to store the sum and count of polarities for each span
        polarity_sums = {}
        polarity_counts = {}

        # Iterate over each item in the list of dictionaries
        for item in data[0]:
            span = item['span']
            polarity = item['polarity']

            # Update the sum and count for each span
            if span in polarity_sums:
                polarity_sums[span] += polarity
                polarity_counts[span] += 1
            else:
                polarity_sums[span] = polarity
                polarity_counts[span] = 1

        # Calculate the average polarity for each span
        averaged_data = []
        for span in polarity_sums:
            avg_polarity = polarity_sums[span] / polarity_counts[span]
            averaged_data.append({'polarity': avg_polarity, 'span': span})

        # Wrap the result in a list
        result = [averaged_data]
        return result

    def gen_int_random_size(self, size):
        return int((random() * 100) % size)

    def count_lines(self, filename):
        with open(filename, 'r') as file:
            return sum(1 for line in file)

    def join_stock_symbol(self, text):
        return re.sub(r'\$\s+', '$', text)

    def purify(self, text):
        text = text.replace("URL", "")
        text = text.replace("AT_USER", "")

        return text

    # Function to process the file
    def get_random_json_line(self, filename):
        line_count = self.count_lines(filename)

        random_number = self.gen_int_random_size(line_count)

        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, start=0):
                if line_number != random_number:
                    continue

                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on line {line_number}")

    def load_random_tweet(self):
        current_directory = os.getcwd()

        preprocessed_directory = os.path.join(current_directory + "/stocknet-dataset/tweet/preprocessed")

        directories = os.listdir(preprocessed_directory)

        random_directory = os.path.join(preprocessed_directory, directories[self.gen_int_random_size(len(directories))])

        files = os.listdir(random_directory)

        random_file = os.path.join(random_directory, files[self.gen_int_random_size(len(files))])
        # print(random_file)

        random_json_line = self.get_random_json_line(random_file)
        # print(random_json_line["text"])
        joined_text = " ".join(random_json_line["text"])

        return self.join_stock_symbol(joined_text)

    def replace_symbols_with_company_names(self, text):
        # Regular expression to find stock symbols
        symbols_to_be_replaced = re.findall(r"\$\w+", text)
        returned_company_symbols_pairs = []

        for symbol in symbols_to_be_replaced:
            # Remove the dollar sign for the API query
            symbol_without_dollar_sign = symbol[1:]

            # Use a financial data API to get the company name
            company_symbol_pair = self.get_company_name(symbol_without_dollar_sign)
            if company_symbol_pair is None:
                continue

            # company_symbol = company_symbol_pair["symbol"]
            company_name = company_symbol_pair["company_name"]

            # Replace the symbol with the company name in the text
            text = text.replace(symbol, f" {company_name} ,")
            returned_company_symbols_pairs.append(company_symbol_pair)

        return text, returned_company_symbols_pairs

    def suppress_errors(self):
        for loggers in self.suppressed_loggers:
            # Get the logger for yfinance
            logger = logging.getLogger(loggers)

            # Set the logging level to ERROR to suppress warnings and below
            logger.setLevel(logging.CRITICAL)

    def get_company_name(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            company_symbol_pair = {
                "symbol": symbol.upper(),
                "company_name": stock.info['longName']
            }

            self.company_symbol_pairs.append(company_symbol_pair)
            # Fetching company name
            return company_symbol_pair
        except:
            # Return None if company name not found or error occurs
            return None

    def get_polarity_by_span(self, data, span):
        polarity = None
        for hier1 in data:
            for hier2 in hier1:
                # print(hier2)
                if hier2.get('span') == span:
                    # Retrieve the polarity value
                    polarity = hier2.get('polarity')

        return polarity

    def update_graph(self, graph, companies):
        companies = [company[f"{self.company_or_symbol}"] for company in companies]
        # print("Companies", companies)

        # Iterate over all possible pairs of companies
        for company1, company2 in combinations(companies, 2):
            if company1 == company2:
                continue
            # Ensure both companies are nodes in the graph
            if not graph.has_node(company1):
                graph.add_node(company1)
            if not graph.has_node(company2):
                graph.add_node(company2)

            # If an edge already exists, increment the 'mention' attribute
            if graph.has_edge(company1, company2):
                graph[company1][company2]['mention'] += 1
            else:
                # Otherwise, add a new edge with 'mention' set to 1
                graph.add_edge(company1, company2, mention=1)

    def load_tweets_for_date(self, file_date, symbol, second_symbol=""):
        symbol_directory = os.path.join(self.sentiment_directory, symbol)

        filename = os.path.join(symbol_directory, file_date)
        # tweets_list = []
        # companies_list = []
        pprint(filename)

        if second_symbol == "":
            main_company = self.get_company_name(symbol)  # [f"{self.company_or_symbol}"]
        else:
            main_company = self.get_company_name(second_symbol)  # [f"{self.company_or_symbol}"]
        main_company = main_company["company_name"]

        polarity_score = 0
        polarity_count = 0

        self.polarities_symbol_company_list = []

        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, start=0):
                polarity_count, polarity_score = self.process_tweet(line, line_number, main_company, polarity_count,
                                                                    polarity_score)
                # break
        pprint(self.polarities_symbol_company_list)

        main_company_polarity_score = polarity_score / polarity_count if polarity_count != 0 else 0
        # print(f"{main_company} -> main_company_polarity_score: {main_company_polarity_score}")

        return main_company_polarity_score
        # return {"tweets": tweets_list, "companies": companies_list}

    def process_tweet(self, line, line_number, main_company, polarity_count, polarity_score):
        try:
            tweet = json.loads(line)

            joined_tweet_text = " ".join(tweet["text"])
            stock_symbol_tweet = self.join_stock_symbol(joined_tweet_text)
            purified_tweet = self.purify(stock_symbol_tweet)

            named_companies_tweet, companies = self.replace_symbols_with_company_names(purified_tweet)

            companies_list = [company for company in companies]
            self.update_graph(self.graph, companies_list)

            print(50 * "=")
            print("Named Companies Tweet: ")
            pprint(named_companies_tweet)

            # print(50*"=")
            sentences = [named_companies_tweet]
            polarity_predictions = self.polarity_model.predict(sentences)
            polarity_predictions = self.encode_polarity_string2numeric(polarity_predictions)
            print("Polarity Predictions: ")
            pprint(polarity_predictions)

            # print(50*"=")
            entity_company_matches = self.map_entities_to_companies_by_comparison(polarity_predictions,
                                                                                  companies)
            print("Entity Company matches: ")
            pprint(entity_company_matches)

            entity_company_thresholded = self.filter_entity_company_matches(entity_company_matches,
                                                                            self.similarity_threshold)
            print(50*"=")
            print("Thresholded data: ")
            pprint(entity_company_thresholded)


            entity_company_maxscore = self.replace_entity_with_best_matching_company(entity_company_thresholded)
            print(50*"=")
            print("MaxScore data: ")
            pprint(entity_company_maxscore)


            company_polarities = self.replace_polarity_spans_with_companies(polarity_predictions,
                                                                            entity_company_maxscore)
            print(50*"=")
            print("Company Polarities: ")
            pprint(company_polarities)

            average_company_polarities = self.tweet_line_company_average_polarity_prediction(company_polarities)
            print("Average Company Polarities: ")
            pprint(average_company_polarities)

            print("All companies mentioned in tweet line: ", companies)

            self.update_company_polarities(average_company_polarities, companies)

            single_polarity_score = self.get_polarity_by_span(average_company_polarities,
                                                              main_company)
            if single_polarity_score:
                polarity_score += single_polarity_score
                polarity_count += 1

            # tweets_list.append(named_companies_tweet)
            # companies_list.append(companies)
        except json.JSONDecodeError:
            print(f"Error decoding JSON on line {line_number}")

        return polarity_count, polarity_score

    def update_company_polarities(self, average_company_polarities, companies_mentioned):
        # Convert the list of polarities into a dictionary for easier access
        polarity_dict = {item['span']: item['polarity'] for item in average_company_polarities[0]}

        for company in companies_mentioned:
            # Check if the company is already in the list
            existing_entry = next((item for item in self.polarities_symbol_company_list if
                                   item['symbol'] == company['symbol'] and item['company_name'] == company[
                                       'company_name']), None)

            # Calculate the company's polarity score
            company_polarity = polarity_dict.get(company['company_name'], 0)

            if existing_entry:
                # If the company exists, update its polarity score list
                existing_entry['polarity_score'].append(company_polarity)
            else:
                # Otherwise, add a new entry for the company
                self.polarities_symbol_company_list.append({
                    'symbol': company['symbol'],
                    'company_name': company['company_name'],
                    'polarity_score': [company_polarity],
                })

    # def load_random_symbol_n_dates_tweets(self, size):
    #     current_directory = os.getcwd()
    #
    #     preprocessed_directory = os.path.join(current_directory + "/stocknet-dataset/tweet/preprocessed")
    #
    #     directories = os.listdir(preprocessed_directory)
    #
    #     self.random_symbol = directories[self.gen_int_random_size(len(directories))]
    #     # random_symbol = "SPLP"
    #
    #     random_directory = os.path.join(preprocessed_directory, self.random_symbol)
    #
    #     # dates_tweets_companies = {}
    #
    #     files = sorted(os.listdir(random_directory))
    #     pprint(random_directory)
    #
    #     for counter, file in enumerate(files, start=1):
    #
    #         # dates_tweets_companies[file] = load_tweets_for_date(filename, random_symbol)
    #         self.load_tweets_for_date(random_directory, file, self.random_symbol)
    #
    #         print(50 * "=")
    #         if (counter == size):
    #             break

    def find_common_mentioned_companies(self, symbol, n_common):
        # Assuming G is your graph and 'target_node' is the node you're interested in
        top_nodes = []

        company_symbol_pair = self.get_company_name(symbol)
        target_node = company_symbol_pair[f"{self.company_or_symbol}"]

        # Check if the target node is in the graph
        if target_node in self.graph:
            # Get edges to the target node with attributes
            edges = self.graph.in_edges(target_node, data=True) if self.graph.is_directed() else self.graph.edges(
                target_node, data=True)

            # Sort the edges based on 'mentions' attribute and get the top 5
            top_edges = sorted(edges, key=lambda x: x[2].get('mention', 0), reverse=True)[:n_common]

            # # Extract the nodes from the top edges
            for edge in top_edges:
                top_nodes.append({
                    f"{self.company_or_symbol}": edge[1],
                    "mention": edge[2]["mention"],
                })

            # print(f"Top {n_common} nodes connected to", target_node, "by mentions:", top_nodes)
        else:
            print("Node not found in the graph.", target_node)

        return top_nodes

    def find_sentiment_for_companies(self, date, original_symbol, companies, n_common):
        for company in companies:
            symbol = company['symbol']
            try:
                company['polarity_score'] = self.load_tweets_for_date(date, original_symbol, symbol)
            except FileNotFoundError:
                print(f"Could not find tweets for company {symbol}")

        pprint(companies)
