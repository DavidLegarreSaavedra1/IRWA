import datetime
from random import random
import numpy as np

from faker import Faker
from app.search_engine.algorithms import create_corpus

fake = Faker()

class Document:
    def __init__(self, id, title, tweet, username, date, hashtags, likes, retweets, url, details=None, query=None):
        self.id = id
        self.title = title
        self.tweet = tweet
        self.username = username
        self.date = date
        self.hashtags = hashtags
        self.likes = likes
        self.retweets = retweets
        self.url = url
        self.details = details
        self.queries = [query]
        


def load_documents_corpus():
    """
    Load documents corpus from dataset_tweets_WHO.txt file
    :return:
    """

    tweets = create_corpus()
    docs = {}
    for tweet_info in tweets:
        title = f"{tweet_info[0][:25]}...\n"
        tweet_id = tweet_info[7]
        new_doc = Document(tweet_id,
                           title,
                           tweet_info[0],
                           tweet_info[1],
                           tweet_info[2],
                           tweet_info[3],
                           tweet_info[4],
                           tweet_info[5],
                           tweet_info[6])
        docs[tweet_id] = new_doc
    return docs


class Query:
    def __init__(self, text, num_terms):
        self.text = text
        self.num_terms = num_terms


def average_q_length(query_list):
    avg_len = []
    for query in query_list:
        avg_len.append(query.num_terms)

    return np.mean(avg_len)
