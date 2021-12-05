import random
from re import search

from app.core.utils import get_random_date
from app.search_engine.algorithms import *

import os
import pickle
import time


def read_index():
    data_path = os.path.join(*['app', 'search_engine', 'indexes'])

    with open(os.path.join(data_path, 'index.pkl'), 'rb') as index_file:
        index = pickle.load(index_file)

    with open(os.path.join(data_path, 'df.pkl'), 'rb') as df_file:
        df = pickle.load(df_file)

    with open(os.path.join(data_path, 'id_index.pkl'), 'rb') as id_index_file:
        id_index = pickle.load(id_index_file)

    with open(os.path.join(data_path, 'idf.pkl'), 'rb') as idf_file:
        idf = pickle.load(idf_file)

    with open(os.path.join(data_path, 'tf.pkl'), 'rb') as tf_file:
        tf = pickle.load(tf_file)

    return (index, df, id_index, idf, tf)


def search_index(search_query, index, idf, tf, id_index):
    documents = []
    results = search_tf_idf(search_query, index, idf, tf, id_index)
    for tweet_info in results:
        title = f"{tweet_info[0][:25]}...\n"
        tweet_details = "doc_details?id={}&query={}".format(
            tweet_info[7], search_query)    
        new_doc = DocumentInfo(title,
                               tweet_info[0],
                               tweet_info[1],
                               tweet_info[2],
                               tweet_info[3],
                               tweet_info[4],
                               tweet_info[5],
                               tweet_info[6],
                               tweet_info[7],
                               tweet_details)
        documents.append(new_doc)
    return documents


class SearchEngine:
    """educational search engine"""
    start_time = time.time()
    # Read index to go faster at runtime
    index, df, id_index, idf, tf = read_index()
    # index, df, id_index, idf, tf = create_index() # Build the index from our database
    print("Total time to read the index: {} seconds".format(
        np.round(time.time() - start_time, 2)))

    def search(self, search_query):
        print("Search query:", search_query)

        results = []
        # replace with call to search algorithm
        results = search_index(search_query, self.index,
                               self.idf, self.tf, self.id_index)

        return results


class DocumentInfo:
    #info = [Tweet, Username, Date, Hashtags, Likes, Retweets, Url]
    def __init__(self, title, tweet, username, date, hashtags, likes, retweets, url, id, details):
        self.title = title
        self.tweet = tweet
        self.username = username
        self.date = date
        self.hashtags = hashtags
        self.likes = likes
        self.retweets = retweets
        self.url = url
        self.details = details
        self.id = id
