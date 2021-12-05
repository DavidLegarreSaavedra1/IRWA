import random
from re import search

from app.core.utils import get_random_date, Document
from app.search_engine.algorithms import *

import os
import pickle
import time


# This function is only used when using .pkl files to read index instead of building it from scratch
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
    if not results:
        return False

    for tweet_info in results:
        tweet_id = tweet_info[7]
        title = f"{' '.join(tweet_info[0].split(' ')[:5])}...\n"
        tweet_txt = tweet_info[0]
        tweet_usr = tweet_info[1]
        tweet_date = tweet_info[2]
        tweet_hashtags = tweet_info[3]
        tweet_likes = tweet_info[4]
        tweet_retweets = tweet_info[5]
        tweet_url = tweet_info[6]
        tweet_details = "doc_details?id={}&query={}".format(
            tweet_info[7], search_query)

        new_doc = Document(tweet_id,
                           title,
                           tweet_txt,
                           tweet_usr,
                           tweet_date,
                           tweet_hashtags,
                           tweet_likes,
                           tweet_retweets,
                           tweet_url,                            
                           tweet_details)

        documents.append(new_doc)
    return documents


class SearchEngine:
    """educational search engine"""
    start_time = time.time()
    
    def __init__(self, index, df, id_index, idf, tf):
        self.index = index
        self.df = df
        self.id_index = id_index
        self.idf = idf
        self.tf = tf
        pass

    def search(self, search_query):
        print("Search query:", search_query)

        results = []
        # replace with call to search algorithm
        results = search_index(search_query, self.index,
                               self.idf, self.tf, self.id_index)

        if not results:
            return False

        return results