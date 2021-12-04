import random

from app.core.utils import get_random_date
from app.search_engine.algorithms import *

import os
import pickle
import time

def read_index():
    data_path = os.path.join(*['app', 'search_engine', 'indexes']) 

    with open(os.path.join(data_path,'index.pkl'),'rb') as index_file:
        index = pickle.load(index_file)

    with open(os.path.join(data_path, 'df.pkl'),'rb') as df_file:
        df = pickle.load(df_file)

    with open(os.path.join(data_path,'id_index.pkl'),'rb') as id_index_file:
        id_index = pickle.load(id_index_file)

    with open(os.path.join(data_path,'idf.pkl'),'rb') as idf_file:
        idf = pickle.load(idf_file)

    with open(os.path.join(data_path,'tf.pkl'),'rb') as tf_file:
        tf = pickle.load(tf_file)

    return (index, df, id_index, idf, tf)


def search_index(search_query, index, idf, tf, id_index):

    documents = []
    results = search_tf_idf(search_query, index, idf, tf, id_index)
    print(f"{results=}")
    for tweet_info in results:
        print(f"{tweet_info=}")
        new_doc = DocumentInfo(tweet_info[0],
                               tweet_info[1],
                               tweet_info[2],
                               tweet_info[3],
                               tweet_info[4],
                               tweet_info[5],
                               tweet_info[6])
        documents.append(new_doc)
    return documents


class SearchEngine:
    """educational search engine"""
    i = 12345
    start_time = time.time()
    index, df, id_index, idf, tf = read_index()
    #index, df, id_index, idf, tf = create_index()
    print("Total time to read the index: {} seconds".format(np.round(time.time() - start_time, 2)))
    
    def search(self, search_query):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        results = search_index(search_query, self.index, self.idf, self.tf, self.id_index)  # replace with call to search algorithm
        ##### your code here #####

        return results


class DocumentInfo:
    #info = [Tweet, Username, Date, Hashtags, Likes, Retweets, Url]
    def __init__(self, tweet, username, date, hashtags, likes, retweets, url):
        self.tweet = tweet
        self.username = username
        self.date = date
        self.hashtags = hashtags
        self.likes = likes
        self.retweets = retweets
        self.url = url