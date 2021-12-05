import datetime
from random import random
import numpy as np

from faker import Faker
from app.search_engine.algorithms import create_corpus
fake = Faker()


# fake.date_between(start_date='today', end_date='+30d')
# fake.date_time_between(start_date='-30d', end_date='now')
#
# # Or if you need a more specific date boundaries, provide the start
# # and end dates explicitly.
# start_date = datetime.date(year=2015, month=1, day=1)
# fake.date_between(start_date=start_date, end_date='+30y')

def get_random_date():
    """Generate a random datetime between `start` and `end`"""
    return fake.date_time_between(start_date='-30d', end_date='now')


def get_random_date_in(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((end - start).total_seconds())), )


class Document:
    #info = [Tweet, Username, Date, Hashtags, Likes, Retweets, Url]
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
