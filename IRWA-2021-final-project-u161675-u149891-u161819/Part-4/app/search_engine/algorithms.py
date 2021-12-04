import nltk
nltk.download('stopwords')
nltk.download('punkt')

from collections import defaultdict
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import math
import numpy as np
import collections
from numpy import linalg as la
import time
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import smart_open
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
import os


# Process tweets

def process_tweet(line):
    """
    Pre-process the tweet text removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.
    
    Argument:
    line -- string (text) to be pre-processed
    
    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    line = line.lower() # Transform in lowercase
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    line = tokenizer.tokenize(line)
    line = [word for word in line if word not in stop_words]  #eliminate the stopwords 
    line = [stemmer.stem(word) for word in line] #perform stemming 
    i = 0
    for word in line:
        if word[0:4] == 'http':
            line = line[:i]
        i+=1
    ## END CODE
    return line


def get_tweet_info(tweet):
    Tweet = tweet['full_text']
    Username = tweet['user']['name']
    Date = tweet['created_at']
    Hashtags = []
    hashtags_list = tweet['entities']['hashtags']
    for hashtag in hashtags_list:
        Hashtags.append(hashtag['text'])
    Likes = tweet['favorite_count']
    Retweets = tweet['retweet_count']
    Url = f"https://twitter.com/{tweet['user']['screen_name']}/status/{tweet['id_str']}"
    info = [Tweet, Username, Date, Hashtags, Likes, Retweets, Url]
    return info


# Build the index
def create_index():
    """
    Generates the index from our database to perform queries from
    
    Argument:
    tweets -- collection of tweets
    
    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of documents where these keys appears in as values.
    """
    data_path = os.path.join(*['app', 'search_engine', 'indexes']) 

    docs_path = os.path.join(data_path,'dataset_tweets_WHO.txt')

    with open(docs_path) as fp:
        lines = fp.readline()
    tweets = json.loads(lines)


    proc_tweets = {}
    for tweet_id, tweet in zip(tweets.keys(),tweets.values()):
        proc_tweets[int(tweet_id)] = process_tweet(tweet['full_text'])

    index = {}
    id_index = {}
    tf = {}
    df = defaultdict(int)
    idf = defaultdict(float)
    numDocuments = len(tweets)
    for i in range(numDocuments):
        tweet = tweets[str(i)]
        terms = process_tweet(tweet['full_text']) #get tweet text
        info = get_tweet_info(tweet) # get "document" info
        id_tweet = tweet['id']
        id_index[id_tweet]=info
        
        for term in terms: 
            try:
                index[term].append(id_tweet)  
                
            except:
                index[term]= [id_tweet]
            
        norm=0
        for term,ids in index.items():
            norm += len(ids)**2
        norm = math.sqrt(norm)
        
        for term,ids in index.items():
            if term in tf:
                tf[term][id_tweet] = np.round(len(ids)/norm,4)
            else:
                tf[term] = {id_tweet:np.round(len(ids)/norm,4)}
            df[term] += 1
        
        for term in index:
            idf[term] = np.round(np.log(float(numDocuments/df[term])),4)
        
    return index, tf,df,idf,id_index


# Ranking the documents
def rank_documents(terms, docs, index, idf, tf, title_index):
    """
    Perform the ranking of the results of a search based on the tf-idf weights
    
    Argument:
    terms -- list of query terms
    docs -- list of documents, to rank, matching the query
    index -- inverted index data structure
    idf -- inverted document frequencies
    tf -- term frequencies
    title_index -- mapping between page id and page title
    
    Returns:
    Print the list of ranked documents
    """


    doc_vectors = defaultdict(lambda: [0] * len(terms)) 
    query_vector = [0] * len(terms)

    # compute the norm for the query tf
    query_terms_count = collections.Counter(terms) 

    query_norm = la.norm(list(query_terms_count.values()))

    for termIndex, term in enumerate(terms):  #termIndex is the index of the term in the query
        if term not in index:
            continue

        # Compute tf*idf(normalize TF as done with documents)
        query_vector[termIndex]=query_terms_count[term]/query_norm * idf[term] 

        # Generate doc_vectors for matching docs
        for doc in index[term]:          
            if doc in docs:
                doc_vectors[doc][termIndex] = tf[term][doc] * idf[term]

    doc_scores=[[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items() ]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]
    if len(result_docs) == 0:
        print("No results found, try again")
        query = input()
        if not query:
            return None
        result_docs, doc_scores = search_tf_idf(query, index)
    return result_docs, doc_scores


# Search engine
def search_tf_idf(query, index, idf, tf, id_index):
    """
    output is the list of documents that contain any of the query terms. 
    So, we will get the list of documents for each query term, and take the union of them.
    """
    query = process_tweet(query)
    docs = set()
    first = True
    if not query:
        return None
    for term in query:
        try:
            list_docs = index[term]
            if first:
                docs = set(list_docs)
                first = False
            else:
                docs &= set(list_docs)
        except:
            break
            
    docs = list(docs)
    ranked_docs, doc_scores = rank_documents(query, docs, index, idf, tf, id_index)
    
    top = 10
    tweets = []

    if ranked_docs:
        for d_id in ranked_docs[:top]:
            #print(f"{id_index[d_id]}\n")
            tweets.append(id_index[d_id])
    else:
        return
    return tweets