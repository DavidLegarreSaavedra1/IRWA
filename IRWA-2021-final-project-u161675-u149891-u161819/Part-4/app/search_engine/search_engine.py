import random

from app.core.utils import get_random_date
from app.search_engine.algorithms import create_index
from .algorithms import *

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


def build_demo_data():
    """
    Helper method, just to demo the app
    :return: a list of demo docs sorted by ranking
    """
    samples = ["Messier 81", "StarBurst", "Black Eye", "Cosmos Redshift", "Sombrero", "Hoags Object",
            "Andromeda", "Pinwheel", "Cartwheel",
            "Mayall's Object", "Milky Way", "IC 1101", "Messier 87", "Ring Nebular", "Centarus A", "Whirlpool",
            "Canis Major Overdensity", "Virgo Stellar Stream"]

    res = []
    for index, item in enumerate(samples):
        res.append(DocumentInfo(item, (item + " ") * 5, get_random_date(),
                                "doc_details?id={}&param1=1&param2=2".format(index), random.random()))
    # simulate sort by ranking
    res.sort(key=lambda doc: doc.ranking, reverse=True)
    return res


class SearchEngine:
    """educational search engine"""
    i = 12345
    start_time = time.time()
    #index, df, id_index, idf, tf = read_index()
    index, df, id_index, idf, tf = create_index()
    print("Total time to read the index: {} seconds".format(np.round(time.time() - start_time, 2)))
    
    def search(self, search_query):
        print("Search query:", search_query)

        results = []
        ##### your code here #####
        results = build_demo_data()  # replace with call to search algorithm
        ##### your code here #####

        return results


class DocumentInfo:
    def __init__(self, title, description, doc_date, url, ranking):
        self.title = title
        self.description = description
        self.doc_date = doc_date
        self.url = url
        self.ranking = ranking