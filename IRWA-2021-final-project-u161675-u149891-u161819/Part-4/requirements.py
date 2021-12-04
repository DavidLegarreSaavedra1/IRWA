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
from random import randint
import pickle

import nltk
nltk.download('stopwords')
nltk.download('punkt')
