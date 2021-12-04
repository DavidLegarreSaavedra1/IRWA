import numpy as np

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

