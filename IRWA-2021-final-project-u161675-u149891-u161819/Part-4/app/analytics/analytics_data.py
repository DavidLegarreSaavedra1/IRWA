from app.search_engine.algorithms import get_tweet_info
#from app.core.utils import Document

class AnalyticsData:
    fact_clicks = []
    fact_clicks_per_doc = {}
    fact_doc_queries = {}

    def add_click_to_doc(self, doc_id):
        if doc_id not in self.fact_clicks_per_doc:
            self.fact_clicks_per_doc[doc_id] = 1
        else:
            self.fact_clicks_per_doc[doc_id] += 1
        return
    

    def get_top10_clicked_docs(self, corpus):
        rawtop10 = sorted(self.fact_clicks_per_doc)[:10]
        top10 = []

        for tweet_id in rawtop10:
            top10.append(corpus[tweet_id])

        return top10
    
    def add_query_to_doc(self, doc_id, query):
        if doc_id not in self.fact_doc_queries:
            self.fact_doc_queries[doc_id] = [query]
        else:
            self.fact_doc_queries[doc_id].append(query)


class Agent_data:
    def __init__(self, request):
        self.platform = request.user_agent.platform
        self.browser = request.user_agent.browser
        self.language = request.user_agent.language

class Click:
    def __init__(self, doc_id, description):
        self.doc_id = doc_id
        self.description = description
