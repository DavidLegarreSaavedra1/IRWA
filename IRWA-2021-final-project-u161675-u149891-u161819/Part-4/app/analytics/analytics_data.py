class AnalyticsData:
    fact_clicks = []
    fact_two = []
    fact_three = []

class doc_q:
    def __init__(self, doc_id, query):
        self.doc_id = doc_id
        self.query = query

class Click:
    def __init__(self, doc_id, description, query):
        self.doc_id = doc_id
        self.description = description
        self.query = query
