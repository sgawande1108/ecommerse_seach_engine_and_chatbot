

from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self, tokenized_corpus):
        self.model = BM25Okapi(tokenized_corpus)

    def search(self, query):
        query_tokens = query.split()
        return self.model.get_scores(query_tokens)
