from sentence_transformers import SentenceTransformer

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class CosineSearch:
    def __init__(self, embedding_matrix):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_matrix = embedding_matrix

    def search(self, query):
        query_embedding = self.model.encode([query]).astype('float32')
        return cosine_similarity(query_embedding, self.embedding_matrix)[0]

