
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

class FaissSearch:
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        self.index.add(embedding_matrix)

    def search(self, query_embedding, top_k=5):
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0], distances[0]
