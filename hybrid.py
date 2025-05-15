import numpy as np

def hybrid_search(query, bm25, cosine, df, top_k=5, bm25_weight=0.5, cosine_weight=0.5):
    bm25_scores = np.array(bm25.search(query))
    cosine_scores = np.array(cosine.search(query))

    bm25_scores /= np.linalg.norm(bm25_scores)
    cosine_scores /= np.linalg.norm(cosine_scores)

    combined = bm25_weight * bm25_scores + cosine_weight * cosine_scores
    top_k_indices = combined.argsort()[-top_k:][::-1]

    columns = ['title', 'description', 'filename', 'main_category', 'price',
               'average_rating', 'categories', 'store', 'features', 'details',
               'embedding_text', 'clean_embedding_text']

    return df.iloc[top_k_indices][columns].assign(score=combined[top_k_indices], index=top_k_indices)
