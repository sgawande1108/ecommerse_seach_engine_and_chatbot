
from cosine_search import CosineSearch
from bm25search import BM25Search
from faiss_search import FaissSearch

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
import json
import re
from query_expansion import expand_query_with_llm
from hybrid import hybrid_search
import pandas as pd
import pandas as pd
import numpy as np
import ast




file_path = "product_data_with_index2.csv"
df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')

df['embedding'] = df['embedding'].apply(ast.literal_eval)
embedding_matrix = np.vstack(df['embedding'].values).astype('float32')                 #ban gayi bhai embedding

corpus = df['clean_embedding_text'].apply(lambda x: x.split()).tolist()                 #bm25 ke liye corpus                               

bm25 = BM25Search(corpus)             


cosine = CosineSearch(embedding_matrix)                                          #created module import
faiss_model = FaissSearch(embedding_matrix)   

                             



def retrieve_data(user_query):                               
       
        contextual_query = expand_query_with_llm(user_query)                         

        results = hybrid_search(user_query+":-"+contextual_query, bm25, cosine, df)                      #create module import


        return results

