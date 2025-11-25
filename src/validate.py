import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def calculate_embedding_cosine(row):
    '''
    Calculate cosine similarity between sentence embeddings.
    Input: row with 'sent1' and 'sent2' fields
    Output: Series with Cosine Similarity score between 0 and 1
    '''

    sent1 = row['sent1']
    sent2 = row['sent2']

    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for both sentences
    embedding1 = model.encode([sent1])
    embedding2 = model.encode([sent2])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]

    return pd.Series({'Cosine_Similarity': cosine_sim})

