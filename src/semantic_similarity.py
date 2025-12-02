import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from math import acos, pi

# Function to calculate the metrics for each row
def calculate_overlap(row, string1, string2):
    '''
    Calculate semantic similarity metrics between two sentences.
    Input: row with 'string1' and 'string2' fields defined
    Output: Series with BLEU and Jaccard scores rating token overlap between 0 and 1
    '''

    sent1 = row[string1]
    sent2 = row[string2]

    # Tokenize both sentences to lowercase word tokens
    tokens1 = word_tokenize(sent1.lower())
    tokens2 = word_tokenize(sent2.lower())

    # BLEU score - Measures how one sentence matches with the other
    bleu = sentence_bleu([tokens2], tokens1)

    # Jaccard Similarity - Compares sets of words in a sentence
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform([sent1, sent2]).toarray()
    jaccard = jaccard_score(X[0], X[1])

    return pd.Series({'BLEU': bleu, 'Jaccard': jaccard})

def calculate_embedding_cosine(row, string1, string2):
    '''
    Calculate cosine similarity between sentence embeddings.
    Input: row with 'string1' and 'string2' fields defined
    Output: Series with Cosine Similarity score between 0 and 1
    '''

    sent1 = row[string1]
    sent2 = row[string2]

    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for both sentences
    embedding1 = model.encode([sent1])
    embedding2 = model.encode([sent2])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]

    return pd.Series({'Cosine_Similarity': cosine_sim})

def calculate_embedding_euclidean(row, string1, string2):
    '''
    Calculate euclidean distance between sentence embeddings.
    Input: row with 'string1' and 'string2' fields defined
    Output: Series with euclidean distance greater than or equal to 0
    '''

    sent1 = row[string1]
    sent2 = row[string2]

    if len(sent1.strip()) == 0 or len(sent2.strip()) == 0:
        raise ValueError("Empty string input detected")
    else:
        pass

    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

       # Generate embeddings for both sentences
    embedding1 = model.encode([sent1], convert_to_tensor=True)
    embedding2 = model.encode([sent2], convert_to_tensor=True)

    # Calculate euclidian distance 
    euclidean_dist = distance.euclidean(embedding1.squeeze().cpu().numpy(), embedding2.squeeze().cpu().numpy())

    return pd.Series({'Euclidean_Distance': euclidean_dist})


def calculate_embedding_manhattan(row, string1, string2):
    '''
    Calculate manhattan distance between sentence embeddings.
    Input: row with 'string1' and 'string2' fields defined
    Output: Series with manhattan distance greater than or equal to 0
    '''

    sent1 = row[string1]
    sent2 = row[string2]
    
    if len(sent1.strip()) == 0 or len(sent2.strip()) == 0:
        raise ValueError("Empty string input detected")
    else:
        pass

    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for both sentences
    embedding1 = model.encode([sent1], convert_to_tensor=True)
    embedding2 = model.encode([sent2], convert_to_tensor=True)

    # Calculate manhattan distance 
    manhattan_dist = distance.cityblock(embedding1.squeeze().cpu().numpy(), embedding2.squeeze().cpu().numpy())

    return pd.Series({'Manhattan_Distance': manhattan_dist})

def calculate_embedding_angular(row, string1, string2):
    '''
    Calculate angular distance between sentence embeddings.
    Input: row with 'string1' and 'string2' fields defined
    Output: Series with angular distance greater than or equal to 0
    '''

    sent1 = row[string1]
    sent2 = row[string2]

    if len(sent1.strip()) == 0 or len(sent2.strip()) == 0:
        raise ValueError("Empty string input detected")
    else:
        pass

    # Load pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for both sentences
    embedding1 = model.encode([sent1], convert_to_tensor=True)
    embedding2 = model.encode([sent2], convert_to_tensor=True)

    # Calculate euclidian distance 
    angular_dist = acos(1 - distance.cosine(embedding1.squeeze().cpu().numpy(), embedding2.squeeze().cpu().numpy()))/pi

    return pd.Series({'Angular_Distance': angular_dist})


