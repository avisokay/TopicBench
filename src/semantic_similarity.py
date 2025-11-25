import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sentence_transformers import SentenceTransformer

# Function to calculate the metrics for each row
def calculate_overlap(row):
    '''
    Calculate semantic similarity metrics between two sentences.
    Input: row with 'sent1' and 'sent2' fields
    Output: Series with BLEU and Jaccard scores rating token overlap between 0 and 1
    '''

    sent1 = row['sent1']
    sent2 = row['sent2']

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

