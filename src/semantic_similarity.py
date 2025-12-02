import os
import json
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sentence_transformers import SentenceTransformer
from openai import OpenAI
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

# Helper: lazily create an OpenAI client
_openai_client = None  # cached client so we only construct once

def _get_openai_client() -> OpenAI:
    """
    Return a cached OpenAI client.

    Users are expected to set their own API key in the OPENAI_API_KEY
    environment variable, e.g.:

        export OPENAI_API_KEY="sk-..."

    If you prefer a different key-management scheme, you can edit this
    helper, but we keep the package code generic.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please export your own API key before using LLM-based similarity."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# LLM-as-a-judge semantic similarity
def calculate_llm_as_judge(row, string1, string2, model: str = "gpt-5"):
    """
    Use a Large Language Model (LLM) as a semantic similarity judge.

    This prompts an LLM with two sentences and asks it for:
    1) a similarity score between 0.0 and 1.0 and
    2) a short natural-language rationale.

    Users must provide *their own* OpenAI API key by setting the
    OPENAI_API_KEY environment variable before calling this function.

    Parameters
    ----------
    row : pd.Series
        Row with `string1` and `string2` fields defined.
    string1, string2 : str
        Column names containing the two sentences.
    model : str, optional
        OpenAI model name to use (default: "gpt-5").

    Returns
    -------
    pd.Series
        Series with:
        - "LLM_Similarity": float in [0, 1]
        - "LLM_Rationale" : short explanation string
    """
    sent1 = row[string1]
    sent2 = row[string2]

    prompt = f"""
You are an expert judge of semantic similarity.

Given two sentences, rate how similar they are in meaning on a scale
from 0.0 (completely unrelated) to 1.0 (paraphrase or exact match).
Focus on meaning rather than surface token overlap.

Sentence 1: {sent1}
Sentence 2: {sent2}

Respond ONLY with a JSON object of the form:
{{
  "similarity": <number between 0 and 1>,
  "rationale": "<one-sentence explanation>"
}}
"""

    client = _get_openai_client()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are an expert semantic similarity judge.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    similarity = float(data.get("similarity", 0.0))
    rationale = data.get("rationale", "")

    return pd.Series(
        {
            "LLM_Similarity": similarity,
            "LLM_Rationale": rationale,
        }
    )

# def calculate_embedding_euclidean():
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


