"""
Semantic similarity utilities for TopicBench.

This module implements functions to compute lexical and semantic similarity
between pairs of sentences, including BLEU and Jaccard overlap, embedding-based
cosine/Euclidean/Manhattan/angular distances, and an optional LLM-as-a-judge
similarity score.
"""

from math import acos, pi
import json
import os

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from scipy.spatial import distance


def calculate_overlap(row, string1, string2):
    """
    Calculate BLEU and Jaccard overlap between two sentences.

    Parameters
    ----------
    row : pd.Series
        Row with the text columns.
    string1, string2 : str
        Column names containing the two sentences.

    Returns
    -------
    pd.Series
        Series with:
        - "BLEU"   : BLEU score in [0, 1]
        - "Jaccard": Jaccard overlap in [0, 1]
    """
    sent1 = row[string1]
    sent2 = row[string2]

    tokens1 = word_tokenize(sent1.lower())
    tokens2 = word_tokenize(sent2.lower())

    bleu = sentence_bleu([tokens2], tokens1)

    vectorizer = CountVectorizer(binary=True)
    vectors = vectorizer.fit_transform([sent1, sent2]).toarray()
    jaccard = jaccard_score(vectors[0], vectors[1])

    return pd.Series({"BLEU": bleu, "Jaccard": jaccard})


def calculate_embedding_cosine(row, string1, string2):
    """
    Calculate cosine similarity between sentence embeddings.

    Parameters
    ----------
    row : pd.Series
        Row with the text columns.
    string1, string2 : str
        Column names containing the two sentences.

    Returns
    -------
    pd.Series
        Series with "Cosine_Similarity" in [-1, 1].
    """
    sent1 = row[string1]
    sent2 = row[string2]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding1 = model.encode([sent1])
    embedding2 = model.encode([sent2])

    cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]

    return pd.Series({"Cosine_Similarity": cosine_sim})


def _get_openai_client() -> OpenAI:
    """
    Construct an OpenAI client using the OPENAI_API_KEY environment variable.

    Returns
    -------
    OpenAI
        Configured OpenAI client instance.

    Raises
    ------
    RuntimeError
        If the OPENAI_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please export your API key before using LLM-based similarity."
        )
    return OpenAI(api_key=api_key)


def calculate_llm_as_judge(row, string1, string2, model: str = "gpt-5"):
    """
    Use a Large Language Model (LLM) as a semantic similarity judge.

    Parameters
    ----------
    row : pd.Series
        Row with the text columns.
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


def calculate_embedding_euclidean(row, string1, string2):
    """
    Calculate Euclidean distance between sentence embeddings.

    Parameters
    ----------
    row : pd.Series
        Row with the text columns.
    string1, string2 : str
        Column names containing the two sentences.

    Returns
    -------
    pd.Series
        Series with "Euclidean_Distance" >= 0.

    Raises
    ------
    ValueError
        If either input string is empty.
    """
    sent1 = row[string1]
    sent2 = row[string2]

    if len(sent1.strip()) == 0 or len(sent2.strip()) == 0:
        raise ValueError("Empty string input detected")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding1 = model.encode([sent1], convert_to_tensor=True)
    embedding2 = model.encode([sent2], convert_to_tensor=True)

    euclidean_dist = distance.euclidean(
        embedding1.squeeze().cpu().numpy(),
        embedding2.squeeze().cpu().numpy(),
    )

    return pd.Series({"Euclidean_Distance": euclidean_dist})


def calculate_embedding_manhattan(row, string1, string2):
    """
    Calculate Manhattan distance between sentence embeddings.

    Parameters
    ----------
    row : pd.Series
        Row with the text columns.
    string1, string2 : str
        Column names containing the two sentences.

    Returns
    -------
    pd.Series
        Series with "Manhattan_Distance" >= 0.

    Raises
    ------
    ValueError
        If either input string is empty.
    """
    sent1 = row[string1]
    sent2 = row[string2]

    if len(sent1.strip()) == 0 or len(sent2.strip()) == 0:
        raise ValueError("Empty string input detected")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding1 = model.encode([sent1], convert_to_tensor=True)
    embedding2 = model.encode([sent2], convert_to_tensor=True)

    manhattan_dist = distance.cityblock(
        embedding1.squeeze().cpu().numpy(),
        embedding2.squeeze().cpu().numpy(),
    )

    return pd.Series({"Manhattan_Distance": manhattan_dist})


def calculate_embedding_angular(row, string1, string2):
    """
    Calculate angular distance between sentence embeddings.

    Parameters
    ----------
    row : pd.Series
        Row with the text columns.
    string1, string2 : str
        Column names containing the two sentences.

    Returns
    -------
    pd.Series
        Series with "Angular_Distance" >= 0.

    Raises
    ------
    ValueError
        If either input string is empty.
    """
    sent1 = row[string1]
    sent2 = row[string2]

    if len(sent1.strip()) == 0 or len(sent2.strip()) == 0:
        raise ValueError("Empty string input detected")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding1 = model.encode([sent1], convert_to_tensor=True)
    embedding2 = model.encode([sent2], convert_to_tensor=True)

    cosine_dist = distance.cosine(
        embedding1.squeeze().cpu().numpy(),
        embedding2.squeeze().cpu().numpy(),
    )
    angular_dist = acos(1 - cosine_dist) / pi

    return pd.Series({"Angular_Distance": angular_dist})
