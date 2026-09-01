"""
Semantic similarity utilities for TopicBench.

This module implements functions to compute lexical and semantic similarity
between pairs of sentences: Jaccard token overlap, embedding-based
cosine/Euclidean/Manhattan/angular distances, and an optional LLM-as-a-judge
similarity score.

BLEU was removed after measuring it on the pilot topic labels. Its default
weights score 4-grams, but topic labels are a median of two words, so the
4-gram precision is always zero and the score collapses to the smoothing floor
(max 1.4e-231 over 27 pairs). It also rank-correlated with Jaccard at Spearman
1.00 on that data, so it carried no information Jaccard did not already give,
while costing the whole NLTK dependency and its downloaded corpora.
"""

from math import acos, pi
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from scipy.spatial import distance


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_MODEL_CACHE = {}


def get_model(name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Return a cached SentenceTransformer, loading it once per process.

    The embedding functions are written to be used with ``DataFrame.apply``,
    which calls them once per row. Constructing the model inside the function
    would reload it from disk on every row -- roughly two seconds each, so
    around 85 minutes over a corpus of 2,300 topics. Caching makes the load a
    one-off.

    Parameters
    ----------
    name : str, optional
        SentenceTransformer model name.

    Returns
    -------
    SentenceTransformer
        The cached model instance.
    """
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def calculate_overlap(row, string1, string2):
    """
    Calculate Jaccard token overlap between two sentences.

    Overlap is computed on the binary bag of words produced by
    ``CountVectorizer``, so it measures shared vocabulary and ignores order and
    repetition. On short strings such as topic labels it fires only when the two
    labels literally reuse a word; treat it as a floor and read it alongside an
    embedding or LLM-judged score.

    Parameters
    ----------
    row : pd.Series
        Row with the text columns.
    string1, string2 : str
        Column names containing the two sentences.

    Returns
    -------
    pd.Series
        Series with "Jaccard": token overlap in [0, 1].
    """
    sent1 = row[string1]
    sent2 = row[string2]

    vectorizer = CountVectorizer(binary=True)
    vectors = vectorizer.fit_transform([sent1, sent2]).toarray()
    jaccard = jaccard_score(vectors[0], vectors[1])

    return pd.Series({"Jaccard": jaccard})


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

    model = get_model()

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
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert semantic similarity judge.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        # some models wrap JSON in a markdown fence despite response_format
        content = content.strip("`")
        content = content.split("\n", 1)[1] if "\n" in content else content
        content = content.rsplit("```", 1)[0]
    data = json.loads(content)
    similarity = float(data.get("similarity", 0.0))
    similarity = min(1.0, max(0.0, similarity))
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

    model = get_model()

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

    model = get_model()

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

    model = get_model()

    embedding1 = model.encode([sent1], convert_to_tensor=True)
    embedding2 = model.encode([sent2], convert_to_tensor=True)

    cosine_dist = distance.cosine(
        embedding1.squeeze().cpu().numpy(),
        embedding2.squeeze().cpu().numpy(),
    )
    angular_dist = acos(1 - cosine_dist) / pi

    return pd.Series({"Angular_Distance": angular_dist})
