"""
Alignment scoring utilities for TopicBench.

This module provides functions to compute alignment between human
and AI similarity judgments, including thresholding based on a tau
value (standard deviation based cutoff).
"""
import pandas as pd

from src.topicbench.semantic_similarity import (
    calculate_overlap,
    calculate_embedding_cosine,
    calculate_embedding_euclidean,
    calculate_embedding_manhattan,
    calculate_embedding_angular
)

def score_similarity(label1: str, label2: str, metric = "cosine"):
    '''
    Score the semantic similarity of two labels.

    Input: Two labels (strings).
    Input: Similarity metric (default: Cosine). 
    Options: "cosine", "euclidean", "manhattan", "angular", "bleu", "jaccard"
    Output: Similarity score (float).
    '''

    # Create a temporary DataFrame row structure to use with semantic_similarity functions
    temp_df = pd.DataFrame({'label1': [label1], 'label2': [label2]})
    row = temp_df.iloc[0]

    if metric.lower() == "cosine":
        result = calculate_embedding_cosine(row, 'label1', 'label2')
        return float(result['Cosine_Similarity'])

    elif metric.lower() == "euclidean":
        result = calculate_embedding_euclidean(row, 'label1', 'label2')
        return float(result['Euclidean_Distance'])

    elif metric.lower() == "manhattan":
        result = calculate_embedding_manhattan(row, 'label1', 'label2')
        return float(result['Manhattan_Distance'])

    elif metric.lower() == "angular":
        result = calculate_embedding_angular(row, 'label1', 'label2')
        return float(result['Angular_Distance'])

    elif metric.lower() == "bleu":
        result = calculate_overlap(row, 'label1', 'label2')
        return float(result['BLEU'])

    elif metric.lower() == "jaccard":
        result = calculate_overlap(row, 'label1', 'label2')
        return float(result['Jaccard'])

    else:
        raise ValueError(
            "Unknown metric "
            f"'{metric}'. Choose from: cosine, euclidean, manhattan, "
            "angular, bleu, jaccard"
        )


def compute_alignment(df, human_col="human_similarity", ai_col="ai_similarity", tau=1):
    """
    Compute AI alignment by comparing human and AI similarity scores.

    This function defines a tau-based threshold for AI similarity:
    the threshold is computed as:

        tau_value = mean(AI similarity) - tau * std(human similarity)

    Any AI similarity score >= tau_value is considered "aligned".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing similarity score columns.
    human_col : str, optional
        Column with human similarity ratings (default: "human_similarity").
    ai_col : str, optional
        Column with AI/LLM similarity ratings (default: "ai_similarity").
    tau : float, optional
        Number of standard deviations used to shift the threshold.
        Default = 1.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with two added columns:
        - "tau": numeric threshold value
        - "AI_alignment": binary column (1 = aligned, 0 = not aligned)
    """
    df = df.copy()

    # Compute tau threshold
    tau_value = df[ai_col].mean() - tau * df[human_col].std()

    # Add columns
    df["tau"] = tau_value
    df["AI_alignment"] = (df[ai_col] >= tau_value).astype(int)

    return df
