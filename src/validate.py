import pandas as pd
import numpy as np
from src.semantic_similarity import (
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
    Input: Similarity metric (default: Cosine). Options: "cosine", "euclidean", "manhattan", "angular", "bleu", "jaccard"
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
        raise ValueError(f"Unknown metric '{metric}'. Choose from: cosine, euclidean, manhattan, angular, bleu, jaccard")


def compute_alignment(df, human_col="human_similarity", ai_col="ai_similarity", tau=1):
    '''
    Compute AI alignment based on comparing author vs human and author vs AI semantic similarity scores.

    Input: DataFrame with 'human_similarity' and 'AI_similarity' columns.
    Input: Tau threshold (optional, default=1). Number of standard deviations below mean AI score to set as threshold.
    Output: DataFrame with 'tau' value and binary 'AI_alignment' columns added.
    '''

    # calculate threshold tau
    tau_value = df[ai_col].mean() - tau * df[human_col].std()

    # add tau and alignment columns
    df = df.copy()
    df["tau"] = tau_value
    df["AI_alignment"] = (df[ai_col] >= tau_value).astype(int)

    return df