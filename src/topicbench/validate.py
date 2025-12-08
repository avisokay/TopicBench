"""
Alignment scoring utilities for TopicBench.

This module provides functions to compute alignment between human
and AI similarity judgments, including thresholding based on a tau
value (standard deviation based cutoff).
"""

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
