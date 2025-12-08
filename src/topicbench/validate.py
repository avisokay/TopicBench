import pandas as pd
import numpy as np

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