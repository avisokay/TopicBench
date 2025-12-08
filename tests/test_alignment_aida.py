"""
Tests for the compute_alignment function in src.validate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from topicbench.semantic_similarity import calculate_embedding_cosine
from topicbench.validate import compute_alignment
import pytest
from topicbench.semantic_similarity import calculate_embedding_cosine

"""df = pd.read_csv("/Users/aidamustafanova/TopicBench/data/example.csv")
df.head()
human_similarity = df.apply(calculate_embedding_cosine, axis=1, string1="author_label", string2="alt_human")

df["human_similarity"] = human_similarity["Cosine_Similarity"]
ai_similarity = df.apply(calculate_embedding_cosine, axis=1, string1="author_label", string2="ai_label")

df["ai_similarity"] = ai_similarity["Cosine_Similarity"]
df[["author_label", "human_similarity", "ai_similarity"]].head(10)
result = compute_alignment(df, human_col="human_similarity", ai_col="ai_similarity", tau=1)

result[["author_label", "human_similarity", "ai_similarity", "tau", "AI_alignment"]]
mean_ai = df["ai_similarity"].mean()
std_human = df["human_similarity"].std()
mean_ai, std_human

tau = 1
tau_value = mean_ai - tau * std_human
tau_value"""

def n_df():
    return pd.DataFrame({"human_similarity": [0.460153, 0.857661, 0.737017, 0.403494, 0.629294,
    0.620044, 0.368112, 0.572172, 0.625472, 0.330916,
    0.544564, 0.439801, 0.826124, 0.430239, 0.535075], "ai_similarity": [0.205445, 0.314150, 0.351043, 0.142072, 0.312016,
    0.882199, 0.201424, 0.695650, 0.747439, 0.556487,
    0.169328, 0.530158, 0.445250, 0.275978, 0.303629],})

#SMOKE TEST
def test_compute_alignment_smoke():
    """
    author: Aida Mustafanova
    category: smoke test

    Function should run without errors, return a dataframe
    of the same length and add the expected columns.
    """
    df = n_df()

    result = compute_alignment(df)  # tau = 1

    #Function should return a dataframe, output length must match input length
    assert len(result) == len(df)
    #Function must add the new expected columns
    assert "tau" in result
    assert "AI_alignment" in result.columns

#ONE SHOT TEST
def test_compute_alignment_one_shot():
    """author: Aida Mustafanova
    category: one-shot test
    """
    df = n_df()

    result = compute_alignment(df, human_col="human_similarity", ai_col="ai_similarity", tau=1)
    #expected tau_value: mean(ai) - tau * std(human)
    expected_tau = df["ai_similarity"].mean() - df["human_similarity"].std(ddof=1)

    #tau should be the same in every row
    assert np.isclose(result["tau"].iloc[0], expected_tau) 
    assert np.allclose(result["tau"], expected_tau)

    #expected AI_alignment: 1 if ai_similarity >= tau_value else 0
    expected_alignment = (df["ai_similarity"] >= expected_tau).astype(int).tolist()

    assert result["AI_alignment"].tolist() == expected_alignment, (
        "AI_alignment does not match the expected pattern for this simple dataset."
    )

#EDGE TEST
def edge_df():
    """
    author: Aida Mustafanova
    category: helper (edge)

    Edge case: all human and AI similarities are exactly the same.
    """
    return pd.DataFrame({
        "human_similarity": [0.46, 0.46, 0.46],
        "ai_similarity":    [0.46, 0.46, 0.46],})

def test_compute_alignment_edge():
    df = edge_df()
    expected_tau = df["ai_similarity"].mean()
    result = compute_alignment(df)

    assert np.allclose(result["tau"], expected_tau)
    assert result["AI_alignment"].tolist() == [1,1,1] 

#PATTERN TEST
def test_compute_alignment_pattern():
    df = n_df()
    """
    author: Aida Mustafanova
    category: edge test

    When all values are identical, std(human_similarity) = 0,
    so tau should equal mean(ai_similarity), and all AI_alignment
    values should be 1.
    """
    #The idea of this test is when tau increase, tau_value becomes smaller, AI allignment shouldn't decrease. 
    result_0 = compute_alignment(df, human_col="human_similarity", ai_col="ai_similarity", tau=0)
    #let's choose tau=2
    result_2 = compute_alignment(df, human_col="human_similarity", ai_col="ai_similarity", tau=2)
    aligned_tau0 = result_0["AI_alignment"].to_numpy()
    aligned_tau2 = result_2["AI_alignment"].to_numpy()
    assert np.all(aligned_tau2 >= aligned_tau0)

    