import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from src.validate import compute_alignment
from src.semantic_similarity import (calculate_overlap, calculate_embedding_cosine,calculate_embedding_manhattan, calculate_embedding_euclidean, calculate_embedding_angular, alculate_llm_as_judge)
import pytest
import numpy as np
import os

# hardcoded test data
df = pd.DataFrame({
    'sent1': [
        'the cat sat on the mat',
        'he ran quickly to the store',
        'domestic unrest',
        'turn left at the traffic light'
    ],
    'sent2': [
        'a feline rested atop a rug',
        'he ran quickly to the store',
        'political instability in the country',
        'photosynthesis occurs in plant cells'
    ]
})

# Apply the function
metrics_df = df.apply(calculate_overlap, axis=1, string1='sent1', string2='sent2')
cosine_df = df.apply(calculate_embedding_cosine, axis=1, string1='sent1', string2='sent2')

def test_BLUE():
    """
    author: avisokay
    reviewer: hhbayer
    category: (i) Smoke Test
    """
    # Assert that BLEU column is the correct type
    assert isinstance(metrics_df['BLEU'], pd.Series)

def test_semantic_similarity():
    """
    author: avisokay
    reviewer: hhbayer
    category: (ii) One-Shot Test
    """
    # Assert that scores are as the expected values
    np.testing.assert_allclose(metrics_df['Jaccard'], [0, 1, 0, 0], atol=0.1)

def test_embeddings_cosine():
    """
    author: avisokay
    reviewer: hhbayer
    category: (iii) Edge Test
    """

    # Check boundary conditions: cosine similarity must be between -1 and 1
    assert(cosine_df['Cosine_Similarity'] >= -1).all()
    np.testing.assert_allclose(cosine_df['Cosine_Similarity'],
                               np.clip(cosine_df['Cosine_Similarity'], -1, 1),
                               atol=1e-6)

    # Check edge case: identical strings should have cosine similarity of 1.0
    identical_idx = 1  # 'he ran quickly to the store' appears in both sent1 and sent2
    np.testing.assert_allclose(cosine_df.iloc[identical_idx]['Cosine_Similarity'], 1.0, atol=0.01)

    # Assert that scores are as expected
    np.testing.assert_allclose(cosine_df['Cosine_Similarity'], [0.562432, 1, 0.645155, 0.004850], atol=0.1)


def test_overlap():
    """
    author: Zilin Cheng
    category: smoke test
    """
    df = pd.DataFrame(
        {
            "s1": ["The cat sits on the mat."],
            "s2": ["The cat sits on the mat."],
        }
    )

    result = df.iloc[0].pipe(
        calculate_overlap,
        string1="s1",
        string2="s2",
    )

    # Just check that it runs and returns reasonable values
    assert set(result.index) == {"BLEU", "Jaccard"}
    assert 0.0 <= result["BLEU"] <= 1.0
    assert 0.0 <= result["Jaccard"] <= 1.0


def test_embedding_edge_empty_string():
    """
    author: Zilin Cheng
    category: edge test
    """
    # Edge case: one or both sentences are empty strings
    df = pd.DataFrame(
        {
            "s1": [""],
            "s2": ["Non-empty comparison sentence."],
        }
    )

    result = df.iloc[0].pipe(
        calculate_embedding_cosine,
        string1="s1",
        string2="s2",
    )

    # The important property is that the function does not crash and
    # returns a cosine similarity score in the valid range.
    assert "Cosine_Similarity" in result
    assert 0.0 <= result["Cosine_Similarity"] <= 1.0


def test_llm_as_judge_pattern_parses_response(monkeypatch):
    """
    author: Zilin Cheng
    category: pattern test
    justification: checks the pattern that the function correctly
                   interprets a well-formed LLM JSON response into a
                   similarity score in [0, 1] plus a rationale string.
    """

    # Fake OpenAI-like objects to avoid real network calls
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeResponsesAPI:
        def create(self, *args, **kwargs):
            # Return a JSON object as the model "output"
            return FakeResponse(
                '{"similarity": 0.9, "rationale": "Sentences are nearly paraphrases."}'
            )

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponsesAPI()

    # Monkeypatch the helper in src.semantic_similarity so that
    # calculate_llm_as_judge uses our fake client instead of OpenAI.
    monkeypatch.setattr(
        "src.semantic_similarity._get_openai_client",
        lambda: FakeClient(),
        raising=False,
    )

    # Dummy data frame row
    df = pd.DataFrame(
        {
            "s1": ["The cat sits on the mat."],
            "s2": ["A cat is sitting on a mat."],
        }
    )

    result = df.iloc[0].pipe(
        calculate_llm_as_judge,
        string1="s1",
        string2="s2",
    )

    # Pattern we care about:
    #  - similarity is a float between 0 and 1
    #  - rationale is a non-empty string coming from the JSON
    assert "LLM_Similarity" in result
    assert "LLM_Rationale" in result

    assert isinstance(result["LLM_Similarity"], float)
    assert 0.0 <= result["LLM_Similarity"] <= 1.0

    assert isinstance(result["LLM_Rationale"], str)
    assert "paraphrases" in result["LLM_Rationale"]


def test_embedding_manhattan():
   """
    author: hhbayer
    reviewer: zilin49
    category: (i) Smoke Test
    """
   manhattan_df = calculate_embedding_manhattan(df.iloc[0], 'sent1', 'sent2')
   assert isinstance(manhattan_df, pd.Series)

def test_embedding_euclidean():
   """
    author: hhbayer
    reviewer: zilin49
    category: (ii) One-Shot Test
    """
   euclidean_df = df.apply(calculate_embedding_euclidean, axis=1, string1='sent1', string2='sent2')
   np.testing.assert_allclose(euclidean_df['Euclidean_Distance'], [0.935487, 0.0, 0.842431, 1.41078], atol=0.1)

def test_embeddings_angular():
   """
    author: hhbayer
    reviewer: zilin49
    category: (iii) Edge Test
    """
   bad_input_df = pd.DataFrame({
    'sent1': [
        ''
    ],
    'sent2': [
        ''
    ]
    })

   with pytest.raises(ValueError, match="Empty string input"):
      calculate_embedding_angular(bad_input_df.iloc[0], 'sent1', 'sent2')

def test_distances_pattern():
   """
    author: hhbayer
    reviewer: zilin49
    category: (iv) Pattern Test
    """
   # Calculate three distance measures on sample sentences in df
   euclidean_df = df.apply(calculate_embedding_euclidean, axis=1, string1='sent1', string2='sent2')
   manhattan_df = df.apply(calculate_embedding_manhattan, axis=1, string1='sent1', string2='sent2')
   angular_df = df.apply(calculate_embedding_angular, axis=1, string1='sent1', string2='sent2')

   # Pattern: ranked distances should be consistent across distance measures
   euclidean_ranks = euclidean_df['Euclidean_Distance'].rank().tolist()
   manhattan_ranks = manhattan_df['Manhattan_Distance'].rank().tolist()
   angular_ranks = angular_df['Angular_Distance'].rank().tolist()
   assert euclidean_ranks == manhattan_ranks == angular_ranks
   
   
   

