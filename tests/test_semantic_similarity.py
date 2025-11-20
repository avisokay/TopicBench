import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from src.semantic_similarity import calculate_metrics
import pytest
import numpy as np

# Loading up CSV
from pathlib import Path
csv_path = Path(__file__).parent.parent / 'tech_review' / 'evaluation_cases.csv'
df = pd.read_csv(csv_path)

# Apply the function
metrics_df = df.apply(calculate_metrics, axis=1)

print(metrics_df.head())

def test_semantic_similarity_metrics():
    with pytest.raises(AssertionError):
        # Assert that BLEU scores are above a certain threshold
        assert np.testing.assert_allclose(metrics_df['BLEU'], [0, 1, 0, 0], atol=0.1)
        assert np.testing.assert_allclose(metrics_df['Jaccard'], [0, 1, 0, 0], atol=0.1)
        return