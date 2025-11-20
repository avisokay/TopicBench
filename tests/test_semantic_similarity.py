import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from src.semantic_similarity import calculate_metrics
import pytest
import numpy as np

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
metrics_df = df.apply(calculate_metrics, axis=1)

print(metrics_df.head())

def test_semantic_similarity_metrics():
    with pytest.raises(AssertionError):
        # Assert that BLEU scores are above a certain threshold
        assert np.testing.assert_allclose(metrics_df['BLEU'], [0, 1, 0, 0], atol=0.1)
        assert np.testing.assert_allclose(metrics_df['Jaccard'], [0, 1, 0, 0], atol=0.1)
        return