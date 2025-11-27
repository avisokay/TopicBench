import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from src.semantic_similarity import calculate_overlap, calculate_embedding_cosine
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