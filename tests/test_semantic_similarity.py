import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from src.semantic_similarity import calculate_overlap, calculate_embedding_cosine,calculate_embedding_manhattan, calculate_embedding_euclidean, calculate_embedding_angular
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

print(pd.concat([df, metrics_df, cosine_df], axis=1))

def test_semantic_similarity_metrics():
    # Assert that scores are as expected
    np.testing.assert_allclose(metrics_df['BLEU'], [0, 1, 0, 0], atol=0.1)
    np.testing.assert_allclose(metrics_df['Jaccard'], [0, 1, 0, 0], atol=0.1)

def test_embeddings_cosine():
    # Assert that scores are as expected
    np.testing.assert_allclose(cosine_df['Cosine_Similarity'], [0.562432, 1, 0.645155, 0.004850], atol=0.1)

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
   
   
   

