import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import pytest
import numpy as np

# Loading up CSV
csv_path = Path(__file__).parent.parent / 'tech_review' / 'evaluation_cases.csv'
df = pd.read_csv(csv_path)

def calculate_metrics(row):
    sent1 = row['sent1']
    sent2 = row['sent2']
    tokens1 = word_tokenize(sent1.lower())
    tokens2 = word_tokenize(sent2.lower())
    bleu = sentence_bleu([tokens2], tokens1)
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform([sent1, sent2]).toarray()
    jaccard = jaccard_score(X[0], X[1])
    return pd.Series({'BLEU': bleu, 'Jaccard': jaccard})

metrics_df = df.apply(calculate_metrics, axis=1)


# Smoke Test
def test_smoke_metrics_execute():
    """
    author: James Stewart
    category: smoke test
    justification: Verifies that calculate_metrics runs and produces expected columns without error.
    """
    assert metrics_df is not None # Is the dataframe there?
    assert len(metrics_df) == 4 # Are there four rows?
    assert 'BLEU' in metrics_df.columns and 'Jaccard' in metrics_df.columns # Does it have BLEU and Jaccard?


# One-Shot Test
def test_oneshot_identical_sentences():
    """
    author: James Stewart
    category: one-shot test
    justification: Checks that identical sentences yield perfect similarity scores.
    """
    test_data = pd.DataFrame({
        'sent1': ['he ran quickly to the store'],
        'sent2': ['he ran quickly to the store']
    })
    result = test_data.apply(calculate_metrics, axis=1) # With the two same sentences, do they score the same?
    assert result.iloc[0]['BLEU'] == 1.0
    assert result.iloc[0]['Jaccard'] == 1.0


# Edge Test
def test_edge_no_token_overlap():
    """
    author: James Stewart
    category: edge test
    justification: Validates that completely different sentences yield zero similarity scores.
    """
    test_data = pd.DataFrame({
        'sent1': ['turn left at the traffic light'],
        'sent2': ['photosynthesis occurs in plant cells']
    })
    result = test_data.apply(calculate_metrics, axis=1) # With two completely different sentences, do they score zero?
    np.testing.assert_allclose(result.iloc[0]['BLEU'], 0.0, atol=0.01)
    np.testing.assert_allclose(result.iloc[0]['Jaccard'], 0.0, atol=0.01)


# Pattern Test
def test_pattern_token_overlap_behavior():
    """
    author: James Stewart
    category: pattern test
    justification: Confirms that the expected pattern of token overlap scores is produced for the sample data.
    """
    expected_bleu = [0, 1, 0, 0] # These are the scores we're expecting from the sample
    expected_jaccard = [0, 1, 0, 0]
    np.testing.assert_allclose(metrics_df['BLEU'], expected_bleu, atol=0.1)
    np.testing.assert_allclose(metrics_df['Jaccard'], expected_jaccard, atol=0.1)