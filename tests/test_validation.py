import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.semantic_similarity import calculate_embedding_cosine
from src.validate import compute_alignment
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import pytest

# manually create data for testing
df = pd.DataFrame({
    'author_label': [
        'Inspirational Language',
        'Group Identity Debates',
        'Neocolonialism',
        'Eco−Literature',
        'News and Culture',
        'Species Conservation',
        'General Concern',
        'Land Conservation',
        'Violent Protest',
        'Direct Action/Ecotage',
        'Occupation/Camps',
        'Sustainable Societies',
        'International Terror',
        'Admonishments',
        'Anti−Capatalist Left'
    ],
    'alt_human': [
        'Motivational Rhetoric',
        'Collective Identity Discussions',
        'Neo-imperialism',
        'Environmental Writing',
        'Media and Society',
        'Wildlife Protection',
        'Broad Anxiety',
        'Habitat Preservation',
        'Militant Demonstration',
        'Environmental Sabotage',
        'Protest Encampments',
        'Green Communities',
        'Global Terrorism',
        'Reprimands',
        'Socialist Left'
    ],
    'ai_label': [
        'Solidarity and Collective Action',
        'Political Ideology and Activism',
        'Corporate Exploitation and Indigenous Resistance',
        'Publications and Resources',
        'Law and Society',
        'Biodiversity Conservation',
        'Environmental Nostalgia',
        'Environmental and Indigenous Land Protection',
        'Protest Policing and Clashes',
        'Direct Action Campaigns',
        'Anti-Road Protests',
        'Social Ecology',
        'State Repression of Radical Activists',
        'Personal Reflection',
        'Class Struggle Against Global Capitalism'
    ]
})

# calculate cosine similarity
human_similarity = df.apply(calculate_embedding_cosine, axis=1, string1='author_label', string2='alt_human')
ai_similarity = df.apply(calculate_embedding_cosine, axis=1, string1='author_label', string2='ai_label')

df['human_similarity'] = human_similarity['Cosine_Similarity']
df['ai_similarity'] = ai_similarity['Cosine_Similarity']

# compute alignment
results = compute_alignment(df, human_col='human_similarity', ai_col='ai_similarity', tau=1)

print(results)

@pytest.mark.parametrize("tau_param", [0, 1, 2]) # test for different tau values

def test_tau(tau_param):
    """
    author: avisokay
    reviewer: hhbayer
    category: (iv) Pattern Test
    """

    # Compute alignment with the specified tau value
    test_results = compute_alignment(df, human_col='human_similarity', ai_col='ai_similarity', tau=tau_param)

    # Tau computation correctness
    expected_tau = df['ai_similarity'].mean() - tau_param * df['human_similarity'].std()
    np.testing.assert_allclose(test_results['tau'], expected_tau, rtol=1e-10)
def test_compute_alignment():
    # check that AI_alignment is computed correctly
    np.testing.assert_allclose(results['AI_alignment'], [0,1,1,0,1,1,0,1,1,1,0,1,1,1,1], atol=0.1)



def test_compute_alignment_one_shot():
    """
    author: Zilin Cheng
    category: one-shot test
    """
    # Small, hand-crafted example where we know the alignment labels
    df = pd.DataFrame(
        {
            "human_similarity": [0.2, 0.8, 0.9],
            "ai_similarity": [0.1, 0.7, 0.95],
        }
    )

    result = compute_alignment(
        df,
        human_col="human_similarity",
        ai_col="ai_similarity",
        tau=1,
    )

    # Tau should be the same for every row and follow the spec:
    # tau = mean(ai) - 1 * std(human)
    expected_tau = df["ai_similarity"].mean() - df["human_similarity"].std()

    # All tau values are identical – we just check the first
    assert pytest.approx(result["tau"].iloc[0], rel=1e-6) == expected_tau
    assert result["tau"].nunique() == 1

    # For this example, we expect AI_alignment to be [0, 1, 1]
    assert list(result["AI_alignment"]) == [0, 1, 1]
