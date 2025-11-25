import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.semantic_similarity import calculate_embedding_cosine
from src.validate import compute_alignment
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
print(results['AI_alignment'])

def test_compute_alignment():
    # check that AI_alignment is computed correctly
    np.testing.assert_allclose(results['AI_alignment'], [0,1,1,0,1,1,0,1,1,1,0,1,1,1,1], atol=0.1)