import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.validate import calculate_embedding_cosine
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
    'generated_label': [
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

