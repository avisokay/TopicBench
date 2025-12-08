"""
demo_similarity.py

This script demonstrates how to use the TopicBench package to compare
topic labels using the 'calculate_embedding_cosine' function.

It mocks a data row (similar to what you would find in a CSV) and 
passes it to the function.
"""

import pandas as pd
from topicbench.semantic_similarity import calculate_embedding_cosine

def run_demo():
    print("--- TopicBench Demo: Cosine Similarity ---\n")

    # 1. Define Mock Data
    # Your function expects a 'row' (dict or Series) and the keys to access the text.
    mock_row_1 = {
        'human_label': "Public Health",
        'ai_label': "Health Policy"
    }
    
    mock_row_2 = {
        'human_label': "Computer Science",
        'ai_label': "Cooking Recipes" # Intentionally different
    }

    print(f"Test Case 1: '{mock_row_1['human_label']}' vs '{mock_row_1['ai_label']}'")

    # 2. Call the function
    # Note: We pass the dictionary (row) and the keys ('human_label', 'ai_label')
    # The function returns a pd.Series with 'Cosine_Similarity'
    result_1 = calculate_embedding_cosine(mock_row_1, 'human_label', 'ai_label')
    score_1 = result_1['Cosine_Similarity']
    
    print(f"Score: {score_1:.4f} (High similarity expected)\n")

    print(f"Test Case 2: '{mock_row_2['human_label']}' vs '{mock_row_2['ai_label']}'")
    
    result_2 = calculate_embedding_cosine(mock_row_2, 'human_label', 'ai_label')
    score_2 = result_2['Cosine_Similarity']
    
    print(f"Score: {score_2:.4f} (Low similarity expected)")

    # 3. Optional: Pandas DataFrame Demo (Batch Processing)
    print("\n--- Batch Processing (DataFrame) ---")
    df = pd.DataFrame([mock_row_1, mock_row_2])
    
    # Apply the function to every row
    results_df = df.apply(calculate_embedding_cosine, args=('human_label', 'ai_label'), axis=1)
    
    # Combine results for display
    final_df = pd.concat([df, results_df], axis=1)
    print(final_df)

if __name__ == "__main__":
    run_demo()