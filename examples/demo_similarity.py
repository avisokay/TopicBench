"""
demo_similarity.py

This script demonstrates how to use the TopicBench package to compare
topic labels (e.g., Human vs. AI) using semantic similarity.
"""

# 1. Import your package
# note: we use 'topicbench' because we set up the package structure in src/
from topicbench.semantic_similarity import calculate_embedding_cosine 

def run_demo():
    print("--- TopicBench Demo: Comparing Labels ---\n")

    # 2. Define some mock data (Hardcoded for simplicity)
    # In a real scenario, you might load this from 'data/example.csv'
    author_label = "Public Health"
    ai_label_1 = "Health Policy"
    ai_label_2 = "Quantum Physics"

    print(f"Target Label: '{author_label}'")
    
    # 3. Use the package to calculate scores
    score_1 = calculate_embedding_cosine(author_label, ai_label_1)
    score_2 = calculate_embedding_cosine(author_label, ai_label_2)

    # 4. Print the results
    print(f"Comparing with '{ai_label_1}': Score = {score_1:.4f}")
    print(f"Comparing with '{ai_label_2}': Score = {score_2:.4f}")

    if score_1 > score_2:
        print("\nSUCCESS: The AI label related to health scored higher.")
    else:
        print("\nCHECK: Something looks odd with the similarity scores.")

if __name__ == "__main__":
    run_demo()