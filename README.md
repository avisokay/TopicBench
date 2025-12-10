# TopicBench
Benchmarking LLMs on the labeling of keyword clusters from topic modeling in computational social science.

## Intended audience
Based on our user research, TopicBench is designed for three primary groups:

* **Computational Social Scientists:** Researchers who need to select the most suitable LLM for their datasets without writing complex API integration code.
* **LLM Developers:** Programmers who want to benchmark their custom models against state-of-the-art models to evaluate performance on specific NLP tasks.
* **Data Scientists:** Analysts looking to understand the trade-offs between different topic modeling algorithms (e.g., LDA, BERTopic) and LLM labeling capabilities.

## How to install the package
To install TopicBench, you need Python 3.9 or higher. Since this package is currently in development, we recommend installing it in "editable" mode.

1.  **Clone the repository:**
    ```
    git clone https://github.com/avisokay/TopicBench
    cd TopicBench
    ```

2.  **Install dependencies and the package:**
    This command installs the package defined in `pyproject.toml` and allows you to edit code in `src/` without reinstalling.
    ```
    pip install -e .
    ```

3.  **Verify Installation:**
    Run the test suite to ensure everything is working correctly:
    ```
    pytest
    ```

## Usage example
This example demonstrates how to use `TopicBench` to compare "Human Labels" against "AI Labels" using a dataset.

### 1. Acquire the Data
We provide a sample dataset included directly in this repository.
* **Location:** `data/example.csv`
* **Format:** A CSV file containing columns for the human-assigned label, and the AI-generated label.

*(Note: In a real-world scenario, you would replace this file with your own topic modeling output).*

### 2. Run the Comparison
TopicBench allows you to calculate semantic similarity scores between "Human Labels" and "AI-Generated Labels" using embeddings. Because the core functions are designed to work with data pipelines, they expect a dictionary or data row as input.

```
from topicbench.semantic_similarity import calculate_embedding_cosine

# 1. Organize your data into a dictionary or row
df = pd.DataFrame({
    'human': ['Computer Science', 'Biology', 'Economy'],
    'ai': ['Software Engineering', 'Life Science', 'Stock Market']
})

# 2. Calculate Cosine Similarity
# Arguments: (data_row, key_for_text1, key_for_text2)
result = calculate_embedding_cosine(df, 'human', 'ai')

# 3. Join results back to original data
final_df = pd.concat([df, df_results], axis=1)
print(final_df)

```
### 3. Expected Output

| | human | ai | Cosine_Similarity |
| :--- | :--- | :--- | :---: |
| **0** | Computer Science | Software Engineering | 0.6821 |
| **1** | Biology | Life Science | 0.7412 |
| **2** | Economy | Stock Market | 0.5123 |

## Team Members

  Adam Visokay  
  Department of Sociology, University of Washington  

  Hilary Bayer  
  Center for Quantitative Sciences, University of Washington  
  - Wrote functions to calculate distance measures and statistical tests  
  - Edited documentation  
  - Created YAML file for the Conda environment  
  - Set up GitHub Actions for testing and linting  

  Zilin Cheng  
  School of Pharmacy, University of Washington  

  Aida Mustafanova  
  Department of Applied Mathematics, University of Washington  

  James Stewart  
  Department of Urban Design and Planning, University of Washington

