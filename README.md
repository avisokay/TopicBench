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

## Usage Example
This example demonstrates the complete TopicBench workflow: generating labels with LLMs, computing similarity metrics, and evaluating alignment. See [examples/example.ipynb](examples/example.ipynb) for a detailed notebook walkthrough.

### Step 1: Install and Import
```python
# Install the package (see installation instructions above)
pip install -e .

# Import required modules
import pandas as pd
from topicbench.label_topics import label_topics
from topicbench.validate import score_similarity, compute_alignment
```

### Step 2: Load Your Dataset
```python
# Load a dataset with keyword clusters
df = pd.read_csv('data/data_cleaned.csv')

# Required columns: 'field', 'keywords', 'author_label'
# Optional: 'alt_human' for human baseline comparison
```

### Step 3: Generate Labels with LLMs
```python
# Configure models to benchmark (local or cloud-based)
models_config = {
    'llama3.2:latest': {'api_key_path': None, 'type': 'local'},
    'gpt-3.5-turbo': {'api_key_path': 'path/to/openai_key.txt', 'type': 'api'},
}

# Generate labels for each model
for model_name, config in models_config.items():
    df = label_topics(df, model_name=model_name, API_KEY_PATH=config['api_key_path'])
```

### Step 4: Compute Similarity Scores
```python
# Compare AI labels to human labels using cosine similarity
metric = 'cosine'
all_scores = []

for idx in range(len(df)):
    row = df.iloc[idx]
    author_label = row['author_label']
    ai_label = row['llama3.2:latest']

    score = score_similarity(author_label, ai_label, metric=metric)
    all_scores.append(float(score))

df['llama3.2_cosine_similarity'] = all_scores
```

### Step 5: Evaluate Alignment
```python
# Compute alignment between AI and human scores
# tau parameter controls sensitivity to human variability
alignment_df = compute_alignment(
    df,
    human_col='alt_human_cosine_similarity',
    ai_col='llama3.2_cosine_similarity',
    tau=0
)

# Calculate final benchmark score (mean alignment)
final_score = alignment_df['AI_alignment'].mean()
print(f"Model alignment score: {final_score:.3f}")
```

### Expected Output

#### Input Data Structure
After generating labels, the DataFrame contains:

| Field | Keywords | Author Labels | llama3.2:latest | gpt-3.5-turbo |
|-------|----------|---------------|-----------------|---------------|
| sociology | [['one', 'made', 'anoth'], ['us', 'peopl', 'time'], ...] | ['Inspirational Language', 'Group Identity', 'Resistance'] | ['Social Movement', 'Activism', 'Resistance'] | ['movement and activism', 'collective identity', 'resistance'] |
| medicine | [['body', 'came', 'dries up'], ['pain', 'head', 'back'], ...] | ['Nervous disease', 'Gynecology', 'Mental illness'] | ['Gastrointestinal Issues', 'Menstrual Cycle', 'Mental Health'] | ['Physical Symptoms', 'Pregnancy-related Issues', 'Mental Health'] |
| hci | [['can', 'get', 'us'], ['error', 'problem', 'help'], ...] | ['Problem solving', 'Desperate effort', 'Social media'] | ['error', 'problem', 'network'] | ['technical support', 'app usability', 'social media'] |

#### Benchmark Results
After computing similarities and alignment, compare model performance:

| Field | Human Similarity (Baseline) | llama3.2 Similarity | gpt-3.5 Similarity | llama3.2 Alignment | gpt-3.5 Alignment | llama3.2 Final Score | gpt-3.5 Final Score |
|-------|----------------------------|--------------------|--------------------|-------------------|------------------|---------------------|---------------------|
| sociology | [0.46, 0.86, 0.74, ...] | [0.23, 0.40, 0.34, ...] | [0.31, 0.44, 0.38, ...] | [0, 1, 0, ...] | [0, 0, 1, ...] | 0.60 | 0.33 |
| medicine | [0.25, 0.32, 0.12, ...] | [0.15, 0.41, 0.35, ...] | [0.18, 0.39, 0.42, ...] | [0, 0, 0, ...] | [0, 0, 1, ...] | 0.22 | 0.20 |
| hci | [0.22, 0.49, 0.14, ...] | [0.30, 0.28, 0.13, ...] | [0.35, 0.24, 0.19, ...] | [0, 1, 0, ...] | [1, 0, 0, ...] | 0.40 | 0.30 |

- **Human Similarity**: Alternative human labeler vs. author labels (reference baseline)
- **AI Similarity**: Cosine similarity scores between author and AI labels for each topic
- **Alignment**: Binary scores per topic (1 = AI label meets/exceeds human baseline, 0 = does not)
- **Final Score**: Mean alignment score across all topics in that field (higher = better)

**Available Metrics**: `cosine`, `euclidean`, `manhattan`, `angular`, `bleu`, `jaccard`, `llm_judge`

## Team Members

  Adam Visokay  
  Department of Sociology, University of Washington  
  - Wrote functions to evaluate similarity metrics with different global anchors.
  - Wrote function to use LLM as topic labelers.
  - Created an example notebook that demonstrates a full benchmarking pipeline.
  - Wrote user stories.
  - Wrote and edited documentation to clarify. 

  Hilary Bayer  
  Center for Quantitative Sciences, University of Washington  
  - Wrote functions to calculate distance measures and tests for these
  - Edited documentation  
  - Created YAML file for making a conda environment  
  - Set up GitHub Actions for testing and linting  

  Zilin Cheng  
  School of Pharmacy, University of Washington
  - Wrote functions to claculate the similarity using LLM as judge.
  - Test for the LLM as judge.
  - Edited documentation, wrote README and first part of component specs.
  - Wrote a deno example for similarity.  

  Aida Mustafanova  
  Department of Applied Mathematics, University of Washington  
  - Wrote functions for embedding cosine similarity for tech review
  - Edited user stories and use cases to match the project requirements
  - Improved docstrings across several modules for clarity and consistency
  - Cleaned code according to pylint style guidelines


  James Stewart  
  Department of Urban Design and Planning, University of Washington
  - Wrote functions under token overlap for tech review
  - Ran tests and fixed bugs, such as updating import paths
  - Assisted with final presentation and theme
  - Wrote tests for token overlap

