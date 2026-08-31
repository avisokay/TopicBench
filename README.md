# TopicBench
Benchmarking LLMs on the labeling of keyword clusters from topic modeling in computational social science.

## Workflow
Given: keyword clusters and topic labels (separate labels for each the original authors and alternative human annotators) for several peer reviewed papers in Computational Social Science. 
1. use LLM(s) to provide model-specific topic labels
2. calculate semantic similarity scores relative to original author labels
3. evaluate against human-human comparison
4. score each model on each paper

## Dataset

The benchmark corpus lives in `data/`. It was built by stratified harvest from OpenAlex
across 19 social-science discipline strata, screened for papers that actually *run* a
classical topic model on a substantive research question, then extracted from full text.
See [`data/topicbench_process.html`](data/topicbench_process.html) for the full pipeline,
yields, and validation.

| file | rows | unit | use |
|---|---|---|---|
| `data/topicbench_topics.csv` | 2,290 | one **topic** | **benchmarking** — drop-in for the usage example below |
| `data/topicbench_bibliography.csv` | 178 | one **paper** | provenance, citation, per-paper metadata |
| `data/topicbench_process.html` | — | — | how the corpus was built, with a codebook for every variable |
| `data/PREPROCESSING.md` | — | — | every cleaning rule with its measured prevalence |
| `data/data_dashboard.html` | — | — | interactive explorer — open in a browser |
| `data/corpus_coverage.png` | — | — | coverage and label-rate figure |
| `data/data_cleaned.csv` | 24 | one paper | original pilot set |

### Source records

Kept because the report's numbers are reproducible only from them, not because they are
inputs to normal use:

| file | what it is |
|---|---|
| `data/EXTRACTION_PROMPT.md` | the prompt governing the vision extraction pass |
| `data/topicbench_vision_extractions.jsonl` | raw vision output, including the papers that yielded **no** table — the only record of why each failed, and the basis for the reporting-practice analysis |
| `data/vision_extraction_manifest_ext.csv` | which papers were attempted; fixes the denominators |
| `data/topicbench_pairs.csv` | raw per-topic output of the local extraction pass, incl. per-row provenance flags |
| `data/fill_provenance.json`, `data/pass2_corrections.json` | provenance and corrections for the `corpus` / `method` columns |


### Exploring the corpus

`data/data_dashboard.html` is a self-contained explorer: filter by discipline, tier,
extraction route, year and quality flag, browse every keyword→label pair, and inspect
distributions and per-paper provenance. Open it directly — no server, no network, no
build step.

It embeds a frozen snapshot of the CSVs rather than reading them at view time, so
regenerate it after any change to the corpus:

```bash
python scripts/build_dashboard.py
```

Standard library only — it does not need the packages in `pyproject.toml`. Run
`--help` for explicit `--topics` / `--bibliography` / `--out` paths.

`topicbench_topics.csv` is the analysis unit: each row is one keyword cluster paired with
the label its own authors gave it. Columns `field`, `keywords`, `author_label` match the
signature the pipeline below expects.

### Filtering before you benchmark

Every row carries an `eval_tier` and a `flags` string; nothing is dropped, so you set the
threshold:

| tier | n | meaning |
|---|---|---|
| `strict` | 1,807 | clean, English, no contamination — **headline analysis** |
| `core` | 388 | adds trivially-easy labels and abstract-leakage cases |
| `sensitivity` | 55 | non-English or sentence-like labels |
| `excluded` | 40 | degenerate — no usable label |

```python
df = pd.read_csv('data/topicbench_topics.csv')
df = df[df.eval_tier == 'strict']            # headline; use 'core' for robustness
df['keywords'] = df.keywords.apply(json.loads)
```

Three cautions that affect inference:

1. **Cluster standard errors by `work_id`.** Topics within a paper share a corpus and one
   labeler's habits, so 1,825 topics are not 1,825 independent observations — they come
   from 130 papers.
2. **Exclude `extraction_source == "vision_local"` for any prevalence claim.** Those 17
   papers were hand-picked *because* they publish topic tables (17/17 carry labels), which
   is selection on the outcome. They are valid benchmark items but must not enter an
   estimate of how often social scientists label topics.
3. **Fix the number of keywords shown.** `n_keywords` ranges 2–47 (median 10). Left
   unfixed, prompt difficulty varies by paper.

### Known corpus properties

- Author labels are short: median 2 words, 76% are 1–3 words. Exact match is too strict a
  metric; embedding similarity plus a human-adjudicated subsample is the defensible pairing.
- **Prompt asymmetry.** Authors saw top terms *plus* representative documents, their
  research question, and domain expertise. A keyword-only prompt is a harder task than the
  one the humans performed — state it as the design or supply matched context.
- Of 396 papers with full text, only 160 (40%) published an author-labeled topic table.
  In the stratified sample, CS-adjacent fields label at 18% versus 39% elsewhere
  (Fisher OR 0.34, p = 0.015), so the ground truth skews toward interpretive traditions.
  Report per-discipline performance rather than a single pooled score.

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
import json

# Load the benchmark corpus (one row per topic)
df = pd.read_csv('data/topicbench_topics.csv')
df = df[df.eval_tier == 'strict']              # see "Filtering" above
df['keywords'] = df.keywords.apply(json.loads)

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

Adam Visokay (contact)
avisokay@uw.edu
Department of Sociology, University of Washington

an earlier version of this package was built as a collaborative class project with contributions from Hilary Bayer, Zilin Cheng, Aida Mustafanova, and James Stewart.
