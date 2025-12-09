# Use Cases for TopicBench

## UC1 – Evaluate a new LLM on topic labeling
**Related User Story:** LLM Developer  

**Actor:** LLM Developer  
**Goal:** Evaluate how a new LLM performs on labeling keyword clusters.  

**Preconditions:**
- TopicBench is installed and configured.
- Keyword clusters and reference labels (gold/human) are available.
- The new LLM is accessible (API or local).

**Expected Input:**
- Keyword clusters.
- LLM configuration (model name, endpoint, API key, parameters).

**Expected Output:**
- Metrics (e.g., accuracy, F1, similarity scores) comparing the new LLM to reference labels.

**Main Flow:**
1. Developer loads a dataset of keyword clusters into TopicBench.
2. Developer selects the new LLM to evaluate.
3. TopicBench sends clusters to the LLM and collects predicted labels.
4. TopicBench compares predicted labels to reference labels.
5. TopicBench computes and displays evaluation metrics.

**Exceptions:**
- LLM API call fails or times out.
- Dataset is missing or in the wrong format.

---

## UC2 – Benchmark off-the-shelf LLMs without API coding
**Related User Story:** Computational Social Scientist  

**Actor:** Computational Social Scientist  
**Goal:** Compare several pre-configured LLMs on a dataset without writing API integration code.  

**Preconditions:**
- TopicBench has pre-configured integrations for major LLM providers.
- User has a dataset of keyword clusters.

**Expected Input:**
- Choice of dataset.
- Selection of built-in models to compare.

**Expected Output:**
- Benchmark table showing performance of each model on the dataset.

**Main Flow:**
1. User opens TopicBench and selects a dataset.
2. User chooses one or more pre-configured models (e.g., GPT-4, Claude).
3. TopicBench runs labeling for each model.
4. TopicBench computes performance metrics per model.
5. TopicBench shows a comparison table or simple report.

**Exceptions:**
- Missing or invalid API keys.
- One of the models is unavailable.

---

## UC3 – Run a full benchmark and save results for research
**Related User Story:** Original Researchers  

**Actor:** Original Researcher  
**Goal:** Automate benchmarking of multiple LLMs and store results for publication and future reuse.  

**Preconditions:**
- TopicBench is configured with multiple LLMs.
- Datasets and label sets are available.
- Environment is reproducible (requirements, Docker, etc.).

**Expected Input:**
- List of models to benchmark.
- Dataset(s) and benchmark configuration (metrics, prompt template, seeds).

**Expected Output:**
- Saved benchmark results (metrics, logs).
- Files for figures and tables used in papers.

**Main Flow:**
1. Researcher selects dataset(s) and the set of LLMs.
2. Researcher starts an automated benchmark run.
3. TopicBench runs all models, logs outputs, and computes metrics.
4. TopicBench saves results, configuration, and logs with a run ID.
5. Researcher exports tables/plots for use in publications.

**Exceptions:**
- A model fails mid-benchmark.
- Disk or permission issues when saving results.

---

## UC4 – Compare topic models + LLM labels
**Related User Story:** Data Scientist  

**Actor:** Data Scientist  
**Goal:** Compare how different LLMs label topics produced by different topic modeling algorithms.  

**Preconditions:**
- Outputs from multiple topic modeling algorithms (LDA, NMF, BERTopic, etc.) are available.
- TopicBench can ingest these outputs.

**Expected Input:**
- Topic-model outputs with keyword clusters, tagged by algorithm.
- List of LLMs to evaluate.

**Expected Output:**
- Matrix or plots showing performance per (algorithm, model) pair.

**Main Flow:**
1. Data scientist loads topic-model outputs into TopicBench.
2. Selects the LLMs to evaluate.
3. TopicBench runs labeling for each (algorithm, model) combination.
4. TopicBench computes metrics for each combination.
5. TopicBench visualizes results as tables/heatmaps to inspect trade-offs.

**Exceptions:**
- Some topic-model outputs are missing required fields.
- One model fails for a specific algorithm’s topics.

---

## UC5 – Update dependencies and check nothing broke
**Related User Story:** Maintainer  

**Actor:** Maintainer  
**Goal:** Update TopicBench when models/APIs or dependencies change, and ensure everything still works.  

**Preconditions:**
- Project has a clear dependency specification (requirements, setup.py).
- Test suite exists.

**Expected Input:**
- New dependency versions or updated API specifications.

**Expected Output:**
- Updated, working code that passes all tests.

**Main Flow:**
1. Maintainer identifies a dependency or API that changed.
2. Updates config and integration code accordingly.
3. Runs the test suite (locally or via CI).
4. Fixes any failing tests.
5. Commits and documents the update.

**Exceptions:**
- Breaking changes require refactoring multiple modules.
- Tests reveal regressions that need design changes.

---

## UC6 – Contribute new label sets via pull request
**Related User Story:** Contributor  

**Actor:** Contributor  

**Goal:** Add new label sets (beyond built-in gold/human labels) and share them with others.  

**Preconditions:**
- Contributor has a GitHub account and fork of the repo.
- Standard label format (JSON schema) is documented.

**Expected Input:**
- Label file in the documented format.
- Short description of the label source or method.

**Expected Output:**
- New label set available as an option in TopicBench.

**Main Flow:**
1. Contributor prepares label file following the documented schema.
2. Runs local validation script (if provided).
3. Opens a pull request with the new labels and description.
4. Maintainer reviews, may request changes, and merges if acceptable.

**Exceptions:**
- File fails validation or violates schema.
- Missing documentation or incomplete labels.

---

## UC7 – Attempt to find API keys in the repository
**Related User Story:** Nefarious User  

**Actor:** Nefarious User  

**Goal:** Search for API keys or other private data to steal from the repository.  

**Preconditions:**
- Repository is public (or accessible).

**Expected Input:**
- Access to repository files and commit history.

**Expected Output (Desired System Behavior):**
- No real API keys or secrets present.

**Main Flow:**
1. Malicious user clones or browses the repository.
2. Searches for patterns like "api_key" or ".env".
3. Scans commit history for accidentally committed credentials.

**Expected System Response:**
- Secrets are absent because they are kept in environment variables.
- `.gitignore` prevents committing secret files.
- Docs warn users not to commit keys.

**Exceptions (Bad Case):**
- If a secret is ever found, maintainers must revoke it and rotate keys.
