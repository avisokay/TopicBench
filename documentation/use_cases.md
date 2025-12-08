# Use Cases for TopicBench

## UC1.1 – Benchmark newly developed LLMs against established models

**Actor:** LLM Developer  

**Goal / Description:**  
Benchmark newly developed LLMs against established models.

**Preconditions:**
- Developer has access to a new LLM and baseline models.
- Keyword clusters and reference labels are available.

**Expected Input:**
- Keyword clusters derived from topic modeling.
- Predictions from the newly developed LLM.
- Predictions or stored results from established models.

**Expected Output:**
- Comparative metrics (e.g., accuracy, precision, recall, F1) for new vs. established models.
- Summary report of relative performance.

**Main Flow:**
1. User selects dataset and models (new + established).
2. System runs labeling with each model.
3. System computes evaluation metrics.
4. System produces a comparison report.


---

## UC1.2 – Evaluate model performance on keyword cluster labeling tasks

**Actor:** LLM Developer  

**Goal / Description:**  
Evaluate model performance on keyword cluster labeling tasks.

**Preconditions:**
- Keyword clusters and gold/human labels exist.

**Expected Input:**
- Keyword clusters.
- LLM-generated labels.
- Gold or human labels.

**Expected Output:**
- Evaluation metrics for the model on the labeling task.

**Main Flow:**
1. User loads clusters and reference labels.
2. System calls the LLM to generate labels.
3. System compares predictions to references.
4. System stores and displays metrics.


---
## UC2.1 – Compare off-the-shelf LLMs without writing API integration code

**Actor:** Computational Social Scientist  

**Goal / Description:**  
Compare off-the-shelf LLMs without writing API integration code.

**Preconditions:**
- TopicBench has pre-configured integrations for major LLM providers.

**Expected Input:**
- Dataset with keyword clusters.
- List of pre-configured models to compare.

**Expected Output:**
- Ranking or table of model performance on the dataset.

**Main Flow:**
1. User selects dataset and built-in models.
2. System runs benchmarks with default settings.
3. System computes and displays comparison metrics.


---

## UC2.2 – Select optimal model for specific research datasets

**Actor:** Computational Social Scientist  

**Goal / Description:**  
Select optimal model for specific research datasets.

**Preconditions:**
- Several candidate models are available in TopicBench.

**Expected Input:**
- Research dataset (topic modeling output).
- List of candidate models.

**Expected Output:**
- Recommendation or clear comparison enabling model selection.

**Main Flow:**
1. User chooses dataset and candidate models.
2. System benchmarks each model.
3. User reviews results and picks the most suitable model.

---

## UC3.1 – Automate benchmarking across multiple LLMs simultaneously

**Actor:** Original Researcher  

**Goal / Description:**  
Automate benchmarking across multiple LLMs simultaneously.

**Expected Input:**
- List of models to benchmark.
- Dataset(s) and configuration (metrics, prompts).

**Expected Output:**
- Benchmark results for all models in one run.

**Main Flow:**
1. User specifies models and benchmark configuration.
2. System runs all models automatically.
3. System saves combined results.


---

## UC3.2 – Generate publication-ready figures and tables

**Actor:** Original Researcher  

**Goal / Description:**  
Generate publication-ready figures and tables.

**Expected Input:**
- Benchmark results stored in TopicBench.

**Expected Output:**
- Figures and tables suitable for papers (e.g., PDF/PNG/LaTeX/CSV).

**Main Flow:**
1. User selects results and desired formats.
2. System generates formatted figures and tables.


---

## UC4.1 – Compare LLM performance across different topic modeling algorithms

**Actor:** Data Scientist  

**Goal / Description:**  
Compare LLM performance across different topic modeling algorithms (LDA, NMF, BERTopic).

**Expected Input:**
- Topic-model outputs from multiple algorithms.
- LLM configurations.

**Expected Output:**
- Performance statistics per (algorithm, model) combination.

**Main Flow:**
1. User loads topic outputs and associates them with algorithms.
2. System runs LLM labeling over each algorithm’s topics.
3. System aggregates and displays comparison metrics.


---

## UC4.2 – Cross-comparison analysis tools

**Actor:** Data Scientist  

**Goal / Description:**  
Cross-comparison analysis tools.

**Expected Input:**
- Benchmark results across algorithms and models.

**Expected Output:**
- Visualizations (e.g., matrices) that support cross-comparison.

**Main Flow:**
1. System provides analysis views over results.
2. User explores trade-offs between topic methods and LLM performance.

---

## UC5.1 – Update LLM API integrations when providers change APIs

**Actor:** Maintainer  

**Goal / Description:**  
Update LLM API integrations when providers change APIs.

**Expected Input:**
- New API specs, keys, and endpoints.

**Expected Output:**
- Updated integration layer that still passes tests.

**Main Flow:**
1. Maintainer modifies integration code.
2. Maintainer runs tests.
3. Changes are merged after successful checks.


---
## UC5.2 – Run tests to ensure updates don't break functionality

**Actor:** Maintainer  

**Goal / Description:**  
Run tests to ensure updates don't break functionality.

**Expected Input:**
- Test suite and updated code.

**Expected Output:**
- Pass/fail test results.

**Main Flow:**
1. Maintainer runs local tests or CI.
2. System reports any failures.
3. Maintainer fixes issues before release.


---

## UC6.1 – Contribute new gold standard labels for existing keyword clusters

**Actor:** Contributor  

**Goal / Description:**  
Contribute new gold standard labels for existing keyword clusters.

**Expected Input:**
- New label sets in standardized format.

**Expected Output:**
- Additional label resources available in TopicBench.

**Main Flow:**
1. Contributor prepares label file.
2. Contributor submits via pull request.
3. Maintainer reviews and merges.


---
## UC6.2 – Submit labels via pull request

**Actor:** Contributor  

**Goal / Description:**  
Submit labels via pull request.

**Expected Input:**
- GitHub fork/branch with label files.

**Expected Output:**
- Merged contribution adding labels.

**Main Flow:**
1. Contributor opens PR.
2. Maintainers review and request changes if needed.
3. PR is merged.


---
## UC7.1 – Search repository for exposed API keys

**Actor:** Nefarious User  

**Goal / Description:**  
Search repository for exposed API keys.

**Expected Input:**
- Public repository contents and commit history.

**Expected Output (Desired System Behavior):**
- No real API keys present; search yields no usable secrets.

**Main Flow:**
1. User searches repo files and history.
2. System design ensures keys are not stored there.


---
## UC7.2 – Find credentials in commit history

**Actor:** Nefarious User  

**Goal / Description:**  
Find credentials in commit history.

**Expected Input:**
- Git history of the repository.

**Expected Output (Desired System Behavior):**
- History does not contain secrets; any config uses placeholders.

**Main Flow:**
1. User inspects past commits.
2. Repository policy and tooling prevent committing secrets.
