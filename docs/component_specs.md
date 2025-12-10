# Component Specifications for TopicBench

The TopicBench package is structured into three primary modules located in src/topicbench/. Each module handles a specific stage of the pipeline: generation, calculation, and validation.

## Component 1: Topic Label Generator
File: src/topicbench/label_topics.py

**What it does**: 
This component generates descriptive labels for topic clusters using Large Language Models. It constructs a prompt based on a field of study and keywords, sends it to a model (Local via Ollama or Cloud via OpenAI), and parses the output into a clean Python list.

**Inputs**:

DataFrame: Input data containing columns for field, keywords, and paper_title.

Model Name: String identifier for the model (e.g., llama3.2, gpt-4).

API Key Path: Optional path to a file containing the OpenAI API key.

**Outputs**:

DataFrame: Returns the original DataFrame with a new column (named after the model) containing the generated labels as strings.

**How it uses other components**:

Ollama: For running local inference.

OpenAI SDK: For running cloud inference.

Regex (re): For parsing and cleaning the unstructured text response from the LLM.

**Side Effects**:

Console I/O: Displays a progress bar (tqdm) during generation.

Network: heavy network traffic for API calls.

Compute: High CPU/GPU usage if running local models via Ollama.

## Component 2: Semantic Similarity Engine
File: src/topicbench/semantic_similarity.py

**What it does**: 
This module contains the core mathematical and logic functions for calculating the distance or similarity between two text strings. It supports multiple metrics including lexical overlap (BLEU, Jaccard), embedding-based distances (Cosine, Euclidean, Manhattan, Angular), and LLM-based evaluation.

**Inputs**:

Data: A Pandas Series (row) containing two strings to compare.

Arguments: Column names for the two strings (e.g., 'string1', 'string2').

Configuration: OpenAI API key (environment variable) if using calculate_llm_as_judge.

**Outputs**:

A pd.Series containing the calculated score (e.g., {'Cosine_Similarity': 0.85}).

**How it uses other components**:

Uses NLTK and Scikit-Learn for lexical processing.

Uses SentenceTransformers (all-MiniLM-L6-v2) to generate vector embeddings.

Uses OpenAI API for the "LLM as Judge" metric.

**Side Effects**:

Network/Disk: Downloads the SentenceTransformer model (~80MB) on the first run.

Cost: Incurs API costs when using calculate_llm_as_judge.

## Component 3: Alignment & Validation Utilities
File: src/topicbench/validate.py

**What it does**: 
This module acts as the high-level interface for the package. It provides a wrapper (score_similarity) to easily switch between different metrics and a statistical tool (compute_alignment) to classify AI performance based on human agreement variability (tau-thresholding).

**Inputs**:

Raw Text: Two strings (for score_similarity).

DataFrame: A Pandas DataFrame containing columns of similarity scores (for compute_alignment).

Parameters: Metric name (e.g., "cosine") or Tau value (standard deviation multiplier).

**Outputs**:

Float: A single similarity score.

DataFrame: A copy of the input DataFrame with two new columns: tau (the calculated threshold) and AI_alignment (1 or 0 binary flag).

**How it uses other components**:

Imports and calls specific functions from semantic_similarity.py based on the user's selected metric string.

**Side Effects**:

None (Pure functions).

# Interactions to accomplish use case 1: 

For a user wishing to evaluate a new LLM on topic labeling, they simply need to apply the first, second, and third components sequentially. The user begins by inputing the name of an LLM and an API key, if necessary, along with the example keyword data provided with TopicBench to the first component, and receives a set of labels as output. Next, they use the second component to calculate the similarity between these generated labels and the example human labels included with TopicBench. Finally, using both sets of labels as input, they apply the third component to calculate alignment between them. The greater the fraction of labels for which the human and LLM labels are aligned, the better the LLM performed.

# Components for different users

## **CS1 — Components for LLM Developer**
- Flexible API integration system supporting custom model integration  
- Performance metrics calculator (accuracy, precision, recall, F1)  
- Results comparison dashboard or report generator  
- Visualization tools for performance metrics  

---

## **CS2 — Components for Computational Social Scientist**
- Pre-configured integrations for major LLM providers  
- Comprehensive documentation with examples for non-experts  
- Interface for basic benchmarking tasks  

---

## **CS3 — Components for Original Researchers**
- Batch processing system for multiple models (e.g., OpenRouter support)  
- Data loading and preprocessing scripts, extensible to new datasets  
- Visualization library for generating publication-ready figures and tables  
- Reproducible environment specification (requirements.txt, Docker images)  
- Logging and tracking system for model and dataset updates  

---

## **CS4 — Components for Data Scientist**
- Modular data ingestion system for a variety of topic modeling outputs (LDA, NMF, BERTopic, others)  
- Tools to visualize performance matrices across algorithm–LLM combinations  

---

## **CS5 — Components for Maintainer**
- Clear dependency management (requirements.txt, setup.py)  
- Modular architecture with separation of concerns  
- Comprehensive automated test suite  
- Automated dependency update checks  

---

## **CS6 — Components for Contributor**
- Standardized label format specification (JSON schema)  
- Label validation system ensuring correct format and completeness  
- Documentation for adding new labels, templates, and preprocessing scripts  

---

## **CS7 — Components for Security (Nefarious User Prevention)**
- Explicit documentation of security best practices  
- `.gitignore` configured to exclude secrets and credentials  
- Environment variable system for API keys (with `.env.example` template)  
