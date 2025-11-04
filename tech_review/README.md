# Evaluating the Semantic Similarity of Two Strings
## Technical Review of Third Party Options

The goal is to evaluate the semantic similarity between two strings. For example, given the strings "The cat sat on the mat" and "A feline rested atop a rug", we want to determine how similar their meanings are, and represent that similarity as a score between 0 (completely unrelated/orthogonal) and 1 (exactly the same). There are several third-party libraries and tools that can be used to achieve this. We categorize some examples below by the approach they use. They are implemented in the python notebooks located in the `tech_review/notebooks/semantic_similarity/` directory: <br><br>
`01-token_overlap.ipynb`<br>
`02-sentence_embeddings_cosine.ipynb`<br>
`03-sentence_embeddings_other.ipynb`<br>
`04-llm_as_judge.ipynb`

Cases to evaluate can be found in the `evaluation_cases.csv` file. They include:
- "the cat sat on the mat" vs. "a feline rested atop a rug"
- "he ran quickly to the store" vs. "she ran quickly to the store"
- "domestic unrest" vs. "political instability in the country"
- "turn left at the traffic light" vs. "photosynthesis occurs in plant cells"

### Technical Approaches

1. Token Overlap (e.g., GLU, Jaccard, BLEU)<br>
Use these for surface-level lexical similarity:

- `NLTK`: Offers tokenization, BLEU score, and other lexical similarity metrics.

- `Scikit-learn`: Use `CountVectorizer` or `TfidfVectorizer` to compute token overlap and cosine similarity.

- `TextDistance`: Implements Jaccard, Levenshtein, Hamming, and other string similarity metrics.

```python
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np

# Tokenize
tokens1 = word_tokenize(text1.lower())
tokens2 = word_tokenize(text2.lower())

# BLEU Score
bleu = sentence_bleu([tokens1], tokens2)
print(f"BLEU Score: {bleu:.4f}")

# Jaccard Similarity
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform([text1, text2]).toarray()
jaccard = jaccard_score(X[0], X[1])
print(f"Jaccard Similarity: {jaccard:.4f}")
```

2. Sentence Embeddings + Cosine Similarity<br>
Use these for semantic similarity via vector space:

- `SentenceTransformers`: Most popular for generating sentence embeddings using models like `BERT`, `RoBERTa`, etc. Easily compute cosine similarity between embeddings.

- `Scikit-learn`: Use cosine_similarity from `sklearn.metrics.pairwise` to compare embeddings.

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode("Text A", convert_to_tensor=True)
emb2 = model.encode("Text B", convert_to_tensor=True)
similarity = util.cos_sim(emb1, emb2)
```

3. Sentence Embeddings + Other Comparisons<br>
Beyond cosine similarity:

- `SentenceTransformers`: Most popular for generating sentence embeddings using models like `BERT`, `RoBERTa`, etc. Easily compute cosine similarity between embeddings.

- `Euclidean Distance`: Available via `scipy.spatial.distance.euclidean`.

- `Manhattan Distance`, `Dot Product`, or `Angular Distance`: Use `scipy` or `sklearn` for these.

```python
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import distance
model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode("Text A", convert_to_tensor=True)
emb2 = model.encode("Text B", convert_to_tensor=True)
similarity = distance.euclidean(emb1, emb2)
```

4. LLM as Judge<br>
Use large language models to evaluate similarity with reasoning:

- `OpenAI GPT` / `Claude` / `Mistral` via APIs: Prompt the model with both texts and ask for a similarity score or judgment.

- `Ollama`: Load models like GPT-2, LLaMA, or Mistral locally and prompt them similarly.

```python
prompt = f"Compare the following texts and rate their semantic similarity from 0 (orthogonal) to 1 (exact match):\nText A: {text1}\nText B: {text2}"
```