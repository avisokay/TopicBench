# Human topic labeling

Collects independent human labels for keyword clusters, to sit alongside the original
author labels and the AI-generated ones.

The interface is a **single self-contained HTML file**. Annotators need no install, no
account and no network — they open it, label, and export a JSON file. Work autosaves in
the browser, so closing the tab does not lose it.

## For the coordinator

### 1. Build the task

```bash
cd human_labeling
python3 build_task.py --ids-file final20.txt      # one work_id per line
```

Other ways to choose papers:

```bash
python3 build_task.py --suggest 20                # balanced proposal across disciplines
python3 build_task.py --work-ids W2132314509,W4392133645
python3 build_task.py --suggest 20 --kmax 12      # cap clusters per paper
python3 build_task.py --hide-abstract             # omit abstracts (see Leakage below)
```

`--suggest` prefers papers that carry a corpus description, have 5–20 clusters, and whose
topics are ≥80% `strict` tier, then spreads the pick across `field`. Treat it as a starting
point for the conversation with collaborators, not a decision — write the agreed IDs into
`final20.txt` and rebuild.

Outputs:

| file | purpose |
|---|---|
| `annotate.html` | send this to each annotator |
| `task.json` | the same payload, for the record |
| `task_papers.csv` | the paper list, for discussing the selection |
| `leakage_report.json` | clusters whose author label is visible (written only if any) |

### 2. Send `annotate.html` to each annotator

One file per person, by email or shared drive. Everyone labels the **same** clusters —
that is what makes inter-annotator agreement computable.

### 3. Merge what comes back

```bash
python3 merge_annotations.py returned/*.json
```

| file | shape |
|---|---|
| `human_labels_long.csv` | one row per (annotator, paper, cluster) |
| `human_labels_wide.csv` | one row per cluster; a column per annotator, plus `author_label` — the analysis-ready file |
| `human_labels_scored.csv` | similarity scores, written by `score_labels.py` |

It reports per-annotator completion and, for clusters with two or more labels, exact-match
agreement on normalized strings. Malformed files are skipped with a reason, and if someone
exports twice only their latest submission is used.

Exact match is a floor, not the measure. Semantic similarity is deliberately not computed
here -- it lives in `src/topicbench/semantic_similarity.py`.

### Setup (once, before step 4)

`build_task.py` and `merge_annotations.py` are stdlib-only and need nothing. `score_labels.py`
needs the package, and installing it has one trap worth knowing about on Apple Silicon:

```bash
cd /path/to/TopicBench
conda deactivate                 # if your shell auto-activates conda base
which python3                    # must NOT be .../anaconda3/bin/python3
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip        # old pip fails to build a transitive dep (libcst)
pip install -e .
```

The trap: Anaconda's base Python on macOS is an x86_64 build running under Rosetta. PyTorch
shipped no Intel-macOS wheels after 2.2.2, so pip silently resolves to that old torch, which
predates the NumPy 2.x ABI that `pyproject.toml` requires. The symptom is
`Failed to initialize NumPy: _ARRAY_API not found`, cascading into
`NameError: name 'nn' is not defined` from sentence-transformers. Use a native arm64 Python
and it resolves.

### 4. Score against the author labels

```bash
python3 score_labels.py                              # Jaccard + cosine
python3 score_labels.py --metrics cosine,llm         # add the LLM judge
python3 score_labels.py --against human__ada@x.edu   # one annotator
```

Writes `human_labels_scored.csv`, one row per (reference, comparison) pair. Unlike the
other two scripts this needs the package extras (pandas, scikit-learn,
sentence-transformers), so run it on the coordinator's machine. `--metrics llm` also needs
`OPENAI_API_KEY`.

Measured on the 27-cluster pilot (author label vs one human labeler):

| metric | mean | n > 0 |
|---|---:|---:|
| Jaccard | 0.037 | 2 / 27 |
| Cosine | 0.313 | 27 / 27 |

BLEU used to be reported here and was removed. Labels are median 2 words and never reach 4
on both sides, so BLEU's default 4-gram weights could never fire -- its maximum over the
pilot was 1.4e-231, the smoothing floor. It also rank-correlated with Jaccard at Spearman
1.00, so it said nothing Jaccard did not, while requiring the whole NLTK dependency and its
downloaded corpora. The package no longer depends on NLTK.

Jaccard fires only on literal word reuse (2 of 27). Cosine is the only metric producing a
gradient, but it is not obviously right either: it scores `Anti-Capatalist Left` /
`capitalist critique` at 0.21. The `tech_review/` notebooks (LLM-as-judge, embedding
anchors) exist for that reason, and these 27 pairs are small enough to adjudicate by hand
as a calibration set.

Pass `--no-author-label` if you want to circulate the merged file without the comparison
column in it.

## For annotators

1. Open `annotate.html` in any browser (double-click is fine).
2. Enter your name and email.
3. For each paper you see the title, corpus description, abstract, and **every keyword
   cluster at once**. Read all the clusters before writing anything — a label is a choice
   about how one cluster differs from its neighbours.
4. Type a short label next to each cluster. Revise freely; going back as the set becomes
   clearer is the intended way to work. The sidebar lists the labels you have already used
   on the current paper, which helps keep the vocabulary consistent.
5. Tick **unsure** rather than guessing silently, and **not interpretable** for a cluster
   that does not cohere. Use the paper note for anything a label cannot carry.
6. Press **Export my labels** and send the downloaded file back.

`Enter` moves to the next cluster. Work saves continuously in your browser; **Clear my
work** discards it.

## Design notes

**All clusters are visible simultaneously.** The labeling process is inductive over the
full set, so the interface never shows one cluster at a time. Paper context stays at the
top and every cluster is on the page below it, each with its own input.

**Author labels are never shown.** `build_task.py` refuses to write a task if any
author-label field reaches the payload, and the built HTML contains none. The comparison
happens later, in `merge_annotations.py`.

**Leakage is measured, not assumed.** An author label can still be recoverable when it is
itself one of its cluster's keywords, or when it is a proper noun that necessarily appears
in the title or corpus description. Every build reports how many clusters are affected and
writes the detail to `leakage_report.json`. On a 20-paper / 180-cluster build this was 3
clusters (1.7%), none of them via the abstract — which is why abstracts are shown by
default. Use `--hide-abstract` if your selection changes that.

**One annotator, one file.** Identity is the email address. The merge keys on it, so an
annotator who exports twice does not become two people.
