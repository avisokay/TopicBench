# Task: extract author-labeled topic tables from 172 PDFs

You are helping build TOPICBENCH, a benchmark that tests whether LLMs can label
topic-model clusters the way domain-expert authors do. The benchmark needs pairs of:

  - a KEYWORD CLUSTER produced by a topic model (LDA/STM/CTM/NMF/BERTopic), and
  - the TOPIC LABEL the paper's own authors assigned to that cluster.

An earlier pass already extracted 44 papers via structured XML and PDF text-layer
parsing. These 172 are the ones that failed, almost all because the topic table is
rendered as an IMAGE (no text layer) or uses a layout that defeats column detection.
Your job is the vision pass.

## Inputs

Manifest (CSV, 172 rows):
  /Users/adam/Zotero/topicbench/vision_extraction_manifest.csv

Gold-standard examples (JSON, verified — use to calibrate what a correct row looks like):
  /Users/adam/Zotero/topicbench/gold_examples.json

Manifest columns you will use:
  priority              1..172, already ordered by which disciplines are most starved
  field                 discipline stratum
  work_id               OpenAlex id — REQUIRED in your output, it is the join key
  doi, title            paper identity
  pdf_path              absolute path to the PDF on this machine (all 172 verified to exist)
  candidate_pages       pipe-separated 1-indexed pages whose text mentions a topic-ish
                        table caption. 113 rows have these; a BLANK value does NOT mean
                        no table — it means the caption was not machine-readable, so scan
                        the whole paper.
  raster_pages          pages containing a large image; the table is often here
  prior_status          no_table_found | partial_labels | table_found_unlabeled
  prior_n_topics        topics found earlier, if any

## What counts as a valid extraction

INCLUDE a table only if it pairs model-derived keyword clusters with author-written
labels. Typical captions: "Topic labels and top terms", "Validated topic model",
"Topics identified by LDA", "Themes and representative keywords".

REJECT — these are the traps that would corrupt the benchmark:

1. LABELS THAT ARE NOT LABELS. Column detection loves running headers, journal names,
   author lists, page numbers. If your "label" is "Journal of Communication, Vol 12",
   it is page furniture. Drop the row.
2. REPRESENTATIVE QUOTES AS KEYWORDS. Many papers show "Example tweets" or
   "Representative comment" beside each topic. Those are documents, not keyword
   clusters. A keyword cluster is 5-20 short terms, often stemmed ("polic", "govern").
   If cells are full sentences, it is the wrong column.
3. LLM-GENERATED LABELS. If the paper used GPT/TopicGPT/LLooM to produce the labels,
   the row is unusable — the benchmark compares LLMs against HUMANS. Note it and skip.
4. CLASSIFIER OUTPUT. Feature-importance or regression-coefficient tables look
   structurally identical to topic tables. Check the caption.
5. NUMERIC COLUMNS. Topic proportion / prevalence / beta / coherence are not labels.

Do NOT invent, translate, normalize, or tidy labels or keywords. Copy them
VERBATIM, including stemming, capitalization, and typos. If the paper writes
"25 GM Food", the label is "25 GM Food" — do not strip the number. Fidelity to the
source is the entire point; a cleaned-up label is a wrong label.

If a table spans several pages, capture ALL topics. If a paper reports two topic
models, take the primary/final one and say so in `notes`.

## Method

Work in batches of ~15 papers so nothing is lost to a context reset. For each paper:

1. Render the candidate pages at high resolution and look at them:
      pages = candidate_pages, plus raster_pages, plus any page whose text
      matches /topic|theme/i near a table caption
   Read the page images directly — for image-based tables that is the only way.
   For a dense multi-page table, crop to one region at a time so small type stays legible.
2. Identify the keyword column and the label column. State which is which in `notes`.
3. Transcribe every topic row.
4. Self-check before writing the record:
      - len(clusters) == len(author_labels), and index i in one corresponds to index i
        in the other. THIS ALIGNMENT IS THE MOST IMPORTANT INVARIANT. A row where
        topic 3's label sits beside topic 4's keywords is worse than no row at all.
      - labels are short human phrases, not sentences, numbers, or journal names
      - keywords are short terms, not sentences
      - no label repeated where the paper shows distinct labels

## Output

Append one JSON object per paper (JSONL, one line each) to:
  /Users/adam/Zotero/topicbench/topicbench_vision_extractions.jsonl

Schema:
{"work_id": "W1234567890",
 "doi": "10.1177/...",
 "status": "extracted" | "no_table" | "llm_labeled" | "unreadable",
 "source_table": "Table 3: Topic labels and top 10 terms",
 "source_pages": [7, 8],
 "extraction_method": "vision" | "text",
 "n_topics": 12,
 "clusters": [["polic", "govern", "state"], ["vote", "elect", "campaign"]],
 "author_labels": ["Public policy", "Electoral politics"],
 "keyword_column_header": "Top 10 terms",
 "label_column_header": "Topic label",
 "algorithm": "LDA" | "STM" | "CTM" | "NMF" | "BERTopic" | "other" | "unclear",
 "labeling_process_described": true | false,
 "labeling_process_quote": "verbatim sentence describing HOW they labeled, or \"\"",
 "confidence": "high" | "medium" | "low",
 "notes": "which column was which; anything ambiguous"}

Write a record for EVERY paper including failures — `status:"no_table"` with empty
clusters is a real result and tells us the corpus ceiling. Never skip a row silently.

`labeling_process_described` is a bonus field the benchmark wants: does the paper
explain how the authors arrived at their labels (inspected top terms, read sample
documents, two coders reconciled, etc.)? Quote the sentence if so.

## Report when done

  - papers attempted / extracted / no_table / llm_labeled / unreadable
  - total topic-keyword pairs, and how many carry a non-empty author label
  - counts by `field`
  - the papers you were least sure about and why

Start with priority 1 and work down; the ordering front-loads the disciplines with
the least coverage. Do not stop at the first batch — report progress and continue.
