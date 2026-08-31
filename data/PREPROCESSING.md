# Preprocessing for the label-comparison analysis

`topicbench_topics.csv` — one row per **topic** (n=2,290), the unit at which
you compare an author label against an AI or alternative-human label.
Derived from `topicbench_bibliography.csv` (one row per paper).

## Columns

| column | use |
|---|---|
| `topic_uid` | join key, `<work_id>_t<NN>` |
| `author_label` | cleaned label — **compare against this** |
| `author_label_raw` | verbatim from the paper, for audit |
| `keywords` | cleaned keyword list (JSON) — **prompt with this** |
| `keywords_raw` | verbatim, for audit |
| `eval_tier` | `strict` / `core` / `sensitivity` / `excluded` |
| `flags` | why a row is not `strict` |
| `k_paper` | topics in the source model (difficulty covariate) |

## Tiers

Nothing is deleted; every row carries flags so you can set your own threshold.

| tier | n | share | meaning |
|---|---|---|---|
| `strict` | 1,807 | 79% | clean, English, no contamination — headline analysis |
| `core` | 388 | 17% | adds trivial-label and abstract-leakage cases |
| `sensitivity` | 55 | 2% | non-English or sentence-like labels |
| `excluded` | 40 | 2% | degenerate — no usable label |

Report `strict` as primary and `core` as robustness. If the two diverge, the
contaminated subsets are doing work and that itself is a finding.

## What the cleaning does, and why

Each rule below was measured on the corpus, not assumed.

**1. Strip topic-index prefixes — 12.2% of labels.**
Tables number their rows, and the number bleeds into the label cell:
`"25 GM Food"`, `"26. counsel"`, `"3. Campaign Finance"`. An AI labeler never
sees the index, so leaving it in guarantees a string mismatch that has nothing
to do with semantic agreement. Applied to `author_label`; `author_label_raw`
keeps the original.

**2. Do NOT filter on label length.** This was the intuition to check first, and
the data says it would backfire. Sentence-like labels are only **1.4%** (21
rows). A naive `>= 7 words` cut would remove **84 rows**, of which **82 are
valid noun phrases** — "Member state support and guidance for emissions
reduction", "Aviation compliance to GHG reduction and Projects of Common
Interest". Those are exactly the hard, informative cases a benchmark should
keep. Use the syntactic flag (`label_sentence_like`: finite verb, pronoun, or
terminal punctuation), not word count. Median label is **2 words**; 71% are 1-3.

**3. Flag labels that are literally one of their own keywords — 5.3%.**
`"Islam" ← [islam, muslime, religion, ...]`, `"FEMA" ← [disast, fema, storm]`.
No abstraction is required, so any method scores near-perfectly. Keeping them in
the headline number inflates measured agreement; they belong in a separate
difficulty stratum.

**4. Flag labels appearing verbatim in the paper's own abstract — 7.2%.**
Only matters if you give the model the abstract or full text as context — then
it is retrieval, not labeling. Harmless for keyword-only prompting. Kept in
`core`, excluded from `strict` so the headline is safe under either design.

**5. Handle non-English separately — 1.9% (28 topics, 13 papers).**
Chinese, German, Arabic, French keyword sets. Two distinct problems: an English
label over non-English keywords, and both in another language. Where the paper
printed inline glosses — `服务态度(service attitude)` — the gloss is extracted
into `keywords` and the row flagged `translation_used`. Decide deliberately:
including them tests cross-lingual labeling, which is a different construct
from what you are measuring.

**6. Strip probability weights from keyword cells — 0.7%.**
Some tables print `dog: 0.42677134`. Numbers are not part of the keyword and
would leak topic-proportion information into the prompt.

**7. Watch repeated labels within a paper — 84 topics across 6 papers.**
Some papers assign *category* labels rather than per-topic names, so 41 topics
collapse to 21 distinct labels. Exact-match scoring is ill-defined here. Not
auto-excluded (they are legitimate), but cluster your standard errors by
`work_id` regardless — topics within a paper share a corpus, a preprocessing
pipeline, and one labeler's habits, so they are not independent observations.

**8. Two flags come from the local vision pass.**
`keyword_order_unreliable` (62 topics) marks clusters read from word clouds, where
font size gives approximate rather than exact rank — membership is sound, order is
not, so do not use rank position as a feature on these. `source_row_defect` marks a
mislabel *in the published paper* (a panel title that does not match its own terms);
it is flagged rather than silently corrected. `labels_borrowed_from_prior_study`
marks a replication whose labels were written by the original authors — human-written
and valid, but not this paper's team.

## Things to decide before you run the comparison

- **Prompt symmetry.** Authors saw top terms *plus* representative documents,
  their research question, and domain knowledge. A keyword-only prompt is a
  harder task than the one the humans performed. Either state this as the
  design, or supply matched context.
- **Metric.** Exact match is too strict for a 2-word median ("Electoral
  politics" vs "Elections and voting"). Embedding cosine plus a human-adjudicated
  subsample on a random slice is the defensible pairing.
- **Number of keywords shown.** `n_keywords` ranges 2–47 (median 10). Fix it
  (e.g. top 10) or the prompt difficulty varies by paper.
- **Field imbalance.**General Social Science has 253 strict topics, Education 24. Report per-field or weight.

## Scope note

Every paper is tagged `curated_source`. `stratified_harvest` (163 papers) is the OpenAlex stratified draw, including papers whose full text was only retrieved later through institutional access. `purposive` (15 papers) were added from domain familiarity because they were known to publish topic tables; they are benchmark items and are excluded from prevalence estimates. Methods-only papers were dropped for failing the inclusion criteria.
## Provenance flags from extraction

Three flags come from the extraction records rather than from text analysis. `keyword_order_unreliable` (112 topics) marks word-cloud clusters whose keyword order is font-size-derived and only approximate. `source_row_defect` (24 topics) marks a mislabeled panel in the published figure, preserved rather than corrected. `caption_not_topic` (32 topics) marks a table whose caption did not identify it as a topic table. Where the extraction recorded a flag per topic row it is applied per row; where it recorded only a paper-level flag it applies to every topic of that paper. `source_row_defect` sets `excluded`; the others set `core`.
