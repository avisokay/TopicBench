#!/usr/bin/env python3
"""Score merged human labels against the author labels.

Uses the repo's own similarity functions (src/topicbench/semantic_similarity.py)
rather than reimplementing them.

    python3 score_labels.py human_labels_wide.csv
    python3 score_labels.py human_labels_wide.csv --metrics overlap,cosine
    python3 score_labels.py human_labels_wide.csv --against human__a@x.edu

Writes human_labels_scored.csv with one row per (reference, comparison) pair.

Unlike merge_annotations.py this is NOT stdlib-only: it needs the package extras
(pandas, scikit-learn, sentence-transformers). Run it on the coordinator's
machine, not the annotators'.
"""
import argparse
import importlib.util
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_similarity():
    """Load semantic_similarity.py directly.

    Loading by path keeps this script independent of how the package is
    installed, so it works from a source checkout without `pip install -e .`.
    """
    path = os.path.join(ROOT, "src", "topicbench", "semantic_similarity.py")
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    spec = importlib.util.spec_from_file_location("tb_semantic_similarity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wide_csv", nargs="?",
                   default=os.path.join(HERE, "human_labels_wide.csv"))
    p.add_argument("--reference", default="author_label",
                   help="column treated as the reference label")
    p.add_argument("--against", default=None,
                   help="comma-separated columns to score; default = every human__* column")
    p.add_argument("--metrics", default="overlap,cosine",
                   help="overlap (Jaccard), cosine, euclidean, angular, llm")
    p.add_argument("--out", default=os.path.join(HERE, "human_labels_scored.csv"))
    a = p.parse_args(argv)

    ss = load_similarity()
    df = pd.read_csv(a.wide_csv)
    if a.reference not in df.columns:
        p.error(f"no column {a.reference!r} in {a.wide_csv}")
    cols = ([c.strip() for c in a.against.split(",")] if a.against
            else [c for c in df.columns if c.startswith("human__")])
    cols = [c for c in cols if c in df.columns]
    if not cols:
        p.error("no comparison columns found")
    wanted = [m.strip() for m in a.metrics.split(",") if m.strip()]

    FN = {"overlap": ss.calculate_overlap,
          "cosine": ss.calculate_embedding_cosine,
          "euclidean": ss.calculate_embedding_euclidean,
          "angular": ss.calculate_embedding_angular,
          "llm": ss.calculate_llm_as_judge}
    unknown = [m for m in wanted if m not in FN]
    if unknown:
        p.error(f"unknown metric(s): {unknown}; choose from {sorted(FN)}")

    out = []
    for col in cols:
        sub = df[df[a.reference].notna() & df[col].notna()].copy()
        sub = sub[sub[a.reference].astype(str).str.strip().ne("")
                  & sub[col].astype(str).str.strip().ne("")]
        if sub.empty:
            print(f"{col}: no comparable rows, skipped", file=sys.stderr)
            continue
        res = sub[["work_id", "topic_index", "field", a.reference, col]].copy()
        res = res.rename(columns={col: "comparison_label", a.reference: "reference_label"})
        res.insert(2, "comparison_column", col)
        for m in wanted:
            t0 = time.time()
            try:
                scores = sub.apply(lambda r: FN[m](r, a.reference, col), axis=1)
            except Exception as e:
                print(f"{col}: metric {m!r} failed: {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue
            for c in scores.columns:
                res[c] = scores[c].values
            print(f"{col}: {m} over {len(sub)} pairs in {time.time()-t0:.1f}s")
        out.append(res)

    if not out:
        raise SystemExit("no scores produced")
    scored = pd.concat(out, ignore_index=True)
    scored.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(scored)} rows)")

    num = [c for c in scored.columns
           if scored[c].dtype.kind in "fc" and c != "topic_index"]
    if num:
        print("\n" + scored[num].describe().loc[["mean", "50%", "min", "max"]]
              .round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
