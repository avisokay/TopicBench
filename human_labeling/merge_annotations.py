#!/usr/bin/env python3
"""Merge exported human label files into one tidy table.

    python3 merge_annotations.py returned/*.json

Writes:
  human_labels_long.csv   one row per (annotator, paper, cluster)
  human_labels_wide.csv   one row per cluster, one column per annotator, plus the
                          author label for comparison -- the analysis-ready file

Reports per-annotator completion and, where two or more annotators labeled the same
cluster, exact and token-overlap agreement. Standard library only.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

SCHEMA = "topicbench.human_labels.v1"


def norm(s):
    """Lowercase, strip punctuation, collapse whitespace -- for comparison only."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())).strip()


def load_submissions(paths):
    subs = []
    for p in paths:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (ValueError, OSError) as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        if d.get("schema") != SCHEMA:
            print(f"skip {p}: not a {SCHEMA} file", file=sys.stderr)
            continue
        who = (d.get("annotator") or {})
        if not who.get("email"):
            print(f"skip {p}: no annotator email", file=sys.stderr)
            continue
        d["_path"] = p
        subs.append(d)
    # One submission per annotator: keep the latest if someone exports twice.
    best = {}
    for d in subs:
        k = norm(d["annotator"]["email"])
        if k not in best or (d.get("submitted") or "") > (best[k].get("submitted") or ""):
            best[k] = d
    dropped = len(subs) - len(best)
    if dropped:
        print(f"note: {dropped} superseded re-export(s) ignored", file=sys.stderr)
    return list(best.values())


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="exported *.json files (globs allowed)")
    p.add_argument("--task", default=os.path.join(here, "task.json"))
    p.add_argument("--bibliography",
                   default=os.path.join(root, "data", "topicbench_bibliography.csv"))
    p.add_argument("--out-dir", default=here)
    p.add_argument("--no-author-label", action="store_true",
                   help="omit the author label column from the wide file")
    a = p.parse_args(argv)

    paths = []
    for f in a.files:
        paths.extend(sorted(glob.glob(f)) if any(ch in f for ch in "*?[") else [f])
    subs = load_submissions(paths)
    if not subs:
        p.error("no valid submissions")

    task = json.load(open(a.task, encoding="utf-8")) if os.path.exists(a.task) else None
    if task:
        for d in subs:
            if d.get("task_id") != task.get("task_id"):
                print("warning: %s is for task %s, not %s"
                      % (os.path.basename(d["_path"]), d.get("task_id"),
                         task.get("task_id")), file=sys.stderr)

    author, meta = {}, {}
    if os.path.exists(a.bibliography):
        for r in csv.DictReader(open(a.bibliography, newline="", encoding="utf-8")):
            meta[r["work_id"]] = r
            try:
                for i, lab in enumerate(json.loads(r.get("author_label") or "[]")):
                    author[(r["work_id"], i)] = str(lab).strip()
            except (ValueError, TypeError):
                pass

    long_rows = []
    for d in subs:
        who = d["annotator"]
        for L in d.get("labels") or []:
            long_rows.append({
                "annotator_name": who.get("name", ""),
                "annotator_email": who.get("email", ""),
                "task_id": d.get("task_id", ""),
                "work_id": L.get("work_id", ""),
                "topic_index": L.get("topic_index", ""),
                "human_label": (L.get("human_label") or "").strip(),
                "unsure": bool(L.get("unsure")),
                "not_interpretable": bool(L.get("not_interpretable")),
                "keywords": json.dumps(L.get("keywords") or [], ensure_ascii=False),
                "labeled_at": L.get("labeled_at") or "",
                "submitted": d.get("submitted", ""),
                "paper_note": (d.get("paper_notes") or {}).get(L.get("work_id"), ""),
            })
    if not long_rows:
        p.error("submissions contain no label rows")
    long_rows.sort(key=lambda r: (r["annotator_email"], r["work_id"],
                                  int(r["topic_index"] or 0)))
    lp = os.path.join(a.out_dir, "human_labels_long.csv")
    with open(lp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        w.writerows(long_rows)

    people = sorted({r["annotator_email"] for r in long_rows})
    cells = collections.defaultdict(dict)
    for r in long_rows:
        cells[(r["work_id"], int(r["topic_index"]))][r["annotator_email"]] = r
    cols = ["work_id", "topic_index", "title", "field", "keywords"]
    if not a.no_author_label:
        cols.append("author_label")
    for e in people:
        cols += ["human__" + e, "unsure__" + e]
    cols += ["n_human_labels", "exact_agreement"]

    wide = []
    for (w_, i), byp in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        any_r = next(iter(byp.values()))
        labs = [byp[e]["human_label"] for e in people
                if e in byp and byp[e]["human_label"]]
        row = {"work_id": w_, "topic_index": i,
               "title": (meta.get(w_) or {}).get("title", ""),
               "field": (meta.get(w_) or {}).get("field", ""),
               "keywords": any_r["keywords"],
               "n_human_labels": len(labs),
               "exact_agreement": ("" if len(labs) < 2
                                   else int(len({norm(x) for x in labs}) == 1))}
        if not a.no_author_label:
            row["author_label"] = author.get((w_, i), "")
        for e in people:
            row["human__" + e] = byp.get(e, {}).get("human_label", "")
            row["unsure__" + e] = byp.get(e, {}).get("unsure", "")
        wide.append(row)
    wp = os.path.join(a.out_dir, "human_labels_wide.csv")
    with open(wp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(wide)

    print("wrote %s  (%d rows)" % (lp, len(long_rows)))
    print("wrote %s  (%d clusters x %d annotator(s))" % (wp, len(wide), len(people)))
    print("")
    print("per annotator:")
    for e in people:
        mine = [r for r in long_rows if r["annotator_email"] == e]
        done = [r for r in mine if r["human_label"]]
        uns = sum(1 for r in done if r["unsure"])
        noise = sum(1 for r in mine if r["not_interpretable"])
        name = next(r["annotator_name"] for r in mine)
        print("  %s <%s>: %d/%d labeled, %d unsure, %d not interpretable"
              % (name, e, len(done), len(mine), uns, noise))

    multi = [r for r in wide if r["n_human_labels"] >= 2]
    print("")
    if multi:
        ex = [r["exact_agreement"] for r in multi if r["exact_agreement"] != ""]
        print("agreement over %d multiply-labeled clusters:" % len(multi))
        print("  exact (normalized) : %d/%d = %.0f%%"
              % (sum(ex), len(ex), 100.0 * sum(ex) / len(ex)))
        print("  exact match is a floor, not the measure -- for semantic agreement run")
        print("  score_labels.py, which uses src/topicbench/semantic_similarity.py")
    else:
        print("no cluster has two or more labels yet, so no agreement to compute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
