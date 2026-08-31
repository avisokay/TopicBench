#!/usr/bin/env python3
"""Download open-access PDFs for the TopicBench rows that lack one.

Run this ON YOUR MAC (this sandbox has no network to publishers).

    python3 fetch_oa_pdfs.py --email avisokay@uw.edu

Reads missing_pdfs.txt (one DOI per line) from the same folder and writes PDFs to
./oa_pdfs/<doi-slug>.pdf . Re-running skips whatever it already fetched, so it is
safe to interrupt and resume.

Resolution order per DOI:
  1. Unpaywall  (api.unpaywall.org) -> best OA location
  2. Publisher-specific URL patterns for PLOS / Frontiers / BMC, which Unpaywall
     occasionally misses

Nothing here evades a paywall: it only follows links publishers themselves mark
as open access. Closed items are reported at the end so you can grab them through
UW's subscriptions.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

UA = "TopicBench-OA-fetcher/1.0 (mailto:%s)"


def slug(doi):
    return re.sub(r'[^A-Za-z0-9._-]', '_', doi.strip().lower())


def get(url, email, timeout=45, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA % email})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
        return data if binary else json.loads(data.decode("utf-8", "replace"))


def unpaywall(doi, email):
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
    d = get(url, email)
    locs = []
    best = d.get("best_oa_location")
    if best:
        locs.append(best)
    locs += [loc for loc in (d.get("oa_locations") or []) if loc is not best]
    out = []
    for loc in locs:
        for k in ("url_for_pdf", "url"):
            if loc.get(k):
                out.append(loc[k])
    return out, d.get("is_oa"), (d.get("journal_name") or "")


def guessed(doi):
    """Publisher patterns for the three families Unpaywall sometimes under-reports."""
    d = doi.lower()
    if d.startswith("10.1371/journal."):
        journal = {"pone": "plosone", "pmed": "plosmedicine", "pbio": "plosbiology"}.get(
            d.split("journal.")[1][:4], "plosone")
        return [f"https://journals.plos.org/{journal}/article/file?id={doi}&type=printable"]
    if d.startswith("10.3389/"):
        return [f"https://www.frontiersin.org/articles/{doi}/pdf"]
    if d.startswith("10.1186/"):
        return [f"https://link.springer.com/content/pdf/{doi}.pdf"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="your email (Unpaywall requires a contact address)")
    ap.add_argument("--dois", default="missing_pdfs.txt")
    ap.add_argument("--out", default="oa_pdfs")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between DOIs; be polite")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, a.dois)) as fh:
        dois = [line.strip() for line in fh if line.strip()]
    outdir = os.path.join(here, a.out)
    os.makedirs(outdir, exist_ok=True)

    got, skipped, closed, failed = [], [], [], []
    for i, doi in enumerate(dois, 1):
        dest = os.path.join(outdir, slug(doi) + ".pdf")
        if os.path.exists(dest) and os.path.getsize(dest) > 20_000:
            skipped.append(doi)
            print(f"[{i}/{len(dois)}] have  {doi}")
            continue

        urls, is_oa, journal = [], None, ""
        try:
            urls, is_oa, journal = unpaywall(doi, a.email)
        except Exception as e:
            print(f"    unpaywall failed for {doi}: {e}")
        urls += guessed(doi)

        ok = False
        for u in urls:
            try:
                blob = get(u, a.email, binary=True)
                if blob[:5] != b"%PDF-":          # usually an HTML landing page
                    continue
                with open(dest, "wb") as fh:
                    fh.write(blob)
                print(f"[{i}/{len(dois)}] OK    {doi}  ({len(blob)//1024} KB)")
                got.append(doi)
                ok = True
                break
            except Exception:
                continue
        if not ok:
            (closed if is_oa is False else failed).append((doi, journal))
            print(f"[{i}/{len(dois)}] MISS  {doi}  {'(not OA)' if is_oa is False else ''}")
        time.sleep(a.sleep)

    print(f"\ndownloaded {len(got)}   already had {len(skipped)}   "
          f"not open access {len(closed)}   failed {len(failed)}")
    for label, rows in (("NOT OPEN ACCESS", closed), ("FAILED", failed)):
        if rows:
            print(f"\n{label} — get these through UW's subscriptions:")
            for doi, j in rows:
                print(f"  https://doi.org/{doi}   {j}")
    print(f"\nPDFs are in {outdir}")
    print("Then either drag that folder into Zotero, or just tell Claude the path — "
          "the extraction pipeline matches them by DOI from the filename.")


if __name__ == "__main__":
    sys.exit(main())
