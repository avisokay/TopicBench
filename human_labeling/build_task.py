#!/usr/bin/env python3
"""Build a human topic-labeling task from the TopicBench corpus.

Writes two files:

  task.json      the papers and keyword clusters to be labeled (no author labels)
  annotate.html  a self-contained annotation interface -- no server, no install

The annotator opens annotate.html in any browser, enters name and email, labels
every cluster, and exports a JSON file. merge_annotations.py ingests those.

    python3 build_task.py --suggest 20            # propose a balanced set
    python3 build_task.py --work-ids W123,W456    # use an explicit set
    python3 build_task.py --ids-file final20.txt  # one work_id per line

Standard library only.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
import random
import sys

# Columns that must never reach the annotator: seeing the authors' own label, or
# any figure derived from it, would contaminate the independent-labeling design.
FORBIDDEN = ("author_label", "n_topics_labeled", "label_coverage", "labels_complete",
             "labeling_process_quote", "authors detail their labeling process")


def load_corpus(bib_csv, topics_csv):
    bib = list(csv.DictReader(open(bib_csv, newline="", encoding="utf-8")))
    topics = list(csv.DictReader(open(topics_csv, newline="", encoding="utf-8")))
    by_work = collections.defaultdict(list)
    for r in topics:
        by_work[r["work_id"]].append(r)
    for w in by_work:
        by_work[w].sort(key=lambda r: int(r["topic_index"]))
    return bib, by_work


def suggest(bib, by_work, n, seed=0, kmin=5, kmax=20):
    """Propose n papers spread across disciplines.

    Prefers papers whose cluster count is small enough to label in one sitting,
    that carry a corpus description, and whose topics are mostly `strict`.
    """
    rng = random.Random(seed)
    cand = []
    for r in bib:
        rows = by_work.get(r["work_id"]) or []
        if not rows or not (kmin <= len(rows) <= kmax):
            continue
        if not (r.get("corpus") or "").strip():
            continue
        strict = sum(1 for t in rows if t["eval_tier"] == "strict") / len(rows)
        if strict < 0.8:
            continue
        cand.append((r, len(rows), strict))
    by_field = collections.defaultdict(list)
    for r, k, s in cand:
        by_field[r["field"]].append((r, k, s))
    for f in by_field:
        by_field[f].sort(key=lambda x: (-x[2], x[1]))
    picked, fields = [], sorted(by_field, key=lambda f: -len(by_field[f]))
    while len(picked) < n and any(by_field.values()):
        for f in fields:
            if len(picked) >= n:
                break
            if by_field[f]:
                picked.append(by_field[f].pop(0)[0])
    rng.shuffle(picked)
    return picked[:n]


def build_task(papers, by_work, hide_abstract=False):
    out = []
    for r in papers:
        rows = by_work[r["work_id"]]
        out.append({
            "work_id": r["work_id"],
            "title": r["title"],
            "authors": r.get("authors", ""),
            "year": r.get("year", ""),
            "journal": r.get("journal", ""),
            "field": r.get("field", ""),
            "doi": r.get("doi", ""),
            "abstract": "" if hide_abstract else (r.get("abstract") or ""),
            "corpus": r.get("corpus", ""),
            "method": (r.get("method") or "").upper(),
            "clusters": [{"i": int(t["topic_index"]),
                          "keywords": json.loads(t["keywords"])} for t in rows],
        })
    payload = {
        "task_id": hashlib.sha256(
            ("|".join(p["work_id"] for p in out)).encode()).hexdigest()[:12],
        "created": datetime.date.today().isoformat(),
        "n_papers": len(out),
        "n_clusters": sum(len(p["clusters"]) for p in out),
        "hide_abstract": bool(hide_abstract),
        "papers": out,
    }
    blob = json.dumps(payload)
    leaked = [k for k in FORBIDDEN if k in blob]
    if leaked:
        raise SystemExit(f"refusing to write: author-label fields leaked: {leaked}")
    return payload


def leakage_report(payload, bib_index):
    """Clusters whose author label is recoverable from what the annotator sees.

    Two benign causes dominate: the label is itself one of the cluster's keywords,
    and the label is a proper noun that necessarily appears in the title or corpus
    description. Reported per build so it is a known quantity rather than a surprise.
    """
    def norm(s):
        import re as _re
        return _re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    out = []
    for p_ in payload["papers"]:
        row = bib_index.get(p_["work_id"]) or {}
        try:
            labs = json.loads(row.get("author_label") or "[]")
        except (ValueError, TypeError):
            continue
        for c in p_["clusters"]:
            lab = str(labs[c["i"]]).strip() if c["i"] < len(labs) else ""
            n = norm(lab).strip()
            if not n or len(n) < 8:
                continue
            where = []
            if any(n == norm(k).strip() for k in c["keywords"]):
                where.append("keywords")
            for fld in ("abstract", "title", "corpus"):
                if n in norm(p_.get(fld) or ""):
                    where.append(fld)
            if where:
                out.append({"work_id": p_["work_id"], "topic_index": c["i"],
                            "author_label": lab, "visible_in": where})
    return out


CSS = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 color:#1f2430;background:#f4f6f8}
button{font:inherit;cursor:pointer}
input,textarea{font:inherit;color:inherit}
.hide{display:none!important}

/* ---------- gate ---------- */
#gate{max-width:560px;margin:8vh auto;background:#fff;border:1px solid #dfe3e8;
 border-radius:10px;padding:1.8rem 2rem;box-shadow:0 1px 4px rgba(0,0,0,.05)}
#gate h1{margin:0 0 .2rem;font-size:1.35rem}
#gate .sub{color:#5b6472;font-size:.9rem;margin-bottom:1.3rem}
#gate label{display:block;font-weight:600;font-size:.82rem;margin:.9rem 0 .25rem}
#gate input{width:100%;padding:.55rem .7rem;border:1px solid #cfd6de;border-radius:6px}
#gate .err{color:#b3261e;font-size:.82rem;min-height:1.1rem;margin-top:.5rem}
.primary{background:#1b4d2e;color:#fff;border:0;border-radius:6px;
 padding:.6rem 1.1rem;font-weight:600}
.primary:disabled{background:#9db3a6;cursor:not-allowed}
.ghost{background:#fff;border:1px solid #cfd6de;border-radius:6px;padding:.45rem .85rem}
.brief{background:#f7fbf8;border:1px solid #cfe0d5;border-radius:7px;
 padding:.7rem .9rem;font-size:.85rem;color:#2b3b31;margin-top:1.2rem}
.brief ul{margin:.4rem 0 0;padding-left:1.1rem}
.brief li{margin:.18rem 0}

/* ---------- shell ---------- */
#app{display:grid;grid-template-columns:1fr 250px;gap:1.1rem;
 max-width:1400px;margin:0 auto;padding:1rem 1.2rem 4rem}
@media(max-width:1080px){#app{grid-template-columns:1fr}#rail{position:static!important}}

/* ---------- top bar ---------- */
#bar{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid #dfe3e8;
 padding:.55rem 1.2rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
#bar .who{font-size:.82rem;color:#5b6472}
#bar .spacer{flex:1}
#prog{height:6px;background:#e6eaee;border-radius:4px;width:190px;overflow:hidden}
#prog div{height:100%;background:#2f7d4f;width:0;transition:width .2s}
#progtxt{font-size:.8rem;color:#5b6472;font-variant-numeric:tabular-nums}
#saved{font-size:.76rem;color:#2f7d4f;min-width:92px}

/* ---------- paper context ---------- */
.paper{background:#fff;border:1px solid #dfe3e8;border-radius:10px;padding:1.1rem 1.3rem;
 margin-bottom:.9rem}
.paper h2{margin:0 0 .3rem;font-size:1.16rem;line-height:1.35}
.meta{font-size:.82rem;color:#5b6472;margin-bottom:.8rem}
.meta .fld{background:#eef3f7;border-radius:10px;padding:.05rem .45rem;margin-left:.35rem}
.block{margin-top:.7rem}
.block .lbl{font-size:.72rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
 color:#7a8593;margin-bottom:.2rem}
.corpusbox{background:#fffdf5;border:1px solid #ecdfae;border-radius:7px;
 padding:.5rem .7rem;font-size:.92rem}
.abs{font-size:.9rem;color:#333b47;max-height:7.2rem;overflow:hidden;position:relative}
.abs.open{max-height:none}
.kwnote{margin-top:.8rem;font-size:.8rem;color:#5b6472;background:#f6f8fa;border:1px solid #e3e8ee;border-radius:6px;padding:.4rem .6rem}
.absmore{background:none;border:0;color:#1b4d2e;font-size:.8rem;font-weight:600;
 padding:.2rem 0}

/* ---------- clusters ---------- */
#clusters{display:flex;flex-direction:column;gap:.55rem}
.cl{background:#fff;border:1px solid #dfe3e8;border-radius:9px;padding:.7rem .85rem;
 display:grid;grid-template-columns:2.2rem 1fr 20rem;gap:.75rem;align-items:start}
@media(max-width:900px){.cl{grid-template-columns:2.2rem 1fr}.cl .in{grid-column:2}}
.cl.done{border-color:#bcd8c6;background:#fcfefd}
.cl .num{font-weight:700;color:#7a8593;font-variant-numeric:tabular-nums;
 font-size:.9rem;padding-top:.35rem}
.kws{display:flex;flex-wrap:wrap;gap:.25rem}
.kw{background:#eef2f6;border:1px solid #dde4ea;border-radius:4px;
 padding:.1rem .4rem;font-size:.83rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.in input{width:100%;padding:.45rem .6rem;border:1px solid #cfd6de;border-radius:6px}
.in input:focus{outline:2px solid #2f7d4f33;border-color:#2f7d4f}
.in .row{display:flex;align-items:center;gap:.5rem;margin-top:.3rem;
 font-size:.76rem;color:#5b6472}
.in label{display:flex;align-items:center;gap:.25rem;cursor:pointer}

/* ---------- rail ---------- */
#rail{position:sticky;top:3.4rem;align-self:start;display:flex;flex-direction:column;gap:.7rem}
.card{background:#fff;border:1px solid #dfe3e8;border-radius:9px;padding:.75rem .85rem}
.card h3{margin:0 0 .45rem;font-size:.76rem;letter-spacing:.05em;text-transform:uppercase;
 color:#7a8593}
.vocab{display:flex;flex-wrap:wrap;gap:.25rem}
.vocab span{background:#eef3f7;border:1px solid #dde4ea;border-radius:10px;
 padding:.08rem .45rem;font-size:.78rem;cursor:pointer}
.vocab span:hover{background:#e2ebf3}
.navlist{display:flex;flex-direction:column;gap:.15rem;max-height:44vh;overflow:auto}
.navlist button{text-align:left;background:none;border:0;border-radius:5px;
 padding:.28rem .4rem;font-size:.8rem;color:#333b47;display:flex;gap:.4rem}
.navlist button:hover{background:#f1f4f7}
.navlist button.cur{background:#eaf3ee;font-weight:600}
.navlist .tick{color:#2f7d4f;font-weight:700;min-width:1.6rem;font-size:.74rem;
 font-variant-numeric:tabular-nums}
.note textarea{width:100%;min-height:4.5rem;padding:.45rem .6rem;
 border:1px solid #cfd6de;border-radius:6px;resize:vertical}

/* ---------- footer nav ---------- */
#foot{position:fixed;bottom:0;left:0;right:0;background:#fff;
 border-top:1px solid #dfe3e8;padding:.55rem 1.2rem;display:flex;gap:.6rem;
 align-items:center;z-index:20}
#foot .spacer{flex:1}
#warn{font-size:.8rem;color:#8a5a00}
"""

JS = r"""
const T = __TASK__;
const KEY = 'topicbench_labels_' + T.task_id;
const esc = s => String(s == null ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const $ = s => document.querySelector(s);

let S = {annotator:{name:'',email:''}, started:null, cur:0, labels:{}, notes:{}};

function load(){
  try{
    const raw = localStorage.getItem(KEY);
    if(raw){ const d = JSON.parse(raw); if(d && d.annotator) S = Object.assign(S, d); }
  }catch(e){ /* corrupt or unavailable storage: start clean rather than fail */ }
}
let saveTimer = null;
function save(){
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try{
      localStorage.setItem(KEY, JSON.stringify(S));
      const t = new Date().toLocaleTimeString();
      $('#saved').textContent = 'saved ' + t;
    }catch(e){ $('#saved').textContent = 'not saved (storage blocked)'; }
  }, 350);
}
const cellKey = (w,i) => w + ':' + i;

/* ---------- progress ---------- */
function counts(){
  let done = 0, total = 0;
  T.papers.forEach(p => p.clusters.forEach(c => {
    total++;
    const v = S.labels[cellKey(p.work_id, c.i)];
    if(v && v.label && v.label.trim()) done++;
  }));
  return {done, total};
}
function paperDone(p){
  return p.clusters.filter(c => {
    const v = S.labels[cellKey(p.work_id, c.i)];
    return v && v.label && v.label.trim();
  }).length;
}
function progress(){
  const {done,total} = counts();
  $('#prog').firstElementChild.style.width = (total ? done/total*100 : 0) + '%';
  $('#progtxt').textContent = done + ' / ' + total + ' clusters';
}

/* ---------- render ---------- */
function renderPaper(){
  const p = T.papers[S.cur];
  const bits = [p.authors, p.year, p.journal].filter(Boolean).map(esc).join(' &middot; ');
  $('#paper').innerHTML =
    `<h2>${esc(p.title)}</h2>
     <div class="meta">${bits}${p.field ? `<span class="fld">${esc(p.field)}</span>`:''}
       ${p.method ? `<span class="fld">${esc(p.method)}</span>`:''}</div>
     ${p.corpus ? `<div class="block"><div class="lbl">Corpus described by the authors</div>
       <div class="corpusbox">${esc(p.corpus)}</div></div>`:''}
     ${p.abstract ? `<div class="block"><div class="lbl">Abstract</div>
       <div class="abs" id="abs">${esc(p.abstract)}</div>
       <button class="absmore" id="absmore">Show more</button></div>`:''}
     <div class="kwnote">Keywords appear exactly as this paper's ${esc(p.method || 'topic model')}
       produced them, including stemmed forms (<i>organis</i>, <i>compani</i>,
       <i>indigen</i>). Read them as the authors did.</div>`;
  const more = $('#absmore');
  if(more) more.onclick = () => {
    const a = $('#abs'); a.classList.toggle('open');
    more.textContent = a.classList.contains('open') ? 'Show less' : 'Show more';
  };
  $('#pnote').value = S.notes[p.work_id] || '';
  $('#pnote').oninput = e => { S.notes[p.work_id] = e.target.value; save(); };
}

function renderClusters(){
  const p = T.papers[S.cur];
  $('#clusters').innerHTML = p.clusters.map(c => {
    const v = S.labels[cellKey(p.work_id, c.i)] || {};
    const done = v.label && v.label.trim() ? ' done' : '';
    return `<div class="cl${done}" data-i="${c.i}">
      <div class="num">${c.i + 1}</div>
      <div class="kws">${c.keywords.map(k => `<span class="kw">${esc(k)}</span>`).join('')}</div>
      <div class="in">
        <input data-i="${c.i}" value="${esc(v.label || '')}"
               placeholder="your label for cluster ${c.i + 1}">
        <div class="row">
          <label><input type="checkbox" class="uns" data-i="${c.i}"
            ${v.unsure ? 'checked':''}> unsure</label>
          <label><input type="checkbox" class="noise" data-i="${c.i}"
            ${v.noise ? 'checked':''}> not interpretable</label>
        </div>
      </div>
    </div>`;
  }).join('');

  $('#clusters').querySelectorAll('input[data-i]:not(.uns):not(.noise)').forEach(el => {
    el.oninput = () => { setCell(p.work_id, +el.dataset.i, {label: el.value}); };
    el.onkeydown = e => {
      if(e.key !== 'Enter') return;
      e.preventDefault();
      const all = [...$('#clusters').querySelectorAll('input[data-i]:not(.uns):not(.noise)')];
      const nx = all[all.indexOf(el) + 1];
      if(nx) nx.focus();
    };
  });
  $('#clusters').querySelectorAll('.uns').forEach(el => {
    el.onchange = () => setCell(p.work_id, +el.dataset.i, {unsure: el.checked});
  });
  $('#clusters').querySelectorAll('.noise').forEach(el => {
    el.onchange = () => setCell(p.work_id, +el.dataset.i, {noise: el.checked});
  });
}

function setCell(w, i, patch){
  if(!S.started) S.started = new Date().toISOString();
  const k = cellKey(w, i);
  S.labels[k] = Object.assign({work_id:w, topic_index:i, label:'', unsure:false, noise:false},
                              S.labels[k] || {}, patch);
  S.labels[k].ts = new Date().toISOString();
  const row = $('#clusters').querySelector(`.cl[data-i="${i}"]`);
  if(row) row.classList.toggle('done', !!(S.labels[k].label || '').trim());
  progress(); renderVocab(); renderNav(); save();
}

/* Labels already used on THIS paper -- supports the inductive pass, where
   consistency across clusters matters more than any single label. */
function renderVocab(){
  const p = T.papers[S.cur];
  const seen = [];
  p.clusters.forEach(c => {
    const v = S.labels[cellKey(p.work_id, c.i)];
    const t = v && (v.label || '').trim();
    if(t && !seen.includes(t)) seen.push(t);
  });
  $('#vocab').innerHTML = seen.length
    ? seen.map(t => `<span title="click to copy">${esc(t)}</span>`).join('')
    : '<span style="background:none;border:0;color:#9aa4b0;padding:0">none yet</span>';
  $('#vocab').querySelectorAll('span[title]').forEach(el => {
    el.onclick = () => { navigator.clipboard && navigator.clipboard.writeText(el.textContent); };
  });
}

function renderNav(){
  $('#nav').innerHTML = T.papers.map((p, ix) => {
    const d = paperDone(p), n = p.clusters.length;
    return `<button data-ix="${ix}" class="${ix === S.cur ? 'cur':''}">
      <span class="tick">${d}/${n}</span>
      <span>${esc(p.title.slice(0, 46))}${p.title.length > 46 ? '&hellip;':''}</span></button>`;
  }).join('');
  $('#nav').querySelectorAll('button').forEach(b => {
    b.onclick = () => go(+b.dataset.ix);
  });
}

function go(ix){
  S.cur = Math.max(0, Math.min(T.papers.length - 1, ix));
  save(); render(); window.scrollTo({top:0});
}
function render(){
  renderPaper(); renderClusters(); renderNav(); renderVocab(); progress();
  const p = T.papers[S.cur];
  const left = p.clusters.length - paperDone(p);
  $('#warn').textContent = left ? left + ' cluster' + (left>1?'s':'') + ' still blank on this paper' : '';
  $('#prev').disabled = S.cur === 0;
  $('#next').disabled = S.cur === T.papers.length - 1;
}

/* ---------- export ---------- */
function exportLabels(){
  const {done,total} = counts();
  if(done < total && !confirm(`${total - done} of ${total} clusters are still blank. Export anyway?`))
    return;
  const rows = [];
  T.papers.forEach(p => p.clusters.forEach(c => {
    const v = S.labels[cellKey(p.work_id, c.i)] || {};
    rows.push({work_id:p.work_id, topic_index:c.i,
               human_label:(v.label || '').trim(),
               unsure:!!v.unsure, not_interpretable:!!v.noise,
               keywords:c.keywords, labeled_at:v.ts || null});
  }));
  const out = {schema:'topicbench.human_labels.v1', task_id:T.task_id,
    annotator:S.annotator, started:S.started, submitted:new Date().toISOString(),
    n_clusters:total, n_labeled:done,
    paper_notes:S.notes, labels:rows};
  const slug = (S.annotator.name || 'annotator').toLowerCase()
    .replace(/[^a-z0-9]+/g,'_').replace(/^_|_$/g,'');
  const blob = new Blob([JSON.stringify(out, null, 1)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `topicbench_labels_${slug}_${T.task_id}.json`;
  a.click(); URL.revokeObjectURL(a.href);
}

/* ---------- start ---------- */
function start(){
  const name = $('#g_name').value.trim(), email = $('#g_email').value.trim();
  if(name.length < 2){ $('#g_err').textContent = 'Please enter your name.'; return; }
  if(!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email)){
    $('#g_err').textContent = 'Please enter a valid email address.'; return; }
  S.annotator = {name, email};
  S.started = S.started || new Date().toISOString();
  $('#gate').classList.add('hide'); $('#shell').classList.remove('hide');
  $('#who').textContent = name + ' \u00b7 ' + email;
  save(); render();
}

load();
$('#g_go').onclick = start;
['g_name','g_email'].forEach(id => {
  $('#' + id).onkeydown = e => { if(e.key === 'Enter') start(); };
});
if(S.annotator && S.annotator.name && S.annotator.email){
  $('#g_name').value = S.annotator.name;
  $('#g_email').value = S.annotator.email;
  $('#g_err').textContent = '';
}
$('#prev').onclick = () => go(S.cur - 1);
$('#next').onclick = () => go(S.cur + 1);
$('#export').onclick = exportLabels;
$('#reset').onclick = () => {
  if(!confirm('Delete all labels you have entered in this browser? This cannot be undone.')) return;
  localStorage.removeItem(KEY);
  S = {annotator:S.annotator, started:new Date().toISOString(), cur:0, labels:{}, notes:{}};
  save(); render();
};
window.addEventListener('beforeunload', e => {
  const {done} = counts();
  if(done > 0 && !S._exported){ e.preventDefault(); e.returnValue = ''; }
});
"""

BODY = """
<div id="gate">
  <h1>TopicBench &mdash; human topic labeling</h1>
  <div class="sub">__NPAPERS__ papers, __NCLUSTERS__ keyword clusters. Your labels are
    recorded independently; the papers' own topic labels are deliberately not shown.</div>

  <label for="g_name">Your name</label>
  <input id="g_name" autocomplete="name" placeholder="Jane Doe">
  <label for="g_email">Your email</label>
  <input id="g_email" type="email" autocomplete="email" placeholder="jane@university.edu">
  <div class="err" id="g_err"></div>
  <button class="primary" id="g_go">Start labeling</button>

  <div class="brief">
    <b>How to label</b>
    <ul>
      <li>Every cluster for a paper is on screen at once. Read them all before writing
        anything &mdash; a label is a choice about how one cluster differs from its
        neighbours.</li>
      <li>Revise freely. Going back to change an earlier label as the set becomes clearer
        is the intended way to work.</li>
      <li>Short noun phrases, in the register you would use in a paper.</li>
      <li>Tick <i>unsure</i> rather than guessing silently, and use the paper note for
        anything the label cannot carry.</li>
      <li>Your work saves in this browser as you type. Export the file at the end and send
        it back.</li>
    </ul>
  </div>
</div>

<div id="shell" class="hide">
  <div id="bar">
    <b>TopicBench labeling</b>
    <span class="who" id="who"></span>
    <span class="spacer"></span>
    <span id="saved"></span>
    <div id="prog"><div></div></div>
    <span id="progtxt"></span>
  </div>

  <div id="app">
    <div>
      <div class="paper" id="paper"></div>
      <div id="clusters"></div>
    </div>
    <div id="rail">
      <div class="card">
        <h3>Papers</h3>
        <div class="navlist" id="nav"></div>
      </div>
      <div class="card">
        <h3>Labels you have used here</h3>
        <div class="vocab" id="vocab"></div>
      </div>
      <div class="card note">
        <h3>Note on this paper</h3>
        <textarea id="pnote" placeholder="Anything the labels cannot carry &mdash; overlapping
clusters, a topic that looks like noise, terminology you were unsure about."></textarea>
      </div>
    </div>
  </div>

  <div id="foot">
    <button class="ghost" id="prev">&larr; Previous paper</button>
    <button class="ghost" id="next">Next paper &rarr;</button>
    <span id="warn"></span>
    <span class="spacer"></span>
    <button class="ghost" id="reset">Clear my work</button>
    <button class="primary" id="export">Export my labels</button>
  </div>
</div>
"""


def write_html(payload, out_path):
    html = ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
            "<title>TopicBench &mdash; human topic labeling</title>\n<style>"
            + CSS + "</style></head><body>\n"
            + BODY.replace("__NPAPERS__", str(payload["n_papers"]))
                  .replace("__NCLUSTERS__", str(payload["n_clusters"]))
            + "\n<script>\n"
            + JS.replace("__TASK__", json.dumps(payload, ensure_ascii=False))
            + "\n</script>\n</body></html>\n")
    open(out_path, "w", encoding="utf-8").write(html)
    return len(html)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    dflt = os.path.join(root, "data")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bibliography", default=os.path.join(dflt, "topicbench_bibliography.csv"))
    p.add_argument("--topics", default=os.path.join(dflt, "topicbench_topics.csv"))
    p.add_argument("--work-ids", help="comma-separated OpenAlex work IDs")
    p.add_argument("--ids-file", help="file with one work_id per line (# comments allowed)")
    p.add_argument("--suggest", type=int, metavar="N",
                   help="propose N papers spread across disciplines")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kmin", type=int, default=5, help="min clusters per paper (--suggest)")
    p.add_argument("--kmax", type=int, default=20, help="max clusters per paper (--suggest)")
    p.add_argument("--hide-abstract", action="store_true",
                   help="omit abstracts; use when label leakage from the abstract is a concern")
    p.add_argument("--out-dir", default=here)
    a = p.parse_args(argv)

    for f in (a.bibliography, a.topics):
        if not os.path.exists(f):
            p.error(f"not found: {f}")
    bib, by_work = load_corpus(a.bibliography, a.topics)
    index = {r["work_id"]: r for r in bib}

    ids = []
    if a.ids_file:
        for line in open(a.ids_file, encoding="utf-8"):
            line = line.split("#")[0].strip()
            if line:
                ids.append(line)
    elif a.work_ids:
        ids = [x.strip() for x in a.work_ids.split(",") if x.strip()]

    if ids:
        missing = [w for w in ids if w not in index]
        if missing:
            p.error("work_id not in the bibliography: " + ", ".join(missing[:5]))
        no_topics = [w for w in ids if not by_work.get(w)]
        if no_topics:
            p.error("no topic rows for: " + ", ".join(no_topics[:5]))
        papers = [index[w] for w in ids]
    elif a.suggest:
        papers = suggest(bib, by_work, a.suggest, a.seed, a.kmin, a.kmax)
        if len(papers) < a.suggest:
            print(f"note: only {len(papers)} papers met the criteria "
                  f"(k in [{a.kmin},{a.kmax}], has corpus description, >=80% strict)",
                  file=sys.stderr)
    else:
        p.error("give --work-ids, --ids-file, or --suggest N")

    payload = build_task(papers, by_work, a.hide_abstract)
    os.makedirs(a.out_dir, exist_ok=True)
    tj = os.path.join(a.out_dir, "task.json")
    th = os.path.join(a.out_dir, "annotate.html")
    json.dump(payload, open(tj, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    n = write_html(payload, th)

    print(f"wrote {tj}")
    print(f"wrote {th}  ({n/1024:.0f} KB, self-contained)")
    print(f"  {payload['n_papers']} papers, {payload['n_clusters']} clusters"
          f"  (task_id {payload['task_id']})")
    byf = collections.Counter(x["field"] for x in payload["papers"])
    print("  by field: " + ", ".join(f"{k} {v}" for k, v in byf.most_common()))
    ks = sorted(len(x["clusters"]) for x in payload["papers"])
    print(f"  clusters per paper: min {ks[0]}, median {ks[len(ks)//2]}, max {ks[-1]}")
    leaks = leakage_report(payload, index)
    if leaks:
        by_src = collections.Counter(s for L in leaks for s in L["visible_in"])
        print(f"  label recoverable for {len(leaks)}/{payload['n_clusters']} clusters "
              f"({len(leaks)/payload['n_clusters']:.1%}): "
              + ", ".join(f"{k} {v}" for k, v in by_src.most_common()))
        json.dump(leaks, open(os.path.join(a.out_dir, "leakage_report.json"), "w"),
                  indent=1, ensure_ascii=False)
        print("  detail in leakage_report.json")
    else:
        print("  no author label is recoverable from what the annotator sees")
    if not a.hide_abstract:
        print("  abstracts included; re-run with --hide-abstract to omit them")
    with open(os.path.join(a.out_dir, "task_papers.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["work_id", "title", "authors", "year", "journal", "field",
                    "n_clusters", "doi"])
        for x in payload["papers"]:
            w.writerow([x["work_id"], x["title"], x["authors"], x["year"], x["journal"],
                        x["field"], len(x["clusters"]), x["doi"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
