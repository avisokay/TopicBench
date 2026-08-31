"""Generate a self-contained TopicBench data-exploration dashboard.

Reads the two corpus CSVs and writes a single HTML file with the data inlined as
JSON and every chart hand-drawn as SVG, so the result works offline and from a
file:// URL with no CDN, no build step, and no JavaScript dependencies.

Standard library only — nothing here needs the packages in pyproject.toml.

Usage
-----
    # from the repo root, using default paths under data/
    python scripts/build_dashboard.py

    # explicit paths
    python scripts/build_dashboard.py --topics data/topicbench_topics.csv \\
        --bibliography data/topicbench_bibliography.csv \\
        --out data/data_dashboard.html

Re-run this after any change to the corpus CSVs; the dashboard embeds a frozen
snapshot rather than reading the CSVs at view time.
"""
import argparse
import csv
import json
import os
import sys

# Truncation limits for the inlined paper metadata. These bound the payload size;
# the full values always remain in the source CSVs.
LIM = {"title": 250, "journal": 90, "authors": 60, "source_table": 110, "quote": 400,
       "abstract": 2800, "corpus": 120}

CSS = """
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 margin:0;color:#1a1a1a;background:#f6f7f9}
header{background:#fff;border-bottom:1px solid #e3e6ea;padding:1.1rem 1.6rem .9rem}
h1{margin:0 0 .15rem;font-size:1.3rem;letter-spacing:-.01em}
.tag{color:#666;font-size:.83rem}
.wrap{max-width:1380px;margin:0 auto;padding:0 1.6rem 3rem}
.kpis{display:flex;gap:.6rem;flex-wrap:wrap;margin:1rem 0 .2rem}
.kpi{flex:1;min-width:104px;background:#fff;border:1px solid #e3e6ea;border-radius:7px;
 padding:.55rem .7rem}
.kpi b{display:block;font-size:1.22rem;color:#1b4d2e;font-variant-numeric:tabular-nums;
 line-height:1.2}
.kpi span{font-size:.71rem;color:#6b7280;text-transform:uppercase;letter-spacing:.03em}
.panel{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:1rem 1.15rem;
 margin:.9rem 0}
.panel h2{margin:0 0 .1rem;font-size:.98rem;color:#1b4d2e}
.panel .hint{color:#6b7280;font-size:.79rem;margin:0 0 .8rem}
.filters{display:flex;gap:.9rem;flex-wrap:wrap;align-items:flex-end}
.fg{display:flex;flex-direction:column;gap:.22rem}
.fg label{font-size:.71rem;color:#6b7280;text-transform:uppercase;letter-spacing:.03em}
select,input[type=text]{font:13px inherit;padding:.32rem .45rem;border:1px solid #d2d6dc;
 border-radius:5px;background:#fff;min-width:150px}
input[type=text]{min-width:230px}
button{font:12px inherit;padding:.34rem .7rem;border:1px solid #d2d6dc;background:#fff;
 border-radius:5px;cursor:pointer}
button:hover{background:#f3f4f6}
button.on{background:#1b4d2e;color:#fff;border-color:#1b4d2e}
.chips{display:flex;gap:.32rem;flex-wrap:wrap;margin:.15rem 0 0}
.chip{font-size:.73rem;padding:.16rem .5rem;border:1px solid #d2d6dc;border-radius:11px;
 cursor:pointer;background:#fff;white-space:nowrap}
.chip.on{background:#1b4d2e;color:#fff;border-color:#1b4d2e}
.chip .c{opacity:.6;margin-left:.25rem;font-variant-numeric:tabular-nums}
.tabs{display:flex;gap:.3rem;margin:1rem 0 0;border-bottom:1px solid #e3e6ea}
.tabs button{border:none;border-bottom:2px solid transparent;background:none;
 padding:.5rem .85rem;font-size:.87rem;color:#6b7280;border-radius:0}
.tabs button.on{color:#1b4d2e;border-bottom-color:#2f7d4f;font-weight:600;background:none}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem}
@media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:.82rem}
th{text-align:left;border-bottom:1.5px solid #9ca3af;padding:.36rem .45rem;
 position:sticky;top:0;background:#fff;cursor:pointer;white-space:nowrap;font-weight:600}
th:hover{color:#1b4d2e}
td{border-bottom:1px solid #eef0f2;padding:.32rem .45rem;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.scroll{max-height:560px;overflow:auto;border:1px solid #eef0f2;border-radius:6px}
.tcard{border:1px solid #e8eaed;border-left:3px solid #2f7d4f;border-radius:6px;
 padding:.55rem .7rem;margin:.42rem 0;background:#fff}
.tcard.core{border-left-color:#5b8ac4}
.tcard.sensitivity{border-left-color:#dd8a3f}
.tcard.excluded{border-left-color:#c0504d;opacity:.72}
.tl{font-weight:600;font-size:.92rem}
.tm{font-size:.73rem;color:#6b7280;margin:.1rem 0 .3rem}
.kw{display:inline-block;font-size:.75rem;background:#f1f3f5;border-radius:3px;
 padding:.07rem .34rem;margin:.08rem .14rem .08rem 0;font-family:ui-monospace,monospace}
.kw.hi{background:#fde9c8;font-weight:600}
.fl{display:inline-block;font-size:.66rem;background:#fff3e6;color:#8a4b00;
 border:1px solid #f0d3ae;border-radius:9px;padding:.02rem .38rem;margin-right:.2rem}
.bar{height:9px;background:#5b8ac4;border-radius:1px;display:inline-block;
 vertical-align:middle;margin-right:5px}
svg{display:block;overflow:visible}
.lg{font-size:.73rem;color:#4b5563;display:flex;gap:.9rem;flex-wrap:wrap;margin-top:.4rem}
.lg i{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:.25rem}
.empty{color:#9ca3af;font-size:.85rem;padding:1.4rem;text-align:center}
.phead{border:1px solid #d7e3da;background:#f7fbf8;border-radius:7px;
 padding:.7rem .85rem;margin:0 0 .8rem}
.phead table td{border:none;padding:.11rem 0;vertical-align:top}
.phead summary::marker{color:#2f7d4f}
.note{font-size:.76rem;color:#6b7280;background:#f9fafb;border-left:2px solid #d2d6dc;
 padding:.45rem .6rem;margin:.6rem 0 0}
a{color:#1b4d2e}
.pg{display:flex;gap:.4rem;align-items:center;margin-top:.6rem;font-size:.8rem;color:#6b7280}
"""

JS = r"""
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const T=DATA.topics, P=DATA.papers;
const TIERS=['strict','core','sensitivity','excluded'];
const SRC={jats_xml:'#4878a8',pdf_text_layer:'#dd8a3f',vision:'#2f7d4f',vision_local:'#8fc7a5'};
const SRCN={jats_xml:'JATS XML',pdf_text_layer:'PDF text layer',vision:'PDF vision',
            vision_local:'PDF vision (curated)'};
const uniq=(a)=>[...new Set(a)].sort();
const FIELDS=uniq(T.map(t=>t.f)), FLAGS=uniq(T.flatMap(t=>t.g));
const YRS=T.map(t=>t.y).filter(y=>y>0);
const S={field:'',tier:'',src:'',flags:new Set(),q:'',ymin:Math.min(...YRS),
         ymax:Math.max(...YRS),tab:'overview',sort:'-nk',page:0,flagMode:'any',
         work:'',psort:'-n'};

function pass(t){
  if(S.work&&t.w!==S.work)return false;
  if(S.field&&t.f!==S.field)return false;
  if(S.tier&&t.e!==S.tier)return false;
  if(S.src&&t.s!==S.src)return false;
  if(t.y&&(t.y<S.ymin||t.y>S.ymax))return false;
  if(S.flags.size){
    const hit=[...S.flags].filter(f=>t.g.includes(f)).length;
    if(S.flagMode==='any'&&hit===0)return false;
    if(S.flagMode==='all'&&hit!==S.flags.size)return false;
    if(S.flagMode==='none'&&hit>0)return false;
  }
  if(S.q){
    const q=S.q.toLowerCase();
    const hay=(t.L+' '+t.K.join(' ')+' '+(P[t.w]?P[t.w].t:'')).toLowerCase();
    if(!hay.includes(q))return false;
  }
  return true;
}
let F=[];
const med=a=>{if(!a.length)return 0;const b=[...a].sort((x,y)=>x-y);const m=b.length>>1;
  return b.length%2?b[m]:(b[m-1]+b[m])/2};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* ---------- SVG chart helpers (no libraries) ---------- */
function histogram(el,vals,{bins=null,w=380,h=140,color='#5b8ac4',xlab='',fmt=v=>v}={}){
  if(!vals.length){el.innerHTML='<div class=empty>no rows match</div>';return}
  let edges;
  if(bins) edges=bins;
  else{const mx=Math.max(...vals);edges=Array.from({length:Math.min(mx,14)+1},(_,i)=>i)}
  const cnt=edges.slice(0,-1).map(()=>0);
  vals.forEach(v=>{for(let i=edges.length-2;i>=0;i--){
    if(v>=edges[i]&&(i===edges.length-2?v<=edges[i+1]:v<edges[i+1])){cnt[i]++;break}}});
  const mx=Math.max(...cnt,1), pl=34,pb=28,pt=8,pr=6;
  const bw=(w-pl-pr)/cnt.length;
  let s=`<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">`;
  for(let g=0;g<=3;g++){const y=pt+(h-pt-pb)*g/3;const v=Math.round(mx*(1-g/3));
    s+=`<line x1="${pl}" x2="${w-pr}" y1="${y}" y2="${y}" stroke="#eef0f2"/>`;
    s+=`<text x="${pl-5}" y="${y+3}" font-size="9" fill="#9ca3af" text-anchor="end">${v}</text>`}
  cnt.forEach((c,i)=>{const bh=(h-pt-pb)*c/mx;
    s+=`<rect x="${pl+i*bw+1}" y="${h-pb-bh}" width="${Math.max(bw-2,1)}" height="${bh}"
        fill="${color}" rx="1"><title>${fmt(edges[i])}: ${c} topics</title></rect>`;
    if(cnt.length<=14||i%2===0)
      s+=`<text x="${pl+i*bw+bw/2}" y="${h-pb+11}" font-size="9" fill="#6b7280"
          text-anchor="middle">${fmt(edges[i])}</text>`});
  if(xlab)s+=`<text x="${pl+(w-pl-pr)/2}" y="${h-2}" font-size="9.5" fill="#4b5563"
      text-anchor="middle">${xlab}</text>`;
  el.innerHTML=s+'</svg>';
}
function stackedBars(el,cats,keyOf,colors,names,{w=560,rowH=17}={}){
  if(!cats.length){el.innerHTML='<div class=empty>no rows match</div>';return}
  const keys=Object.keys(colors);
  const tot=cats.map(c=>keys.reduce((a,k)=>a+(c.v[k]||0),0));
  const mx=Math.max(...tot,1), pl=168, pr=44, h=cats.length*rowH+14;
  let s=`<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">`;
  cats.forEach((c,i)=>{
    const y=i*rowH+4; let x=pl;
    s+=`<text x="${pl-6}" y="${y+11}" font-size="10.5" fill="#374151"
        text-anchor="end">${esc(c.k)}</text>`;
    keys.forEach(k=>{const v=c.v[k]||0; if(!v)return;
      const bw=(w-pl-pr)*v/mx;
      s+=`<rect x="${x}" y="${y+2}" width="${bw}" height="${rowH-6}" fill="${colors[k]}">
          <title>${esc(c.k)} — ${names[k]}: ${v}</title></rect>`; x+=bw});
    s+=`<text x="${x+5}" y="${y+11}" font-size="10" fill="#4b5563"
        font-variant-numeric="tabular-nums">${tot[i]}</text>`});
  el.innerHTML=s+'</svg>';
}
function scatter(el,pts,{w=560,h=300,xlab='',ylab=''}={}){
  if(!pts.length){el.innerHTML='<div class=empty>no rows match</div>';return}
  const pl=40,pb=32,pt=10,pr=12;
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  const x0=0,x1=Math.max(...xs)*1.05||1, y0=0,y1=Math.max(...ys)*1.08||1;
  const X=v=>pl+(w-pl-pr)*(v-x0)/(x1-x0), Y=v=>h-pb-(h-pt-pb)*(v-y0)/(y1-y0);
  let s=`<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}">`;
  for(let g=0;g<=4;g++){const y=pt+(h-pt-pb)*g/4, v=(y1*(1-g/4));
    s+=`<line x1="${pl}" x2="${w-pr}" y1="${y}" y2="${y}" stroke="#eef0f2"/>`;
    s+=`<text x="${pl-5}" y="${y+3}" font-size="9" fill="#9ca3af"
        text-anchor="end">${v.toFixed(0)}</text>`}
  for(let g=0;g<=4;g++){const x=pl+(w-pl-pr)*g/4;
    s+=`<text x="${x}" y="${h-pb+12}" font-size="9" fill="#9ca3af"
        text-anchor="middle">${(x1*g/4).toFixed(0)}</text>`}
  pts.forEach(p=>{s+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3.1" fill="${p.c}"
      fill-opacity=".62" stroke="${p.c}" stroke-width=".5"><title>${esc(p.t)}</title></circle>`});
  s+=`<text x="${pl+(w-pl-pr)/2}" y="${h-3}" font-size="9.5" fill="#4b5563"
      text-anchor="middle">${xlab}</text>`;
  s+=`<text x="10" y="${pt+(h-pt-pb)/2}" font-size="9.5" fill="#4b5563"
      text-anchor="middle" transform="rotate(-90 10 ${pt+(h-pt-pb)/2})">${ylab}</text>`;
  el.innerHTML=s+'</svg>';
}
const legend=(colors,names)=>'<div class=lg>'+Object.keys(colors)
  .map(k=>`<span><i style="background:${colors[k]}"></i>${names[k]}</span>`).join('')+'</div>';

/* ---------- KPIs ---------- */
function kpis(){
  const nP=new Set(F.map(t=>t.w)).size;
  const wl=F.map(t=>t.n), kn=F.map(t=>t.nk);
  const flagged=F.filter(t=>t.g.length).length;
  $('#kpis').innerHTML=[
    ['topics',F.length.toLocaleString()],['papers',nP],
    ['disciplines',new Set(F.map(t=>t.f)).size],
    ['median label words',med(wl)],['median keywords',med(kn)],
    ['carry a flag',F.length?Math.round(100*flagged/F.length)+'%':'—'],
  ].map(([l,v])=>`<div class=kpi><b>${v}</b><span>${l}</span></div>`).join('');
  $('#shown').textContent=`${F.length.toLocaleString()} of ${T.length.toLocaleString()} topics`;
}

/* ---------- tabs ---------- */
function overview(){
  histogram($('#h_len'),F.map(t=>t.n),{bins:[1,2,3,4,5,6,7,8,24],
    xlab:'words in author label',color:'#2f7d4f',
    fmt:v=>v===8?'8+':v});
  histogram($('#h_kw'),F.map(t=>t.nk),{bins:[0,5,10,15,20,25,30,50],
    xlab:'keywords in cluster',color:'#5b8ac4',fmt:v=>v});
  const byP={};F.forEach(t=>byP[t.w]=t.k);
  histogram($('#h_k'),Object.values(byP),{bins:[0,5,10,15,20,25,30,40,70],
    xlab:'topics (k) in the source model',color:'#dd8a3f',fmt:v=>v});
  const fc={};F.forEach(t=>{(fc[t.f]=fc[t.f]||{})[t.s]=((fc[t.f]||{})[t.s]||0)+1});
  const cats=Object.keys(fc).map(k=>({k,v:fc[k]}))
    .sort((a,b)=>Object.values(b.v).reduce((x,y)=>x+y,0)-Object.values(a.v).reduce((x,y)=>x+y,0));
  stackedBars($('#b_field'),cats,null,SRC,SRCN);
  $('#lg_field').innerHTML=legend(SRC,SRCN);
  const tc={};F.forEach(t=>{(tc[t.f]=tc[t.f]||{})[t.e]=((tc[t.f]||{})[t.e]||0)+1});
  const TC={strict:'#2f7d4f',core:'#5b8ac4',sensitivity:'#dd8a3f',excluded:'#c0504d'};
  const tcats=Object.keys(tc).map(k=>({k,v:tc[k]}))
    .sort((a,b)=>(b.v.strict||0)-(a.v.strict||0));
  stackedBars($('#b_tier'),tcats,null,TC,
    {strict:'strict',core:'core',sensitivity:'sensitivity',excluded:'excluded'});
  $('#lg_tier').innerHTML=legend(TC,{strict:'strict',core:'core',
    sensitivity:'sensitivity',excluded:'excluded'});
  const pw={};F.forEach(t=>{pw[t.w]=pw[t.w]||{n:0,k:t.k,f:t.f};pw[t.w].n++});
  const FC={};FIELDS.forEach((f,i)=>FC[f]=['#4878a8','#2f7d4f','#dd8a3f','#8b5fa8','#c0504d',
    '#1b4d2e','#5b8ac4','#8fc7a5','#b07aa1','#e15759','#76b7b2','#59a14f','#edc948',
    '#af7aa1','#ff9da7','#9c755f','#bab0ac','#4e79a7'][i%18]);
  scatter($('#s_k'),Object.entries(pw).map(([w,v])=>({x:v.k,y:v.n,c:FC[v.f],
    t:`${P[w]?P[w].t:w}\n${v.n} labeled of k=${v.k} — ${v.f}`})),
    {xlab:'k topics in the model',ylab:'labeled topics extracted'});
}
function flagsTab(){
  const n=F.length||1;
  const counts=FLAGS.map(f=>({f,c:F.filter(t=>t.g.includes(f)).length}))
    .filter(x=>x.c).sort((a,b)=>b.c-a.c);
  $('#flag_tbl').innerHTML=counts.length?`<table><thead><tr><th>flag</th><th class=n>topics</th>
    <th class=n>share</th><th>prevalence</th></tr></thead><tbody>`+
    counts.map(x=>`<tr><td><code>${x.f}</code></td><td class=n>${x.c}</td>
      <td class=n>${(100*x.c/n).toFixed(1)}%</td>
      <td><span class=bar style="width:${Math.max(120*x.c/counts[0].c,1)}px"></span></td></tr>`)
    .join('')+'</tbody></table>':'<div class=empty>no flagged rows in this selection</div>';
  const top=counts.slice(0,8).map(x=>x.f);
  if(top.length<2){$('#cooc').innerHTML='<div class=empty>need 2+ flags present</div>';return}
  let s='<table><thead><tr><th></th>'+top.map(f=>
    `<th class=n style="writing-mode:vertical-rl;font-weight:500;font-size:.68rem">${f}</th>`)
    .join('')+'</tr></thead><tbody>';
  top.forEach(a=>{s+=`<tr><td style="font-size:.72rem"><code>${a}</code></td>`;
    top.forEach(b=>{const v=F.filter(t=>t.g.includes(a)&&t.g.includes(b)).length;
      const mx=F.filter(t=>t.g.includes(a)).length||1;
      const op=a===b?0.13:Math.min(v/mx,1)*0.85;
      s+=`<td class=n style="background:rgba(47,125,79,${op})">${v||''}</td>`});
    s+='</tr>'});
  $('#cooc').innerHTML=s+'</tbody></table>';
}
function paperHeader(){
  const el=$('#phead');
  if(!S.work){el.innerHTML='';el.style.display='none';return}
  const p=P[S.work]||{}, mine=T.filter(t=>t.w===S.work);
  const shown=F.length, tiers={};
  mine.forEach(t=>tiers[t.e]=(tiers[t.e]||0)+1);
  const row=(k,v)=>v?`<tr><td style="color:#6b7280;white-space:nowrap;width:1%;
    padding-right:.8rem">${k}</td><td>${v}</td></tr>`:'';
  const dl=p.doi?`<a href="https://doi.org/${p.doi}" target=_blank rel=noopener>${p.doi}</a>`
    :(p.link?`<a href="${esc(p.link)}" target=_blank rel=noopener>source</a>`:'');
  el.style.display='';
  el.innerHTML=`
   <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem">
    <div style="font-size:1.02rem;font-weight:600;line-height:1.35;max-width:78%">
      ${esc(p.t||S.work)}</div>
    <button id="clearwork">&times; all papers</button></div>
   <table style="margin:.55rem 0 0;font-size:.83rem">
    ${row('authors', esc(p.a||''))}
    ${row('journal', esc(p.j||'')+(p.y?` &middot; ${p.y}`:'')+(p.cb?` &middot; ${p.cb} citations`:''))}
    ${row('discipline', esc(p.f||mine[0]&&mine[0].f||''))}
    ${row('method', p.me?`<code>${esc(p.me)}</code>`:'<span style="color:#9ca3af">not recorded</span>')}
    ${row('corpus', p.co?esc(p.co):'<span style="color:#9ca3af">not recorded in the dataset</span>')}
    ${row('identifier', dl+` <span style="color:#9ca3af">&middot; ${S.work}</span>`)}
    ${row('extracted from', esc(p.st||''))}
    ${row('topics', `${mine.length} labeled` +
       (shown!==mine.length?` &middot; <b>${shown}</b> match the current filters`:'') +
       ' &middot; ' + Object.keys(tiers).map(k=>`${tiers[k]} ${k}`).join(', '))}
    ${row('labeling process', p.lp==='yes'
       ? (p.lpq?`&ldquo;${esc(p.lpq)}&rdquo;`:'described in the paper')
       : p.lp==='no'?'<span style="color:#9ca3af">not described</span>':'')}
   </table>
   ${p.ab?`<details style="margin-top:.6rem"><summary style="cursor:pointer;
     font-size:.83rem;color:#1b4d2e">abstract</summary>
     <div style="font-size:.83rem;color:#374151;margin-top:.35rem;max-width:78ch">
     ${esc(p.ab)}</div></details>`
    :'<div style="font-size:.8rem;color:#9ca3af;margin-top:.5rem">no abstract on record</div>'}`;
  $('#clearwork').onclick=()=>{S.work='';S.page=0;render()};
}

function browse(){
  paperHeader();
  const sorters={'-nk':(a,b)=>b.nk-a.nk,'nk':(a,b)=>a.nk-b.nk,
    '-n':(a,b)=>b.n-a.n,'n':(a,b)=>a.n-b.n,
    'L':(a,b)=>a.L.localeCompare(b.L),'-y':(a,b)=>b.y-a.y,'y':(a,b)=>a.y-b.y,
    'w':(a,b)=>a.w.localeCompare(b.w)||a.i-b.i};
  const rows=[...F].sort(sorters[S.sort]||sorters['-nk']);
  const PER=60, pages=Math.max(Math.ceil(rows.length/PER),1);
  S.page=Math.min(S.page,pages-1);
  const slice=rows.slice(S.page*PER,(S.page+1)*PER);
  const q=S.q.toLowerCase();
  $('#cards').innerHTML=slice.length?slice.map(t=>{
    const p=P[t.w]||{};
    const kws=t.K.map(k=>`<span class="kw${q&&k.toLowerCase().includes(q)?' hi':''}">${esc(k)}</span>`).join('');
    // With one paper selected the header already carries authors/journal/field.
    const meta=S.work
      ? `topic ${t.i+1} of ${t.k} · <span style="color:#9ca3af">${t.u}</span>`
      : `${esc(p.a||'')} · ${esc(p.j||'')} · ${t.f} · topic ${t.i+1} of ${t.k} ·
         <span style="color:#9ca3af">${t.u}</span>`;
    return `<div class="tcard ${t.e}">
      <div class=tl>${esc(t.L)||'<i style="color:#9ca3af">(no label)</i>'}</div>
      <div class=tm>${meta}</div>
      <div>${kws}</div>
      ${t.Lr?`<div class=tm style="margin-top:.3rem">raw label in source:
        <code>${esc(t.Lr)}</code></div>`:''}
      ${t.g.length?`<div style="margin-top:.35rem">${t.g.map(f=>`<span class=fl>${f}</span>`).join('')}</div>`:''}
    </div>`}).join(''):'<div class=empty>No topics match these filters.</div>';
  $('#pg').innerHTML=rows.length?`<button ${S.page===0?'disabled':''} id=pv>&larr; prev</button>
    <span>page ${S.page+1} of ${pages} — showing ${slice.length} of ${rows.length.toLocaleString()}</span>
    <button ${S.page>=pages-1?'disabled':''} id=nx>next &rarr;</button>`:'';
  if($('#pv'))$('#pv').onclick=()=>{S.page--;browse()};
  if($('#nx'))$('#nx').onclick=()=>{S.page++;browse()};
}
const PCOLS=[
  {k:'t',   h:'paper',    cls:'',  num:false},
  {k:'f',   h:'field',    cls:'',  num:false},
  {k:'y',   h:'year',     cls:'n', num:true},
  {k:'cb',  h:'cites',    cls:'n', num:true},
  {k:'n',   h:'n_topics', cls:'n', num:true},
  {k:'src', h:'route',    cls:'',  num:false},
  {k:'br',  h:'browse',   cls:'',  num:false},
];
function papers(){
  const agg={};
  F.forEach(t=>{const a=agg[t.w]=agg[t.w]||{n:0,f:t.f,src:t.s,tiers:{}};
    a.n++;a.tiers[t.e]=(a.tiers[t.e]||0)+1});
  let rows=Object.entries(agg).map(([w,a])=>{const p=P[w]||{};
    return {w,n:a.n,f:a.f,src:a.src,tiers:a.tiers,p,
            t:(p.t||w),y:+(p.y||0),cb:+(p.cb||0)}});
  const desc=S.psort.startsWith('-'), key=S.psort.replace(/^-/,'');
  const col=PCOLS.find(c=>c.k===key)||PCOLS[4];
  rows.sort((a,b)=>{
    let d;
    if(col.num) d=(a[key]||0)-(b[key]||0);
    else d=String(a[key]===undefined?'':a[key]).localeCompare(String(b[key]===undefined?'':b[key]));
    if(d===0) d=b.n-a.n;                       // stable secondary: topic count
    return desc?-d:d});
  const arrow=k=>S.psort===k?' \u25b2':S.psort==='-'+k?' \u25bc':'';
  const head=PCOLS.map(c=>c.k==='br'
    ? `<th style="cursor:default">${c.h}</th>`
    : `<th class="${c.cls}" data-k="${c.k}">${c.h}${arrow(c.k)}</th>`).join('');
  $('#paper_tbl').innerHTML=rows.length?`<div class=scroll><table><thead><tr>${head}
    </tr></thead><tbody>`+rows.map(r=>`<tr>
      <td style="max-width:430px">${r.p.doi?`<a href="https://doi.org/${r.p.doi}"
        target=_blank rel=noopener>${esc(r.t)}</a>`:esc(r.t)}
        <div style="color:#9ca3af;font-size:.72rem">${esc(r.p.a||'')}${
          r.p.j?' &middot; '+esc(r.p.j):''}</div></td>
      <td>${r.f}</td><td class=n>${r.y||''}</td><td class=n>${r.cb||''}</td>
      <td class=n>${r.n}</td>
      <td><span style="color:${SRC[r.src]}">&#9632;</span> ${SRCN[r.src]||r.src}</td>
      <td><a href="#" class="tolink" data-w="${r.w}"
        title="open this paper's topics in the browser">${r.n} topics &rarr;</a></td>
      </tr>`).join('')+'</tbody></table></div>':'<div class=empty>no papers match</div>';
  $$('#paper_tbl th[data-k]').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k, c=PCOLS.find(x=>x.k===k);
    // numeric columns open on descending; text columns open on ascending
    S.psort = S.psort===k ? '-'+k : S.psort==='-'+k ? k : (c.num?'-'+k:k);
    papers()});
  $$('#paper_tbl .tolink').forEach(a=>a.onclick=e=>{
    e.preventDefault();
    S.work=a.dataset.w; S.page=0; S.tab='browse';
    $$('.tabs button').forEach(x=>x.classList.toggle('on',x.dataset.t==='browse'));
    $$('.tabpane').forEach(p=>p.style.display=p.id==='tab_browse'?'':'none');
    render(); window.scrollTo({top:0,behavior:'smooth'})});
}
function render(){
  F=T.filter(pass); kpis();
  ({overview,browse,papers,flags:flagsTab}[S.tab]||overview)();
}

/* ---------- controls ---------- */
function buildControls(){
  const opt=(v,l,c)=>`<option value="${v}">${l}${c!=null?` (${c})`:''}</option>`;
  const fc={},sc={},tc={};
  T.forEach(t=>{fc[t.f]=(fc[t.f]||0)+1;sc[t.s]=(sc[t.s]||0)+1;tc[t.e]=(tc[t.e]||0)+1});
  $('#f_field').innerHTML=opt('','all disciplines',T.length)+
    FIELDS.map(f=>opt(f,f,fc[f])).join('');
  $('#f_tier').innerHTML=opt('','all tiers',T.length)+
    TIERS.filter(t=>tc[t]).map(t=>opt(t,t,tc[t])).join('');
  $('#f_src').innerHTML=opt('','all routes',T.length)+
    Object.keys(SRC).filter(s=>sc[s]).map(s=>opt(s,SRCN[s],sc[s])).join('');
  $('#f_ymin').value=S.ymin;$('#f_ymax').value=S.ymax;
  const gc={};FLAGS.forEach(f=>gc[f]=T.filter(t=>t.g.includes(f)).length);
  $('#f_flags').innerHTML=FLAGS.map(f=>
    `<span class=chip data-f="${f}">${f}<span class=c>${gc[f]}</span></span>`).join('');
  $$('#f_flags .chip').forEach(c=>c.onclick=()=>{
    const f=c.dataset.f;
    if(S.flags.has(f))S.flags.delete(f);else S.flags.add(f);
    c.classList.toggle('on');S.page=0;render()});
  $('#f_field').onchange=e=>{S.field=e.target.value;S.page=0;render()};
  $('#f_tier').onchange=e=>{S.tier=e.target.value;S.page=0;render()};
  $('#f_src').onchange=e=>{S.src=e.target.value;S.page=0;render()};
  $('#f_ymin').onchange=e=>{S.ymin=+e.target.value||S.ymin;S.page=0;render()};
  $('#f_ymax').onchange=e=>{S.ymax=+e.target.value||S.ymax;S.page=0;render()};
  $('#f_sort').onchange=e=>{S.sort=e.target.value;S.page=0;browse()};
  let deb;
  $('#f_q').oninput=e=>{clearTimeout(deb);deb=setTimeout(()=>{
    S.q=e.target.value.trim();S.page=0;render()},180)};
  $$('#f_mode button').forEach(b=>b.onclick=()=>{
    $$('#f_mode button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');S.flagMode=b.dataset.m;S.page=0;render()});
  $('#reset').onclick=()=>{
    S.field=S.tier=S.src=S.q=S.work='';S.flags.clear();S.flagMode='any';
    S.ymin=Math.min(...YRS);S.ymax=Math.max(...YRS);S.page=0;
    $('#f_field').value='';$('#f_tier').value='';$('#f_src').value='';$('#f_q').value='';
    $('#f_ymin').value=S.ymin;$('#f_ymax').value=S.ymax;
    $$('#f_flags .chip').forEach(c=>c.classList.remove('on'));
    $$('#f_mode button').forEach((x,i)=>x.classList.toggle('on',i===0));
    render()};
  $$('.tabs button').forEach(b=>b.onclick=()=>{
    $$('.tabs button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');S.tab=b.dataset.t;
    $$('.tabpane').forEach(p=>p.style.display=p.id==='tab_'+S.tab?'':'none');
    render()});
}
buildControls();render();
"""


def load_payload(topics_csv, bib_csv, generated=None):
    """Build the inlined JSON payload from the two corpus CSVs.

    `topics_csv` is one row per topic (the analysis unit); `bib_csv` is one row
    per paper and supplies the metadata shown in the Papers tab and on each card.
    """
    with open(topics_csv, newline="", encoding="utf-8") as fh:
        topics = list(csv.DictReader(fh))
    with open(bib_csv, newline="", encoding="utf-8") as fh:
        bib = list(csv.DictReader(fh))
    if not topics:
        raise SystemExit(f"no rows in {topics_csv}")

    required = {"topic_uid", "work_id", "field", "year", "topic_index", "k_paper",
                "author_label", "author_label_raw", "label_n_words", "keywords",
                "n_keywords", "extraction_source", "eval_tier", "flags"}
    missing = required - set(topics[0])
    if missing:
        raise SystemExit(f"{topics_csv} is missing columns: {sorted(missing)}")

    papers = {}
    for r in bib:
        papers[r["work_id"]] = {
            "t": r.get("title", "")[:LIM["title"]],
            "j": (r.get("journal") or "")[:LIM["journal"]],
            "y": r.get("year", ""),
            "a": (r.get("authors") or "")[:LIM["authors"]],
            "doi": r.get("doi", ""),
            "cb": r.get("cited_by", ""),
            "src": r.get("extraction_source", ""),
            "st": (r.get("source_table") or "")[:LIM["source_table"]],
            "lp": r.get("authors detail their labeling process", ""),
            "lpq": (r.get("labeling_process_quote") or "")[:LIM["quote"]],
            "ab": (r.get("abstract") or "")[:LIM["abstract"]],
            "co": (r.get("corpus") or "")[:LIM["corpus"]],
            "me": (r.get("method") or ""),
            "link": (r.get("link") or ""),
        }

    T = []
    for r in topics:
        raw = r["author_label_raw"]
        T.append({
            "u": r["topic_uid"], "w": r["work_id"], "f": r["field"],
            "y": int(r["year"] or 0), "i": int(r["topic_index"]),
            "k": int(r["k_paper"]), "L": r["author_label"],
            "Lr": raw if raw != r["author_label"] else "",
            "n": int(r["label_n_words"]), "K": json.loads(r["keywords"]),
            "nk": int(r["n_keywords"]), "s": r["extraction_source"],
            "e": r["eval_tier"],
            "g": [x for x in r["flags"].split("|") if x],
        })

    orphans = {t["w"] for t in T} - set(papers)
    if orphans:
        print(f"warning: {len(orphans)} topic rows reference a work_id absent from "
              f"{os.path.basename(bib_csv)}; their cards will show no metadata",
              file=sys.stderr)

    return {"topics": T, "papers": papers,
            "meta": {"n_topics": len(T), "n_papers": len({t["w"] for t in T}),
                     "n_bib": len(bib), "generated": generated or ""}}


def build(payload, out="data_dashboard.html"):
    js = "const DATA=" + json.dumps(payload, ensure_ascii=False,
                                    separators=(",", ":")) + ";\n" + JS
    m = payload["meta"]
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TopicBench data explorer</title><style>{CSS}</style></head><body>
<header><div class="wrap" style="padding:0">
<h1>TopicBench data explorer</h1>
<div class="tag">{m['n_topics']:,} author-labeled topics from {m['n_papers']} papers across
18 social-science disciplines &mdash; one row per topic, the unit of the label-comparison
analysis. Self-contained: filters and charts run locally, no network needed.</div>
</div></header>
<div class="wrap">

<div class="panel">
 <h2>Filters</h2>
 <p class="hint">Every view below reflects the current selection.
   <b id="shown"></b> selected.</p>
 <div class="filters">
  <div class="fg"><label>discipline</label><select id="f_field"></select></div>
  <div class="fg"><label>evaluation tier</label><select id="f_tier"></select></div>
  <div class="fg"><label>extraction route</label><select id="f_src"></select></div>
  <div class="fg"><label>year from</label><input type="text" id="f_ymin"
      style="min-width:74px"></div>
  <div class="fg"><label>to</label><input type="text" id="f_ymax"
      style="min-width:74px"></div>
  <div class="fg"><label>search labels, keywords, titles</label>
      <input type="text" id="f_q" placeholder="e.g. climate, trust, vaccine"></div>
  <div class="fg"><label>&nbsp;</label><button id="reset">reset all</button></div>
 </div>
 <div style="margin-top:.75rem">
  <label style="font-size:.71rem;color:#6b7280;text-transform:uppercase;
    letter-spacing:.03em">quality flags</label>
  <span id="f_mode" style="margin-left:.5rem">
    <button class="on" data-m="any">match any</button>
    <button data-m="all">match all</button>
    <button data-m="none">exclude</button></span>
  <div class="chips" id="f_flags" style="margin-top:.4rem"></div>
 </div>
</div>

<div class="kpis" id="kpis"></div>

<div class="tabs">
 <button class="on" data-t="overview">Distributions</button>
 <button data-t="browse">Topic browser</button>
 <button data-t="papers">Papers</button>
 <button data-t="flags">Quality flags</button>
</div>

<div class="tabpane" id="tab_overview">
 <div class="grid3">
  <div class="panel"><h2>Label length</h2>
    <p class="hint">Words per author label</p><div id="h_len"></div></div>
  <div class="panel"><h2>Cluster size</h2>
    <p class="hint">Keywords published per topic</p><div id="h_kw"></div></div>
  <div class="panel"><h2>Model size</h2>
    <p class="hint">k topics in the source model, per paper</p><div id="h_k"></div></div>
 </div>
 <div class="grid2">
  <div class="panel"><h2>Topics by discipline and extraction route</h2>
    <p class="hint">Which route recovered each discipline's topics</p>
    <div id="b_field"></div><div id="lg_field"></div></div>
  <div class="panel"><h2>Topics by discipline and tier</h2>
    <p class="hint">Where the usable rows sit</p>
    <div id="b_tier"></div><div id="lg_tier"></div></div>
 </div>
 <div class="panel"><h2>Extraction completeness</h2>
  <p class="hint">Labeled topics recovered against the model's k. Points on the diagonal
    are papers whose whole topic table was captured; points below it lost topics to
    partial labeling or extraction. Hover for the paper.</p>
  <div id="s_k"></div></div>
</div>

<div class="tabpane" id="tab_browse" style="display:none">
 <div class="panel">
  <h2>Topic browser</h2>
  <p class="hint">The dataset itself: each card is one author label and the keyword cluster
    it names. Search terms are highlighted in the keywords. Open a single paper from the
    Papers tab to see its full record above its topics.</p>
  <div id="phead" class="phead" style="display:none"></div>
  <div class="fg" style="margin-bottom:.5rem"><label>sort by</label>
   <select id="f_sort">
    <option value="-nk">most keywords</option><option value="nk">fewest keywords</option>
    <option value="-n">longest label</option><option value="n">shortest label</option>
    <option value="L">label A&ndash;Z</option>
    <option value="-y">newest</option><option value="y">oldest</option>
    <option value="w">grouped by paper</option></select></div>
  <div id="cards"></div><div class="pg" id="pg"></div>
 </div>
</div>

<div class="tabpane" id="tab_papers" style="display:none">
 <div class="panel"><h2>Source papers</h2>
  <p class="hint">One row per paper in the current selection. Click any column header to
    sort; <b>n_topics</b> counts the topic rows that survive the current filters. Titles
    link to the DOI, and <b>browse</b> opens that paper's topics with its full record.</p>
  <div id="paper_tbl"></div></div>
</div>

<div class="tabpane" id="tab_flags" style="display:none">
 <div class="grid2">
  <div class="panel"><h2>Flag prevalence</h2>
   <p class="hint">Why rows fall outside the strict tier</p><div id="flag_tbl"></div></div>
  <div class="panel"><h2>Flag co-occurrence</h2>
   <p class="hint">Row flag &cap; column flag, shaded by share of the row flag's topics.
     Diagonal is the flag's own total.</p>
   <div style="overflow:auto"><div id="cooc"></div></div></div>
 </div>
 <div class="panel note">
  <b>Reading the flags.</b> <code>index_prefix_stripped</code> means a table row number was
  removed from the label. <code>label_is_a_keyword</code> and <code>label_in_abstract</code>
  mark trivially-easy or leakage-prone rows. <code>keyword_order_unreliable</code> marks
  word-cloud clusters where membership is sound but rank is approximate.
  <code>source_row_defect</code> marks a mislabel in the published paper, preserved rather
  than corrected. See <code>PREPROCESSING.md</code> for every rule and its prevalence.
 </div>
</div>

</div>
<script>{js}</script>
</body></html>"""
    open(out, "w").write(html)
    return len(html)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)                      # repo root, if run from scripts/
    dflt = os.path.join(root, "data")
    if not os.path.isdir(dflt):                       # fall back to cwd/data
        dflt = os.path.join(os.getcwd(), "data")

    p = argparse.ArgumentParser(
        description="Build the self-contained TopicBench data-explorer dashboard.")
    p.add_argument("--topics", default=os.path.join(dflt, "topicbench_topics.csv"),
                   help="one row per topic (default: data/topicbench_topics.csv)")
    p.add_argument("--bibliography", default=os.path.join(dflt, "topicbench_bibliography.csv"),
                   help="one row per paper (default: data/topicbench_bibliography.csv)")
    p.add_argument("--out", default=os.path.join(dflt, "data_dashboard.html"),
                   help="output HTML (default: data/data_dashboard.html)")
    p.add_argument("--generated", default="",
                   help="optional build stamp recorded in the payload metadata")
    a = p.parse_args(argv)

    for path in (a.topics, a.bibliography):
        if not os.path.exists(path):
            raise SystemExit(f"input not found: {path}\n"
                             f"Run from the repo root, or pass explicit --topics/"
                             f"--bibliography paths.")

    payload = load_payload(a.topics, a.bibliography, generated=a.generated)
    n = build(payload, a.out)
    m = payload["meta"]
    print(f"wrote {a.out}  ({n/1e6:.2f} MB)")
    print(f"  {m['n_topics']:,} topics from {m['n_papers']} papers "
          f"({m['n_bib']} papers in the bibliography)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
