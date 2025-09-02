# app_brandlift_integrated.py (CLEAN REBUILD)
import json, random, httpx, streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd, numpy as np
from pathlib import Path
import re

BRAND_BLUE   = "#445DA7"
BRAND_PURPLE = "#6B3894"
BRAND_NAVY   = "#2E3C71"
COMP_TEAL    = "#2AA9A1"
TITLE_COLOR = BRAND_NAVY

CHANNELS = ["CTV", "DOOH", "Youtube", "TikTok", "Google Ads", "Instagram", "X"]
ATTRS = ["Leadership", "Ease_of_Use", "Quality", "Luxury", "Cost_Benefit", "Trust"]

def _pretty_attr(a: str) -> str:
    return a.replace("_", " ").replace("Benefit", "Benefit").replace("Cost Benefit", "Cost/Benefit")

BRAND_DIVERGING = [(0.0, BRAND_PURPLE),(0.5, "#E9E7F4"),(1.0, BRAND_NAVY)]
PREFERRED_MODEL = "claude-3-5-sonnet-latest"; FALLBACK_MODEL  = "claude-3-5-haiku-latest"
API_BASE = "https://api.anthropic.com/v1/messages"

SYSTEM_SCORE = """You are a brand perception rater. Score ad copy on 6 attributes:
Leadership, Ease_of_Use, Quality, Luxury, Cost_Benefit, Trust.
Each score is a float in [0,1]. Anchors: 0.2 = weak, 0.5 = moderate, 0.8 = strong.
Return STRICT JSON with each attribute as {"score": float, "evidence": "short phrase (<=12 words)"}.
No extra text.
"""
SYSTEM_REWRITE = "You are a precise brand copy editor focused on targeted brand attributes."

def _schema() -> dict:
    return {"type":"object","properties":{k:{"type":"object","properties":{"score":{"type":"number"},"evidence":{"type":"string"}},"required":["score","evidence"]} for k in ATTRS},"required":ATTRS,"additionalProperties":False}

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("\n", 1); text = parts[1] if len(parts) == 2 else ""
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()

def _extract_json_substring(text: str) -> str | None:
    text = _strip_code_fences(text)
    try: json.loads(text); return text
    except Exception: pass
    start_idx=None
    for i,ch in enumerate(text):
        if ch in '{[': start_idx=i; break
    if start_idx is None: return None
    stack=[text[start_idx]]; i=start_idx+1; in_str=False; esc=False
    while i < len(text):
        ch=text[i]
        if in_str:
            if esc: esc=False
            elif ch == '\\': esc=True
            elif ch == '"': in_str=False
        else:
            if ch == '"': in_str=True
            elif ch in '{[': stack.append(ch)
            elif ch in '}]':
                if not stack: return None
                top=stack[-1]
                if (top=='{' and ch=='}') or (top=='[' and ch==']'):
                    stack.pop()
                    if not stack:
                        candidate=text[start_idx:i+1]
                        try: json.loads(candidate); return candidate
                        except Exception: pass
                else:
                    return None
        i+=1
    return None

def _parse_json_block(text: str) -> dict:
    if not isinstance(text, str): raise RuntimeError("Model did not return text content.")
    text=(text.replace("\u201c", '"').replace("\u201d", '"').replace("“", '"').replace("”", '"').replace("’","'").replace("‘","'"))
    candidate=_extract_json_substring(text)
    if candidate is None: raise RuntimeError("Model did not return JSON. Output was: "+text[:400])
    try: return json.loads(candidate)
    except json.JSONDecodeError as e:
        jlike=re.sub(r"(?<!\\)'",'\"',candidate)
        try: return json.loads(jlike)
        except Exception: raise RuntimeError(f"Could not parse JSON. Error: {e}. Output was: {text[:400]}")

def _clean_attribute_tags(text: str) -> str:
    attrs_alt = "|".join([re.escape(a) for a in ATTRS] + [re.escape(a.replace("_"," ")) for a in ATTRS])
    text = re.sub(r"\[(" + attrs_alt + ")\s*:[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:" + attrs_alt + ")[^\)]*\)", "", text, flags=re.IGNORECASE)
    # Curly braces block using string format to avoid f-string escaping
    text = re.sub(r"\{(?:" + attrs_alt + ")[^\}]*\}", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(" + attrs_alt + ")\s*[:\-–]\s*.*?$", "", text, flags=re.IGNORECASE|re.MULTILINE)
    return re.sub(r"\s{2,}", " ", text).strip()

def _call_messages(api_key: str, system: str, user_content: str, model: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    headers={"x-api-key": api_key,"anthropic-version":"2023-06-01","content-type":"application/json"}
    payload={"model":model,"max_tokens":max_tokens,"temperature":temperature,"system":system,"messages":[{"role":"user","content":user_content}]}
    with httpx.Client(timeout=60) as client:
        r=client.post(API_BASE,headers=headers,json=payload); r.raise_for_status()
    data=r.json(); blocks=data.get("content",[])
    texts=[b["text"] for b in blocks if isinstance(b,dict) and b.get("type")=="text" and "text" in b]
    return "\n".join(texts) if texts else ""

def _with_fallback(func,*args,**kwargs):
    try: return func(PREFERRED_MODEL,*args,**kwargs), PREFERRED_MODEL
    except Exception: return func(FALLBACK_MODEL,*args,**kwargs), FALLBACK_MODEL

def score_text(api_key: str, text: str) -> dict:
    user_content=f"""Text:
{text}

Output JSON schema:
{json.dumps(_schema())}"""
    def run(model, api_key_inner, uc):
        out_text=_call_messages(api_key_inner,SYSTEM_SCORE,uc,model)
        scores=_parse_json_block(out_text)
        for k in ATTRS:
            s=float(scores[k]["score"]); scores[k]["score"]=max(0.0,min(1.0,s))
        return scores
    scores,_=_with_fallback(run,api_key,user_content); return scores

def rewrite_copy_to_targets(api_key: str, text: str, target_levels: dict) -> str:
    norm_targets={}
    for k,v in target_levels.items():
        kk=k.replace(" ","_").replace("/","_")
        if kk not in ATTRS and kk.title() in ATTRS: kk=kk.title()
        norm_targets[kk]=max(0.0,min(1.0,float(v)))
    targets_str=", ".join([f"{k}:{v:.2f}" for k,v in norm_targets.items()])
    cleaned=_clean_attribute_tags(text)
    example=json.dumps({"rewrite":"..."},ensure_ascii=False)
    instr=f"""Rewrite the ad copy to better align perceived brand attributes with these targets (0–1):
{targets_str}

Guidelines:
- Keep the meaning and promises credible.
- Use natural language; DO NOT include explicit attribute labels or tags.
- Max 90 words. Return JSON: {example}

Original (cleaned of tags):
{cleaned}"""
    def run(model, api_key_inner, prompt_text):
        out_text=_call_messages(api_key_inner,SYSTEM_REWRITE,prompt_text,model,temperature=0.5,max_tokens=450)
        try:
            data=_parse_json_block(out_text); candidate=_clean_attribute_tags(data.get("rewrite","" ).strip())
            if candidate: return candidate
        except Exception: pass
        clean=_clean_attribute_tags(_strip_code_fences(out_text).strip())
        return " ".join(clean.split()[:90]) if clean else cleaned
    rewrite,_=_with_fallback(run,api_key,instr); return rewrite or cleaned

def radar(scores: dict, title: str, fill_color: str, line_color: str):
    labels=[_pretty_attr(k) for k in ATTRS]; vals=[scores[k]["score"] for k in ATTRS]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals+vals[:1],theta=labels+labels[:1],fill="toself",fillcolor=fill_color,line=dict(color=line_color,width=2),name=title))
    fig.update_layout(title=title,polar=dict(radialaxis=dict(range=[0,1])),showlegend=False,margin=dict(l=30,r=30,t=40,b=30)); return fig

def scores_table(scores: dict)->pd.DataFrame:
    return pd.DataFrame([{ "Attribute":_pretty_attr(k), "Score":round(float(scores[k]["score"]),2), "Key phrase":scores[k]["evidence"] } for k in ATTRS])

def inject_css():
    st.markdown(f"""
    <style>
    .stButton > button {{
        background-color: {BRAND_BLUE};
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1rem; font-weight: 600;
    }}
    .stButton > button:hover {{ background-color: {BRAND_PURPLE}; color: white; }}

    .badge {{
        display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
        color: white; font-size: 0.85rem; margin-right: 0.35rem; margin-bottom: 0.35rem;
    }}
    .badges {{ margin: 0.25rem 0 0.75rem 0; }}
    .block-container {{ padding-top: 1rem; }}
    .app-title {{ color: {TITLE_COLOR}; margin: 0; }}

    .stSlider > div[data-baseweb="slider"] [class*="rail"] {{
        background: #E0E4F2 !important;
        height: 6px !important; border-radius: 6px !important;
    }}
    .stSlider > div[data-baseweb="slider"] [class*="track"] {{
        background: linear-gradient(to right, {BRAND_BLUE}, {BRAND_PURPLE}) !important;
        height: 6px !important; border-radius: 6px !important;
    }}
    .stSlider [role="slider"] {{
        background-color: {BRAND_PURPLE} !important;
        border: 2px solid {BRAND_NAVY} !important; box-shadow: none !important;
    }}
    .stSlider [role="slider"]:focus {{
        outline: 2px solid {BRAND_BLUE} !important; outline-offset: 2px;
    }}
    </style>
    """, unsafe_allow_html=True)

def _init_records():
    if "score_records" not in st.session_state: st.session_state["score_records"]=[]

def _upsert_record(entity: str, channel: str, variant: str, scores: dict):
    _init_records(); key=(entity,channel,variant)
    for r in st.session_state["score_records"]:
        if (r["entity"],r["channel"],r["variant"])==key: r["scores"]=scores; break
    else:
        st.session_state["score_records"].append({"entity":entity,"channel":channel,"variant":variant,"scores":scores})

def _records_to_channel_attr_medians(entities: list[str] | None = None, variants: list[str] | None = None)->pd.DataFrame:
    _init_records(); rows=[]
    for r in st.session_state["score_records"]:
        if entities and r["entity"] not in entities: continue
        if variants and r["variant"] not in variants: continue
        channel=r["channel"]; sc=r["scores"]
        for attr in ATTRS: rows.append({"Channel":channel,"Attribute":_pretty_attr(attr),"Score":float(sc[attr]["score"])})
    if not rows: return pd.DataFrame(columns=["Channel"]+[_pretty_attr(a) for a in ATTRS])
    df=pd.DataFrame(rows); med=df.groupby(["Channel","Attribute"],as_index=False)["Score"].median()
    pivot=med.pivot(index="Channel",columns="Attribute",values="Score").reindex(CHANNELS,fill_value=None)
    pivot=pivot[[_pretty_attr(a) for a in ATTRS]]
    return pivot.reset_index()

def _heatmap(fig_df: pd.DataFrame,title:str="Attribute Importance Heatmap"):
    if fig_df.empty or len(fig_df.columns)<=1:
        st.info("Not enough scored items to build a heatmap yet. Score at least one item."); return
    z=fig_df.drop(columns=["Channel"]).values; x=list(fig_df.columns[1:]); y=list(fig_df["Channel"])
    hm=go.Figure(data=go.Heatmap(colorscale=[[0.0, BRAND_PURPLE],[0.5, BRAND_BLUE],[1.0, BRAND_NAVY]], z=z,x=x,y=y,zmin=0.0,zmax=1.0, colorbar=dict(title="Median Score"),
        hovertemplate="Channel: %{y}<br>Attribute: %{x}<br>Median: %{z:.2f}<extra></extra>"))
    hm.update_layout(title=title,margin=dict(l=40,r=20,t=60,b=40)); st.plotly_chart(hm,use_container_width=True); st.dataframe(fig_df,use_container_width=True)

def _records_to_long_df(records: list[dict])->pd.DataFrame:
    rows=[]
    for r in records:
        e=r.get("entity"); ch=r.get("channel"); v=r.get("variant"); scores=r.get("scores",{})
        for a in ATTRS:
            obj=scores.get(a,{"score":None}); val=obj["score"] if isinstance(obj,dict) and "score" in obj else obj
            rows.append({"entity":e,"channel":ch,"variant":v,"attribute":_pretty_attr(a),"score":float(val) if val is not None else None})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.dropna(subset=["score"]); df=df[(df["score"]>=0)&(df["score"]<=1)]
    return df

def _attribute_correlation(df_long: pd.DataFrame)->pd.DataFrame:
    if df_long.empty: return pd.DataFrame()
    wide=df_long.pivot_table(index=["entity","channel","variant"],columns="attribute",values="score",aggfunc="median")
    if wide.empty: return pd.DataFrame()
    non_const=wide.loc[:,wide.std(numeric_only=True)>1e-12]
    if non_const.shape[1]<2: return pd.DataFrame()
    return non_const.corr(numeric_only=True)

def render_correlation_section():
    st.subheader("Attribute Correlation Explorer")
    st.caption("See how attributes move together across all scored items (entity × channel × variant)." )
    _init_records()
    df_long=_records_to_long_df(st.session_state.get("score_records",[]))
    if df_long.empty: st.info("Not enough data yet."); return
    col1,col2,col3=st.columns(3)
    with col1: ent_sel=st.multiselect("Entities", sorted(df_long["entity"].dropna().unique().tolist()), default=sorted(df_long["entity"].dropna().unique().tolist()))
    with col2: ch_sel=st.multiselect("Channels", sorted(df_long["channel"].dropna().unique().tolist()), default=sorted(df_long["channel"].dropna().unique().tolist()))
    with col3: var_sel=st.multiselect("Variants", sorted(df_long["variant"].dropna().unique().tolist()), default=sorted(df_long["variant"].dropna().unique().tolist()))
    df_filt=df_long[df_long["entity"].isin(ent_sel)&df_long["channel"].isin(ch_sel)&df_long["variant"].isin(var_sel)].copy()
    corr=_attribute_correlation(df_filt)
    if corr.empty: st.info("Not enough variation to compute correlations."); return
    fig=px.imshow(corr.values,x=list(corr.columns),y=list(corr.index),zmin=-1,zmax=1,color_continuous_scale=BRAND_DIVERGING,text_auto=True,aspect="auto")
    fig.update_layout(margin=dict(l=60,r=30,t=30,b=60),coloraxis_colorbar=dict(title="Corr",tickformat=".2f"))
    st.plotly_chart(fig,use_container_width=True)

st.set_page_config(page_title="Brand Lift", layout="wide")
inject_css()

hdr_left,hdr_right=st.columns([4,1])
with hdr_left: st.markdown("<h1 class='app-title'>Brand Lift</h1>", unsafe_allow_html=True)
with hdr_right:
    logo_path=Path("logo.png")
    if logo_path.exists(): st.image(str(logo_path), use_container_width=True)

api_key=st.secrets.get("ANTHROPIC_API_KEY","").strip()
if not api_key: st.info('Add your key in Settings → Secrets as:\nANTHROPIC_API_KEY = "sk-ant-..."')

st.markdown("**Attributes**")
st.markdown("""
<div class="badges">
    <span class="badge" style="background:#445DA7">Leadership</span>
    <span class="badge" style="background:#6B3894">Ease of Use</span>
    <span class="badge" style="background:#2E3C71">Quality</span>
    <span class="badge" style="background:#6B3894">Luxury</span>
    <span class="badge" style="background:#445DA7">Cost/Benefit</span>
    <span class="badge" style="background:#2E3C71">Trust</span>
</div>
""", unsafe_allow_html=True)

text=st.text_area("Paste ad copy or transcript:", height=160, placeholder="Example: Introducing the next-generation sedan—hand-finished interiors, seamless app control, and advanced safety.")
client_url=st.text_input("Client creative URL (optional)", placeholder="https://...")
client_channel=st.selectbox("Client channel", CHANNELS, index=0)

c1,c2,c3=st.columns([1,1,2])
with c1: score_clicked=st.button("Score", use_container_width=True)
with c2: pass
with c3:
    comp_cols=st.columns([1.2,2.8])
    with comp_cols[0]: comp_name=st.text_input("Competitor label", value="Competitor A")
    with comp_cols[1]: comp_url=st.text_input("Competitor ad/creative URL (optional)", placeholder="https://...")
    comp_text=st.text_area("Competitor copy/transcript (optional)", height=120, placeholder="Paste competitor copy/transcript to score (optional)")
    comp_channel=st.selectbox("Competitor channel", CHANNELS, index=0)
    score_comp=st.button("Score Competitor", use_container_width=True)

with st.expander("Desired attribute targets", expanded=True):
    desired_targets={}
    cols=st.columns(3)
    for i,a in enumerate(ATTRS):
        with cols[i%3]:
            pretty=_pretty_attr(a)
            desired_targets[pretty]=st.slider(pretty,0.0,1.0,0.78,0.01, help="Target perceived level for this attribute")
    st.caption("These targets drive the improvement step. The rewrite will avoid explicit attribute tags and aim for these levels in tone and content.")

improve_clicked=st.button("Improve & Rescore", use_container_width=True)

def _maybe_url_media(url:str):
    try:
        lower=url.lower()
        if lower.endswith(('.png','.jpg','.jpeg','.webp','.gif')): st.image(url,use_container_width=True)
        elif lower.endswith(('.mp4','.webm','.ogg')) or 'youtube.com' in lower or 'youtu.be' in lower or 'vimeo.com' in lower: st.video(url)
    except Exception: pass

if score_clicked:
    if not api_key: st.error("No API key found in Secrets.")
    elif not client_channel: st.warning("Please select a channel before scoring.")
    elif not text.strip(): st.warning("Please paste some text.")
    else:
        with st.spinner("Scoring…"):
            base_scores=score_text(api_key,text)
            st.session_state["base_text"]=text; st.session_state["base_scores"]=base_scores; st.session_state["client_channel"]=client_channel

if improve_clicked:
    source_text=(text.strip() or st.session_state.get("improved_text") or st.session_state.get("base_text","" )).strip()
    if not api_key: st.error("No API key found in Secrets.")
    elif not client_channel: st.warning("Please select a channel before scoring.")
    elif not source_text: st.warning("Paste text and click Score first.")
    else:
        with st.spinner("Generating improved copy to match targets…"):
            improved_text=rewrite_copy_to_targets(api_key, source_text, desired_targets)
        with st.spinner("Scoring improved copy…"):
            improved_scores=score_text(api_key, improved_text)
        st.session_state["improved_text"]=improved_text; st.session_state["improved_scores"]=improved_scores

base_scores=st.session_state.get("base_scores"); improved_scores=st.session_state.get("improved_scores")
comp_scores=st.session_state.get("competitor_scores"); comp_label=st.session_state.get("competitor_name","Competitor")
comp_link=st.session_state.get("competitor_url","" )

if "score_comp" in locals() and score_comp:
    st.session_state["competitor_name"]=comp_name or "Competitor"
    st.session_state["competitor_url"]=comp_url.strip()
    if not api_key: st.error("No API key found in Secrets.")
    elif not client_channel: st.warning("Please select a channel before scoring.")
    elif (comp_text or "").strip():
        if not comp_channel: st.warning("Please select a competitor channel before scoring.")
        else:
            with st.spinner("Scoring competitor…"):
                s=score_text(api_key, comp_text); st.session_state["competitor_scores"]=s
    else: st.warning("Paste competitor copy/transcript to score.")

if base_scores or improved_scores or comp_scores:
    st.subheader("Comparison")
    cols_to_use=[bool(base_scores), bool(improved_scores), bool(comp_scores)]; n=sum(cols_to_use) or 1; cols=st.columns(n)
    idx=0
    if base_scores:
        with cols[idx]:
            st.markdown("**Original Copy**"); st.write(st.session_state.get("base_text",""))
            if client_url: st.markdown(f"[Open creative]({client_url})"); _maybe_url_media(client_url)
            st.plotly_chart(radar(base_scores,"Original Scores","rgba(68, 93, 167, 0.45)","rgba(68, 93, 167, 1.0)"), use_container_width=True)
            st.dataframe(scores_table(base_scores), use_container_width=True); idx+=1
    if improved_scores:
        with cols[idx]:
            st.markdown("**Improved Copy**"); st.write(st.session_state.get("improved_text",""))
            st.plotly_chart(radar(improved_scores,"Improved Scores","rgba(107, 56, 148, 0.45)","rgba(107, 56, 148, 1.0)"), use_container_width=True)
            st.dataframe(scores_table(improved_scores), use_container_width=True); idx+=1
    if comp_scores:
        with cols[idx]:
            st.markdown(f"**{comp_label}**")
            if comp_link: st.markdown(f"[Open creative]({comp_link})"); _maybe_url_media(comp_link)
            st.plotly_chart(radar(comp_scores,f"{comp_label} Scores","rgba(42, 169, 161, 0.45)","rgba(42, 169, 161, 1.0)"), use_container_width=True)
            st.dataframe(scores_table(comp_scores), use_container_width=True)

st.subheader("Attribute Importance Heatmap")
st.caption("Color-coded median scores of each attribute across channels from all scored items (client, improved, and competitors)." )

def _seed_full_demo_heatmap():
    _init_records()
    if st.session_state.get("score_records"): return
    for i,ch in enumerate(CHANNELS):
        def _rand(seed): random.seed(seed); return round(random.uniform(0.25,0.9),2)
        st.session_state.setdefault("score_records",[]).append({"entity":"Client","channel":ch,"variant":"Original","scores":{a:{"score":_rand(100+i*6+idx),"evidence":"demo"} for idx,a in enumerate(ATTRS)}})
        st.session_state["score_records"].append({"entity":"Client","channel":ch,"variant":"Improved","scores":{a:{"score":_rand(200+i*6+idx),"evidence":"demo"} for idx,a in enumerate(ATTRS)}})
        st.session_state["score_records"].append({"entity":"Competitor A","channel":ch,"variant":"Competitor","scores":{a:{"score":_rand(300+i*6+idx),"evidence":"demo"} for idx,a in enumerate(ATTRS)}})

_seed_full_demo_heatmap()
all_entities=sorted({r["entity"] for r in st.session_state.get("score_records",[])}|{"Client"})
all_variants=sorted({r["variant"] for r in st.session_state.get("score_records",[])}|{"Original","Improved","Competitor"})
fcol1,fcol2=st.columns([2,2])
with fcol1: sel_entities=st.multiselect("Entities", options=all_entities, default=all_entities)
with fcol2: sel_variants=st.multiselect("Variants", options=all_variants, default=all_variants)
heatmap_df=_records_to_channel_attr_medians(entities=sel_entities or None, variants=sel_variants or None)
_heatmap(heatmap_df, title="Attribute Importance Heatmap (Median by Channel)")

st.subheader("Channel Trends Over Time")
st.caption("12-month synthetic median attribute trends, per selected channel (demo data)." )

def _seed_demo_trends():
    if "monthly_attr_trends" in st.session_state: return
    _init_records()
    if not st.session_state.get("score_records"): _seed_full_demo_heatmap()
    months=pd.date_range(end=pd.Timestamp.today().normalize()+pd.offsets.MonthEnd(0), periods=12, freq="M").strftime("%Y-%m")
    entities=sorted({r["entity"] for r in st.session_state["score_records"]}|{"Client"})
    variants=sorted({r["variant"] for r in st.session_state["score_records"]}|{"Original","Improved","Competitor"})
    def random_walk_series(seed:int,start:float)->list[float]:
        import random as _rnd; _rnd.seed(seed); vals=[max(0.0,min(1.0,start))]; sigma=0.16
        for _ in range(1,12):
            step=_rnd.gauss(0.0,sigma); nxt=vals[-1]+step
            if nxt<0.0: nxt=-nxt
            if nxt>1.0: nxt=2.0-nxt
            vals.append(max(0.0,min(1.0,nxt)))
        vmin,vmax=min(vals),max(vals); span=vmax-vmin
        if span<0.65:
            mean=sum(vals)/len(vals); expand=(0.80/max(span,1e-6)); vals=[mean+(v-mean)*expand for v in vals]
            vals=[min(1.0,max(0.0,(2.0-v) if v>1.0 else (-v if v<0.0 else v))) for v in vals]
        vmin,vmax=min(vals),max(vals)
        if vmax<0.92: add=0.92-vmax; vals=[min(1.0,v+add*0.6) for v in vals]
        if vmin>0.08: sub=vmin-0.08; vals=[max(0.0,v-sub*0.6) for v in vals]
        return [round(v,3) for v in vals]
    rows=[]
    for ent in entities:
        for var in variants:
            for ch in CHANNELS:
                for attr in ATTRS:
                    seed_val=abs(hash(f"rw:{ent}:{var}:{ch}:{attr}"))%(2**32)
                    start=0.05+(seed_val%9000)/9000.0*0.90
                    series=random_walk_series(seed_val,start)
                    for m,v in zip(months,series):
                        rows.append({"Entity":ent,"Variant":var,"Channel":ch,"Month":m,"Attribute":_pretty_attr(attr),"Score":v})
    st.session_state["monthly_attr_trends"]=pd.DataFrame(rows)

def _plot_channel_trends(df_channel: pd.DataFrame):
    fig=go.Figure()
    for attr in sorted(df_channel["Attribute"].unique()):
        sdf=df_channel[df_channel["Attribute"]==attr].sort_values("Month")
        fig.add_trace(go.Scatter(x=sdf["Month"],y=sdf["Score"],mode="lines+markers",name=attr))
    fig.update_layout(title="Attribute trends (12 months)",xaxis_title="Month",yaxis_title="Score",yaxis=dict(range=[0,1]),margin=dict(l=40,r=20,t=60,b=40),legend_title="Attribute")
    st.plotly_chart(fig,use_container_width=True)
    wide=df_channel.pivot(index="Month", columns="Attribute", values="Score").reset_index()
    st.dataframe(wide,use_container_width=True)

_seed_demo_trends()
trend_channel=st.selectbox("Select channel", CHANNELS, index=0, key="trend_channel")
trend_df=st.session_state["monthly_attr_trends"]
filtered=trend_df[(trend_df["Channel"]==trend_channel)]
agg=filtered.groupby(["Month","Attribute"],as_index=False)["Score"].median() if not filtered.empty else filtered.copy()
_plot_channel_trends(agg)

render_correlation_section()
