
import json
import random
import httpx
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import re


# ============================
# Enhanced Heatmap & Correlation Helpers (compatible with existing app structures)
# ============================
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import hashlib, random

def _stable_rng(seed_text: str):
    h = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))

def _ensure_demo_records():
    """
    Prefer the app's own _seed_full_demo_heatmap() if it exists.
    Otherwise, create minimal demo records *matching this app's schema*:
    st.session_state['score_records'] = [{'entity','channel','variant','scores':{attr:{'score':float}}}]
    """
    if "score_records" in st.session_state and st.session_state["score_records"]:
        return
    # Use existing seeder if available
    if "_seed_full_demo_heatmap" in globals():
        try:
            _seed_full_demo_heatmap()
            return
        except Exception:
            pass
    # Fallback: synthesize some stable demo entries
    rng = _stable_rng("brandlift-demo-v2")
    try:
        channels = CHANNELS
    except NameError:
        channels = ["CTV","DOOH","Youtube","TikTok","Google Ads","Instagram","X"]
    try:
        attrs_code = ATTRS
    except NameError:
        attrs_code = ["Leadership","Ease_of_Use","Quality","Luxury","Cost_Benefit","Trust"]
    entities = ["Client","Competitor A"]
    variants = {"Client":["Original","Improved"], "Competitor A":["Competitor"]}
    recs = []
    for ch_i, ch in enumerate(channels):
        for ent in entities:
            for var in variants[ent]:
                scores = {}
                for a_i, a in enumerate(attrs_code):
                    base = 0.55 if ent=="Client" else 0.6
                    val = max(0.0, min(1.0, base + (ch_i%5)*0.03 + (a_i%3)*0.025 + rng.uniform(-0.08,0.08)))
                    scores[a] = {"score": round(val,3), "evidence": ""}
                recs.append({"entity": ent, "channel": ch, "variant": var, "scores": scores})
    st.session_state["score_records"] = recs

def _get_channel_attribute_pivot():
    """
    Return a pivot DataFrame with index 'Channel' and attribute columns (space-normalized),
    using the app's native helper if present.
    """
    if "_records_to_channel_attr_medians" in globals():
        try:
            df = _records_to_channel_attr_medians()
            # If helper returns reset_index(), ensure Channel is index for plotting convenience
            if "Channel" in df.columns:
                df = df.set_index("Channel")
            return df
        except Exception as _e:
            pass
    # Fallback: build from score_records directly
    if "score_records" not in st.session_state or not st.session_state["score_records"]:
        return pd.DataFrame()
    rows = []
    # Determine attribute keys from first record
    attrs_code = None
    for r in st.session_state["score_records"]:
        if isinstance(r.get("scores"), dict):
            attrs_code = list(r["scores"].keys())
            break
    if not attrs_code:
        return pd.DataFrame()
    for r in st.session_state["score_records"]:
        ch = r.get("channel")
        sc = r.get("scores", {})
        for a in attrs_code:
            try:
                rows.append({"Channel": ch, "Attribute": a.replace("_"," "), "Score": float(sc[a]["score"])})
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    med = df.groupby(["Channel","Attribute"], as_index=False)["Score"].median()
    pivot = med.pivot(index="Channel", columns="Attribute", values="Score")
    return pivot

def render_attribute_heatmap_enhanced(show_values=True, decimals=2, sort_by_attr=None, title="Channel × Attribute (Median Scores)"):
    piv = _get_channel_attribute_pivot()
    if piv is None or piv.empty:
        st.info("No scores yet to build the heatmap.")
        return
    # Optional sorting
    if sort_by_attr and sort_by_attr in piv.columns:
        piv = piv.sort_values(by=sort_by_attr, ascending=False)
    z = piv.values
    x = list(piv.columns)
    y = list(piv.index)
    text = None
    texttemplate = None
    if show_values:
        text = np.round(z, decimals).astype(str)
        texttemplate = "%{text}"
    fig = px.imshow(
        z, x=x, y=y, zmin=0, zmax=1,
        color_continuous_scale="RdYlGn",
        text=text, aspect="auto"
    )
    if texttemplate:
        fig.update_traces(texttemplate=texttemplate)
    fig.update_layout(
        title=title, xaxis_title="Attribute", yaxis_title="Channel",
        margin=dict(l=60,r=20,t=60,b=40),
        coloraxis_colorbar=dict(title="Score")
    )
    st.plotly_chart(fig, use_container_width=True)

def _records_long_for_correlation():
    """
    Expand score_records into long format suitable for correlation:
    columns: Entity, Variant, Channel, Attribute, Score
    """
    if "score_records" not in st.session_state or not st.session_state["score_records"]:
        return pd.DataFrame(columns=["Entity","Variant","Channel","Attribute","Score"])
    rows = []
    # Determine attribute keys
    attrs_code = None
    for r in st.session_state["score_records"]:
        if isinstance(r.get("scores"), dict):
            attrs_code = list(r["scores"].keys())
            break
    if not attrs_code:
        return pd.DataFrame(columns=["Entity","Variant","Channel","Attribute","Score"])
    for r in st.session_state["score_records"]:
        ent = r.get("entity","")
        var = r.get("variant","")
        ch  = r.get("channel","")
        sc = r.get("scores",{})
        for a in attrs_code:
            try:
                rows.append({"Entity": ent, "Variant": var, "Channel": ch, "Attribute": a.replace("_"," "), "Score": float(sc[a]["score"])})
            except Exception:
                continue
    return pd.DataFrame(rows)

def render_attribute_correlation_explorer():
    df_long = _records_long_for_correlation()
    if df_long.empty or df_long["Score"].count() < 5:
        st.info("Not enough/valid data for correlation yet.")
        return
    wide = df_long.pivot_table(index=["Entity","Variant","Channel"], columns="Attribute", values="Score", aggfunc="mean").reset_index()
    # Drop non-varying columns
    num_cols = [c for c in wide.columns if c not in ["Entity","Variant","Channel"]]
    keep = []
    for c in num_cols:
        s = pd.to_numeric(wide[c], errors="coerce")
        if s.notna().sum() >= 2 and s.var() > 0:
            keep.append(c)
    if len(keep) < 2:
        st.info("Need at least two varying attributes to compute correlations.")
        return
    corr = wide[keep].corr(method="pearson").round(3)
    fig = px.imshow(corr.values, x=keep, y=keep, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text=corr.values)
    fig.update_traces(texttemplate="%{text}")
    fig.update_layout(
        title="Attribute Correlation Explorer (Pearson)",
        xaxis_title="Attribute", yaxis_title="Attribute",
        margin=dict(l=60,r=20,t=60,b=40), coloraxis_colorbar=dict(title="ρ")
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================
# Demo Seed + Enhanced Heatmap + Correlation Explorer Sections
# ============================
try:
    _ensure_demo_records()
except Exception:
    pass

st.markdown("## Enhanced Channel × Attribute Heatmap")
try:
    col_h1, col_h2, col_h3 = st.columns([1,1,1])
    with col_h1:
        _show_vals = st.checkbox("Show values on heatmap", value=True, help="Annotate each cell with its value.")
    with col_h2:
        _decimals = st.slider("Round to (decimals)", 0, 3, 2)
    # determine attribute options from pivot
    _piv_preview = _get_channel_attribute_pivot()
    opts = ["(none)"] + (list(_piv_preview.columns) if _piv_preview is not None and not _piv_preview.empty else [])
    with col_h3:
        _sort_attr = st.selectbox("Sort channels by attribute (optional)", opts, index=0)
        _sort_attr = None if _sort_attr == "(none)" else _sort_attr
    render_attribute_heatmap_enhanced(show_values=_show_vals, decimals=_decimals, sort_by_attr=_sort_attr)
except Exception as e:
    st.warning(f"Enhanced heatmap could not be rendered: {e}")

st.markdown("## Attribute Correlation Explorer")
try:
    render_attribute_correlation_explorer()
except Exception as e:
    st.warning(f"Correlation explorer could not be rendered: {e}")

