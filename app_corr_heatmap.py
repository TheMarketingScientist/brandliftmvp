
# app_corr_heatmap_branded.py
# Streamlit helpers for Channel×Attribute heatmaps + Attribute Correlation Explorer
# with The Marketing Scientist brand colors, filters, and downloads.

from __future__ import annotations

import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================
# Brand palette (from logo)
# ============================
BRAND_NAVY   = "#283868"
BRAND_BLUE   = "#4058A0"
BRAND_PURPLE = "#603088"

# Sequential scale for 0→1 heatmaps (low→high)
BRAND_SEQ = [BRAND_PURPLE, BRAND_BLUE, BRAND_NAVY]

# Diverging scale for correlations (−1→0→+1)
BRAND_DIVERGING = [
    (0.0, BRAND_PURPLE),   # negative
    (0.5, "#E9E7F4"),      # light neutral
    (1.0, BRAND_NAVY),     # positive
]

# ============================
# Demo data (only used if app hasn't populated session_state yet)
# ============================
DEFAULT_CHANNELS = ["TV", "CTV", "Radio", "YouTube", "TikTok", "Meta", "Instagram", "X", "DOOH", "OOH", "Google Ads"]
DEFAULT_ATTRS    = ["Leadership", "Ease of Use", "Quality", "Luxury", "Cost/Benefit", "Trust"]

def _seed_demo_records(n_entities:int=2, n_variants:int=3, seed:int=42):
    """Fill st.session_state['score_records'] with minimal demo data ONLY if empty.
    Schema expected by this helper:
    {
      'entity': str, 'channel': str, 'variant': str,
      'scores': {attribute: {'score': float in [0,1]}}
    }
    """
    if "score_records" in st.session_state and st.session_state["score_records"]:
        return
    rng = random.Random(seed)
    records = []
    entities = ["Client", "Competitor"]
    channels = DEFAULT_CHANNELS
    attrs    = DEFAULT_ATTRS
    for e in entities[:n_entities]:
        for ch in channels:
            for v in range(1, n_variants+1):
                scores = {}
                base = rng.uniform(0.35, 0.65)
                # channel bias
                ch_bias = (channels.index(ch) / (len(channels)-1) - 0.5) * 0.15
                for a in attrs:
                    a_bias = (attrs.index(a) / (len(attrs)-1) - 0.5) * 0.18
                    val = base + ch_bias + a_bias + rng.uniform(-0.12, 0.12)
                    scores[a] = {"score": max(0.0, min(1.0, val))}
                records.append({
                    "entity": e,
                    "channel": ch,
                    "variant": f"V{v}",
                    "scores": scores
                })
    st.session_state["score_records"] = records

# ============================
# Data transforms
# ============================
def _records_to_long_df(records: List[dict]) -> pd.DataFrame:
    """Flatten records into long form: entity, channel, variant, attribute, score"""
    rows = []
    for r in records:
        e = r.get("entity")
        ch = r.get("channel")
        v = r.get("variant")
        scores = r.get("scores", {})
        for a, obj in scores.items():
            val = obj["score"] if isinstance(obj, dict) and "score" in obj else obj
            rows.append({"entity": e, "channel": ch, "variant": v, "attribute": a, "score": float(val)})
    df = pd.DataFrame(rows)
    # Clean potential weirdness
    if not df.empty:
        df = df.dropna(subset=["score"])
        df = df[(df["score"]>=0) & (df["score"]<=1)]
    return df

def _channel_attribute_medians(df_long: pd.DataFrame) -> pd.DataFrame:
    """Return a Channel×Attribute pivot of medians in [0,1]."""
    if df_long.empty:
        return pd.DataFrame()
    piv = (
        df_long
        .groupby(["channel", "attribute"], as_index=False)["score"]
        .median()
        .pivot(index="channel", columns="attribute", values="score")
        .sort_index()
    )
    return piv

def _attribute_correlation(df_long: pd.DataFrame) -> pd.DataFrame:
    """Compute attribute correlation matrix across creatives (entity×channel×variant rows)."""
    if df_long.empty:
        return pd.DataFrame()
    wide = (
        df_long
        .pivot_table(index=["entity","channel","variant"], columns="attribute", values="score", aggfunc="median")
        .sort_index()
    )
    if wide.empty:
        return pd.DataFrame()
    # Drop attributes with zero variance
    non_const = wide.loc[:, wide.std(numeric_only=True) > 1e-12]
    if non_const.shape[1] < 2:
        return pd.DataFrame()
    corr = non_const.corr(numeric_only=True)
    return corr

# ============================
# UI renderers
# ============================
def render_heatmap(df_long: pd.DataFrame, use_brand: bool=True, show_values: bool=True, decimals: int=2, sort_by_attr: str|None=None):
    piv = _channel_attribute_medians(df_long)
    if piv.empty:
        st.info("No data to show yet.")
        return piv

    if sort_by_attr and sort_by_attr in piv.columns:
        piv = piv.sort_values(by=sort_by_attr, ascending=False)

    z = piv.values
    x = list(piv.columns)
    y = list(piv.index)
    text = np.vectorize(lambda v: f"{v:.{decimals}f}")(z) if show_values else None

    fig = px.imshow(
        z, x=x, y=y, zmin=0, zmax=1,
        text_auto=False,  # we add text via update for better control
        color_continuous_scale=(BRAND_SEQ if use_brand else "RdYlGn"),
        aspect="auto"
    )
    if show_values:
        fig.update_traces(text=text, texttemplate="%{text}", textfont_size=12)
    fig.update_layout(
        margin=dict(l=60, r=30, t=30, b=60),
        coloraxis_colorbar=dict(title="Median score", tickformat=".2f")
    )
    st.plotly_chart(fig, use_container_width=True)
    return piv

def render_correlation(df_long: pd.DataFrame, use_brand: bool=True):
    corr = _attribute_correlation(df_long)
    if corr.empty:
        st.info("Not enough variation to compute correlations (need ≥2 attributes with non-zero variance).")
        return corr

    fig = px.imshow(
        corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, color_continuous_scale=(BRAND_DIVERGING if use_brand else "RdBu_r"),
        text_auto=True, aspect="auto"
    )
    fig.update_layout(
        margin=dict(l=60, r=30, t=30, b=60),
        coloraxis_colorbar=dict(title="Corr", tickformat=".2f")
    )
    st.plotly_chart(fig, use_container_width=True)
    return corr

def render_top_pairs(corr: pd.DataFrame, k:int=5):
    """Show top-k positive and negative attribute pairs from a correlation matrix."""
    if corr.empty:
        return
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    dfp = pd.DataFrame(pairs, columns=["Attribute A", "Attribute B", "Correlation"]).sort_values("Correlation", ascending=False)
    pos = dfp.head(k).reset_index(drop=True)
    neg = dfp.tail(k).sort_values("Correlation").reset_index(drop=True)

    st.markdown("#### Strongest positive relationships")
    st.dataframe(pos.style.format({"Correlation": "{:.2f}"}), use_container_width=True)
    st.markdown("#### Strongest negative relationships")
    st.dataframe(neg.style.format({"Correlation": "{:.2f}"}), use_container_width=True)

# ============================
# Page/Section entrypoint (safe to import from a larger app)
# ============================
def render_section():
    # 1) Ensure data
    _seed_demo_records()

    # 2) Load and flatten
    records = st.session_state.get("score_records", [])
    df = _records_to_long_df(records)

    # 3) Filters
    st.markdown("### Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        ent_sel = st.multiselect("Entities", sorted(df["entity"].dropna().unique().tolist()), default=sorted(df["entity"].dropna().unique().tolist()))
    with col2:
        ch_sel  = st.multiselect("Channels", sorted(df["channel"].dropna().unique().tolist()), default=sorted(df["channel"].dropna().unique().tolist()))
    with col3:
        var_sel = st.multiselect("Variants", sorted(df["variant"].dropna().unique().tolist()), default=sorted(df["variant"].dropna().unique().tolist()))

    df_filt = df[
        df["entity"].isin(ent_sel) &
        df["channel"].isin(ch_sel) &
        df["variant"].isin(var_sel)
    ].copy()

    # 4) Options
    st.markdown("### Display Options")
    c1, c2, c3, c4 = st.columns([1,1,1,2])
    with c1:
        use_brand = st.toggle("Use brand colors", value=True)
    with c2:
        show_vals = st.toggle("Show values", value=True)
    with c3:
        decimals = st.number_input("Decimals", 0, 4, 2, step=1)
    with c4:
        piv_preview = _channel_attribute_medians(df_filt)
        opts = ["(none)"] + (list(piv_preview.columns) if not piv_preview.empty else [])
        sba = st.selectbox("Sort channels by attribute", opts, index=0)
        sort_attr = None if sba == "(none)" else sba

    # 5) Heatmap
    st.markdown("## Channel × Attribute Heatmap")
    piv = render_heatmap(df_filt, use_brand=use_brand, show_values=show_vals, decimals=int(decimals), sort_by_attr=sort_attr)

    # Downloads (heatmap data)
    if not piv.empty:
        st.download_button(
            "Download heatmap data (CSV)",
            data=piv.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="channel_attribute_medians.csv",
            mime="text/csv"
        )

    # 6) Correlation explorer
    st.markdown("## Attribute Correlation Explorer")
    corr = render_correlation(df_filt, use_brand=use_brand)

    # Downloads (correlation)
    if not corr.empty:
        st.download_button(
            "Download correlation matrix (CSV)",
            data=corr.reset_index().rename(columns={"index":"attribute"}).to_csv(index=False).encode("utf-8"),
            file_name="attribute_correlation.csv",
            mime="text/csv"
        )
        render_top_pairs(corr, k=5)

# If this file is executed directly via `streamlit run`, render the section on an empty page.
def _main():
    st.set_page_config(page_title="Heatmaps & Correlations", layout="wide")
    st.title("The Marketing Scientist · Heatmaps & Correlations")
    render_section()

if __name__ == "__main__":
    _main()
