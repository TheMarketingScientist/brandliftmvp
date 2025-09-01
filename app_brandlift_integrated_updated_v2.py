
# app_brandlift_integrated.py
# Fully integrated Brand Lift demo app:
# - Scoring, improvement, competitor comparison, radar charts (from base app)
# - Attribute Importance Heatmap (median by channel, with filters)
# - Channel Trends Over Time (12‑month synthetic demo trends with filters)
# - Attribute Correlation Explorer (correlation matrix + top pairs + CSV downloads)
#
# Notes:
# * Uses The Marketing Scientist brand palette.
# * Keeps your existing session_state["score_records"] schema.
# * Displays labels nicely (underscores -> spaces) while keeping internal keys stable.

import json
import random
import httpx
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import re

# ---------- Brand Theme ----------
BRAND_BLUE   = "#445DA7"  # Original
BRAND_PURPLE = "#6B3894"  # Improved
BRAND_NAVY   = "#2E3C71"
COMP_TEAL    = "#2AA9A1"  # Competitor

# Sequential scale for 0→1 heatmaps (low→high) using brand colors
BRAND_SEQ = [BRAND_PURPLE, BRAND_BLUE, BRAND_NAVY]

# Diverging scale for correlations (−1→0→+1)
BRAND_DIVERGING = [
    (0.0, BRAND_PURPLE),   # negative
    (0.5, "#E9E7F4"),      # neutral
    (1.0, BRAND_NAVY),     # positive
]

# ---------- Config ----------
TITLE_COLOR = BRAND_NAVY  # change to BRAND_BLUE or BRAND_PURPLE if you prefer

# EXACT modern channels as requested
CHANNELS = ["CTV", "DOOH", "Youtube", "TikTok", "Google Ads", "Instagram", "X"]

# Internal attribute keys (stable). UI shows prettified labels with spaces/slashes.
ATTRS = ["Leadership", "Ease_of_Use", "Quality", "Luxury", "Cost_Benefit", "Trust"]

# Pretty label mapping for display
def _pretty_attr(a: str) -> str:
    return a.replace("_", " ").replace("Benefit", "Benefit").replace("Cost Benefit", "Cost/Benefit")

PREFERRED_MODEL = "claude-3-5-sonnet-latest"
FALLBACK_MODEL  = "claude-3-5-haiku-latest"
API_BASE = "https://api.anthropic.com/v1/messages"

SYSTEM_SCORE = """You are a brand perception rater. Score ad copy on 6 attributes:
Leadership, Ease_of_Use, Quality, Luxury, Cost_Benefit, Trust.
Each score is a float in [0,1]. Anchors: 0.2 = weak, 0.5 = moderate, 0.8 = strong.
Return STRICT JSON with each attribute as {"score": float, "evidence": "short phrase (<=12 words)"}.
No extra text.
"""
SYSTEM_REWRITE = "You are a precise brand copy editor focused on targeted brand attributes."
SYSTEM_IDEA = "You are a senior creative strategist. Propose concise alternative ad copy ideas."

# ---------------- Helpers ----------------
def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            k: {
                "type": "object",
                "properties": {"score": {"type": "number"}, "evidence": {"type": "string"}},
                "required": ["score", "evidence"],
            }
            for k in ATTRS
        },
        "required": ATTRS,
        "additionalProperties": False,
    }

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else ""
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()

def _extract_json_substring(text: str) -> str | None:
    text = _strip_code_fences(text)
    try:
        json.loads(text); return text
    except Exception:
        pass
    start_idx = None
    for i, ch in enumerate(text):
        if ch in '{[':
            start_idx = i; break
    if start_idx is None:
        return None
    stack = [text[start_idx]]
    i = start_idx + 1
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch in '{[': stack.append(ch)
            elif ch in '}]':
                if not stack: return None
                top = stack[-1]
                if (top == '{' and ch == '}') or (top == '[' and ch == ']'):
                    stack.pop()
                    if not stack:
                        candidate = text[start_idx:i+1]
                        try:
                            json.loads(candidate); return candidate
                        except Exception:
                            pass
                else:
                    return None
        i += 1
    return None

def _parse_json_block(text: str) -> dict:
    if not isinstance(text, str):
        raise RuntimeError("Model did not return text content.")
    text = (
        text.replace("\\u201c", '"')
            .replace("\\u201d", '"')
            .replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
    )
    candidate = _extract_json_substring(text)
    if candidate is None:
        raise RuntimeError("Model did not return JSON. Output was: " + text[:400])
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        jlike = re.sub(r"(?<!\\\\)'", '\"', candidate)
        try:
            return json.loads(jlike)
        except Exception:
            raise RuntimeError(f"Could not parse JSON. Error: {e}. Output was: {text[:400]}")


# ---- Attribute-tag cleaning & target-based rewriting ----
def _clean_attribute_tags(text: str) -> str:
    """
    Remove any inline attribute tags like [Leadership:0.8], (Quality +++), {Trust}, or "Leadership: 0.7"
    so the model focuses on natural copy, not explicit labels.
    """
    # Build attribute alternation
    attrs_alt = "|".join([re.escape(a) for a in ATTRS] + [re.escape(a.replace("_"," ")) for a in ATTRS])
    # Patterns: [Attr:...], (Attr ...), {Attr ...}
    text = re.sub(rf"\[({attrs_alt})\s*:[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"\((?:{attrs_alt})[^\)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"\{{(?:{attrs_alt})[^\}}]*\}}", "", text, flags=re.IGNORECASE)
    # Prefix labels like "Leadership: ..." or "Quality - ..."
    text = re.sub(rf"\b({attrs_alt})\s*[:\-–]\s*.*?$", "", text, flags=re.IGNORECASE|re.MULTILINE)
    # Collapse extra spaces
    return re.sub(r"\s{2,}", " ", text).strip()

def rewrite_copy_to_targets(api_key: str, text: str, target_levels: dict) -> str:
    """
    Rewrite text to *aim* for numeric target levels for each attribute in [0,1].
    Avoid explicit attribute tags in the output.
    """
    # Ensure keys are internal (with underscores). Accept pretty keys too.
    norm_targets = {}
    for k, v in target_levels.items():
        kk = k.replace(" ", "_").replace("/", "_")
        if kk not in ATTRS and kk.title() in ATTRS:
            kk = kk.title()
        norm_targets[kk] = max(0.0, min(1.0, float(v)))

    # Present targets in a compact JSON-like string for the model
    targets_str = ", ".join([f"{k}:{v:.2f}" for k, v in norm_targets.items()])

    cleaned = _clean_attribute_tags(text)
    example = json.dumps({"rewrite": "..."}, ensure_ascii=False)
    instr = f"""Rewrite the ad copy to better align perceived brand attributes with these targets (0–1):
{targets_str}

Guidelines:
- Keep the meaning and promises credible.
- Use natural language; DO NOT include explicit attribute labels or tags.
- Max 90 words. Return JSON: {example}

Original (cleaned of tags):
{cleaned}"""

    def run(model, api_key_inner, prompt_text):
        out_text = _call_messages(api_key_inner, SYSTEM_REWRITE, prompt_text, model, temperature=0.5, max_tokens=450)
        try:
            data = _parse_json_block(out_text)
            candidate = _clean_attribute_tags(data.get("rewrite", "").strip())
            if candidate:
                return candidate
        except Exception:
            pass
        clean = _clean_attribute_tags(_strip_code_fences(out_text).strip())
        if clean:
            words = clean.split()
            return " ".join(words[:90])
        return cleaned

    rewrite, _ = _with_fallback(run, api_key, instr)
    return rewrite or cleaned

def _call_messages(api_key: str, system: str, user_content: str, model: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(API_BASE, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code} from Anthropic ({model}) — {e.response.text}")
    data = r.json()
    blocks = data.get("content", [])
    texts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text" and "text" in b:
            texts.append(b["text"])
    return "\\n".join(texts) if texts else (blocks[0]["text"] if blocks and isinstance(blocks[0], dict) and "text" in blocks[0] else "")

def _with_fallback(func, *args, **kwargs):
    try:
        return func(PREFERRED_MODEL, *args, **kwargs), PREFERRED_MODEL
    except RuntimeError as e:
        s = str(e)
        if ("HTTP 403" in s) or ("HTTP 404" in s) or ("not_found" in s) or ("model not found" in s):
            return func(FALLBACK_MODEL, *args, **kwargs), FALLBACK_MODEL
        raise

def score_text(api_key: str, text: str) -> dict:
    user_content = f"""Text:
{text}

Output JSON schema:
{json.dumps(_schema())}"""
    def run(model, api_key_inner, uc):
        out_text = _call_messages(api_key_inner, SYSTEM_SCORE, uc, model)
        scores = _parse_json_block(out_text)
        for k in ATTRS:
            s = float(scores[k]["score"])
            scores[k]["score"] = max(0.0, min(1.0, s))
        return scores
    scores, _ = _with_fallback(run, api_key, user_content)
    return scores

def rewrite_copy(api_key: str, text: str, targets: list[str]) -> str:
    t = ", ".join(targets) if targets else "Leadership"
    example = json.dumps({"rewrite": "..."}, ensure_ascii=False)
    instr = f"""Rewrite the text to increase the perception of: {t}.
Preserve original meaning, keep tone professional, avoid hype. Max 70 words.
Return JSON: {example}

Original:
{text}"""
    def run(model, api_key_inner, prompt_text):
        out_text = _call_messages(api_key_inner, SYSTEM_REWRITE, prompt_text, model, temperature=0.4, max_tokens=400)
        try:
            data = _parse_json_block(out_text)
            candidate = data.get("rewrite", "").strip()
            if candidate:
                return candidate
        except Exception:
            pass
        clean = _strip_code_fences(out_text).strip()
        if clean:
            words = clean.split()
            return " ".join(words[:70])
        return text
    rewrite, _ = _with_fallback(run, api_key, instr)
    return rewrite or text

def propose_new_ideas(api_key: str, base_text: str, targets: list[str], n: int = 2) -> list[str]:
    t = ", ".join(targets) if targets else "Leadership"
    example = json.dumps({"ideas": ["...", "..."]}, ensure_ascii=False)
    prompt = f"""Given this ad copy, propose {n} distinct alternative ideas that could outperform a competitor
on the following target attributes: {t}.
Keep each idea under 35 words. Return JSON: {example}

Original:
{base_text}"""
    def run(model, api_key_inner, uc):
        out_text = _call_messages(api_key_inner, SYSTEM_IDEA, uc, model, temperature=0.6, max_tokens=500)
        try:
            data = _parse_json_block(out_text)
            ideas = data.get("ideas", [])
            return [i.strip() for i in ideas if isinstance(i, str) and i.strip()]
        except Exception:
            lines = [l.strip("-• ").strip() for l in _strip_code_fences(out_text).splitlines() if l.strip()]
            return [l for l in lines if l][:n]
    ideas, _ = _with_fallback(run, api_key, prompt)
    return [i.strip() for i in ideas if isinstance(i, str) and i.strip()]

# ---- Competitor random generation ----
COMP_PHRASES = {
    "Leadership": ["market-leading claims", "award-winning reputation", "trend-setting positioning", "recognized authority"],
    "Ease_of_Use": ["simple onboarding cues", "clear action verbs", "one-step instructions", "low-friction setup"],
    "Quality": ["durability emphasis", "precision language", "rigorous testing mention", "craftsmanship cues"],
    "Luxury": ["exclusive tone", "premium finish cues", "elevated aesthetic", "boutique language"],
    "Cost_Benefit": ["value-centric framing", "ROI emphasis", "savings mention", "high benefit per cost"],
    "Trust": ["transparent terms", "secure service", "credible endorsements", "proven track record"],
}
def random_competitor_scores(seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)
    scores = {}
    for k in ATTRS:
        val = round(random.uniform(0.25, 0.9), 2)
        phrase = random.choice(COMP_PHRASES.get(k, ["balanced framing"]))
        scores[k] = {"score": val, "evidence": phrase}
    return scores

def radar(scores: dict, title: str, fill_color: str, line_color: str):
    labels = [_pretty_attr(k) for k in ATTRS]
    vals = [scores[k]["score"] for k in ATTRS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + vals[:1],
        theta=labels + labels[:1],
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        name=title
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(range=[0, 1])),
        showlegend=False,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig

def scores_table(scores: dict) -> pd.DataFrame:
    rows = []
    for k in ATTRS:
        rows.append({
            "Attribute": _pretty_attr(k),
            "Score": round(float(scores[k]["score"]), 2),
            "Key phrase": scores[k]["evidence"]
        })
    return pd.DataFrame(rows)

def delta_table(base: dict, improved: dict) -> pd.DataFrame:
    rows = []
    for k in ATTRS:
        b = float(base[k]["score"])
        i = float(improved[k]["score"])
        rows.append({
            "Attribute": _pretty_attr(k),
            "Original": round(b, 2),
            "Improved": round(i, 2),
            "Δ": round(i - b, 2),
        })
    return pd.DataFrame(rows)

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
    .warn-box {{
        background: #FFF4CC; border: 1px solid #E2C268; border-radius: 8px;
        padding: 0.6rem; margin-top: 0.5rem; margin-bottom: 0.5rem;
    }}
    .warn-title {{ font-weight: 700; color: {BRAND_NAVY}; margin-bottom: 0.25rem; }}
    .pill-warn {{
        display:inline-block; padding: 0.2rem 0.5rem; border-radius: 999px;
        background: white; border: 1px solid #E2C268; color: {BRAND_NAVY};
        font-size: 0.85rem; margin-right: 0.35rem; margin-bottom: 0.35rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------- Score record utilities (for Heatmap) -------
def _init_records():
    if "score_records" not in st.session_state:
        st.session_state["score_records"] = []  # list of dicts: {entity, channel, variant, scores}

def _upsert_record(entity: str, channel: str, variant: str, scores: dict):
    _init_records()
    key = (entity, channel, variant)
    for r in st.session_state["score_records"]:
        if (r["entity"], r["channel"], r["variant"]) == key:
            r["scores"] = scores
            break
    else:
        st.session_state["score_records"].append({"entity": entity, "channel": channel, "variant": variant, "scores": scores})

def _records_to_channel_attr_medians(entities: list[str] | None = None, variants: list[str] | None = None) -> pd.DataFrame:
    _init_records()
    rows = []
    for r in st.session_state["score_records"]:
        if entities and r["entity"] not in entities: continue
        if variants and r["variant"] not in variants: continue
        channel = r["channel"]
        sc = r["scores"]
        for attr in ATTRS:
            rows.append({"Channel": channel, "Attribute": _pretty_attr(attr), "Score": float(sc[attr]["score"])})
    if not rows:
        return pd.DataFrame(columns=["Channel"] + [_pretty_attr(a) for a in ATTRS])
    df = pd.DataFrame(rows)
    med = df.groupby(["Channel", "Attribute"], as_index=False)["Score"].median()
    pivot = med.pivot(index="Channel", columns="Attribute", values="Score").reindex(CHANNELS, fill_value=None)
    pivot = pivot[[_pretty_attr(a) for a in ATTRS]]
    return pivot.reset_index()

def _heatmap(fig_df: pd.DataFrame, title: str = "Attribute Importance Heatmap"):
    if fig_df.empty or len(fig_df.columns) <= 1:
        st.info("Not enough scored items to build a heatmap yet. Score at least one item.")
        return
    z = fig_df.drop(columns=["Channel"]).values
    x = list(fig_df.columns[1:])
    y = list(fig_df["Channel"])
    hm = go.Figure(data=go.Heatmap(colorscale=[
        [0.0, 'rgb(220,20,60)'],
        [0.5, 'rgb(255,215,0)'],
        [1.0, 'rgb(34,139,34)']
    ],
        z=z, x=x, y=y, zmin=0.0, zmax=1.0,
        colorbar=dict(title="Median Score"),
        hovertemplate="Channel: %{y}<br>Attribute: %{x}<br>Median: %{z:.2f}<extra></extra>"
    ))
    hm.update_layout(title=title, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(hm, use_container_width=True)
    st.dataframe(fig_df, use_container_width=True)

# ------------- Correlation Explorer (integrated) -------------
def _records_to_long_df(records: list[dict]) -> pd.DataFrame:
    """Flatten into long form with pretty labels for attributes (for consistent UI)."""
    rows = []
    for r in records:
        e = r.get("entity")
        ch = r.get("channel")
        v = r.get("variant")
        scores = r.get("scores", {})
        for a in ATTRS:
            obj = scores.get(a, {"score": None})
            val = obj["score"] if isinstance(obj, dict) and "score" in obj else obj
            rows.append({"entity": e, "channel": ch, "variant": v, "attribute": _pretty_attr(a), "score": float(val) if val is not None else None})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["score"])
        df = df[(df["score"]>=0) & (df["score"]<=1)]
    return df

def _attribute_correlation(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    wide = (
        df_long
        .pivot_table(index=["entity","channel","variant"], columns="attribute", values="score", aggfunc="median")
        .sort_index()
    )
    if wide.empty:
        return pd.DataFrame()
    non_const = wide.loc[:, wide.std(numeric_only=True) > 1e-12]
    if non_const.shape[1] < 2:
        return pd.DataFrame()
    corr = non_const.corr(numeric_only=True)
    return corr

def render_correlation_section():
    import pandas as pd
    import plotly.express as px

    st.subheader("Attribute Correlation Explorer")

    # Build a long df from score_records if not already present
    df_long = st.session_state.get("score_df_long")
    if df_long is None:
        recs = st.session_state.get("score_records", [])
        if not recs:
            st.info("No scores yet. Score some copies to see correlations.")
            return
        rows = []
        for r in recs:
            ent = r.get("entity", "Client")
            ch = r.get("channel", "TV")
            var = r.get("variant", "Original")
            scores = r.get("scores", {})
            for a in ATTRS:
                val = None
                if isinstance(scores, dict) and a in scores and isinstance(scores[a], dict):
                    val = scores[a].get("score")
                if val is not None:
                    rows.append({
                        "entity": ent,
                        "channel": ch,
                        "variant": var,
                        "attribute": _pretty_attr(a),
                        "score": float(val),
                    })
        df_long = pd.DataFrame(rows)

    corr = _attribute_correlation(df_long)
    if corr.empty:
        st.info("Not enough attribute variation yet to compute correlations.")
        return

    fig = px.imshow(
        corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        zmin=-1, zmax=1,
        color_continuous_scale=BRAND_DIVERGING,
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        margin=dict(l=60, r=30, t=30, b=60),
        coloraxis_colorbar=dict(title="Corr", tickformat=".2f")
    )
    st.plotly_chart(fig, use_container_width=True)

def _plot_channel_trends(df_channel: pd.DataFrame):
    fig = go.Figure()
    for attr in sorted(df_channel["Attribute"].unique()):
        sdf = df_channel[df_channel["Attribute"] == attr].sort_values("Month")
        fig.add_trace(go.Scatter(
            x=sdf["Month"],
            y=sdf["Score"],
            mode="lines+markers",
            name=attr
        ))
    fig.update_layout(
        title="Attribute trends (12 months)",
        xaxis_title="Month",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=40, r=20, t=60, b=40),
        legend_title="Attribute"
    )
    st.plotly_chart(fig, use_container_width=True)
    wide = df_channel.pivot(index="Month", columns="Attribute", values="Score").reset_index()
    st.dataframe(wide, use_container_width=True)

_seed_demo_trends()

trend_channel = st.selectbox("Select channel", CHANNELS, index=0, key="trend_channel")
trend_df = st.session_state["monthly_attr_trends"]

try:
    ef = set(sel_entities) if sel_entities else set(sorted(trend_df["Entity"].unique()))
    vf = set(sel_variants) if sel_variants else set(sorted(trend_df["Variant"].unique()))
    filtered = trend_df[(trend_df["Entity"].isin(ef)) & (trend_df["Variant"].isin(vf)) & (trend_df["Channel"] == trend_channel)]
except NameError:
    filtered = trend_df[trend_df["Channel"] == trend_channel]

if not filtered.empty:
    agg = (filtered.groupby(["Month", "Attribute"], as_index=False)["Score"].median())
else:
    agg = filtered.copy()

_plot_channel_trends(agg)

# ---------------- Attribute Correlation Explorer ----------------
render_correlation_section()
