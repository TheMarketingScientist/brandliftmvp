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
# --- Auth helpers (added) ---
import os
from typing import Optional


def _is_spanish(text: str) -> bool:
    """
    Lightweight Spanish detector: looks for common Spanish function words and accented letters.
    Avoids external deps.
    """
    if not text:
        return False
    text_l = text.lower()
    hits = 0
    # Common function words and patterns
    tokens = [
        " el ", " la ", " los ", " las ", " de ", " del ", " y ", " o ",
        " que ", " como ", " para ", " con ", " sin ", " por ", " más ", " menos ",
        " también ", " aún ", " aún ", " pero ", " porque ", " sobre ",
        " cliente ", " marca ", " producto ", " servicio ", " experiencia "
    ]
    for t in tokens:
        if t in " " + text_l + " ":
            hits += 1
    # Accented characters typical in Spanish
    if re.search(r"[áéíóúñüÁÉÍÓÚÑÜ]", text):
        hits += 2
    return hits >= 3

def _looks_like_transcript(text: str) -> bool:
    """
    Detects transcript-like structure by presence of frequent timestamps or speaker tags.
    """
    if not text:
        return False
    # Timestamp patterns like [00:12], 00:12:34, (00:12), 0:59, [0:59:02]
    ts_count = len(re.findall(r"(\\[?\\b\\d{1,2}:\\d{2}(?::\\d{2})?\\]?|\\(\\d{1,2}:\\d{2}(?::\\d{2})?\\))", text))
    # Speaker lines like "Speaker:" or "HOST:"
    sp_count = len(re.findall(r"^[A-Z][A-Za-z0-9_ ]{0,20}:", text, flags=re.MULTILINE))
    # Multiple short lines suggest transcript
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    short_lines = sum(1 for ln in lines if len(ln) <= 140)
    return (ts_count >= 2) or (sp_count >= 3) or (short_lines >= 8 and len(lines) >= 10)



def _auth_mode() -> str:
    # "PASSWORD" (default) or "AUTH0"
    try:
        return st.secrets.get("AUTH_MODE", "PASSWORD").upper().strip()
    except Exception:
        return "PASSWORD"

def _auth_reset():
    for k in ["authed", "auth_user", "client_name", "company_logo_url", "preload_demo"]:
        if k in st.session_state:
            del st.session_state[k]

def _brand_login_header():
    # Brand header on login screen
    logo_url = st.secrets.get("COMPANY_LOGO_URL", "").strip()
    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            {'<img src="'+logo_url+'" style="height: 140px; width: auto;">' if logo_url else ''}
            <h1 style="margin:0; color:{TITLE_COLOR};">Brand Lift</h1>
        </div>
    """, unsafe_allow_html=True)
    st.caption("Please sign in to continue.")

def _render_login_password() -> bool:
    _brand_login_header()
    with st.container(border=True):
        st.subheader("Password Login")
        username = st.text_input("Username", value=st.session_state.get("last_username",""))
        pw = st.text_input("Access password", type="password", help="Set APP_PASSWORD in Secrets. Test user also available.")
        client = st.text_input("Client name (will show on the dashboard)", value=st.session_state.get("client_name",""))
        logo_url = st.text_input("Company logo URL (optional PNG/JPG/SVG)", value=st.session_state.get("company_logo_url",""))
        colA, colB = st.columns([1,1])
        with colA:
            login_click = st.button("Sign in", use_container_width=True)
        with colB:
            st.write("")

    if login_click:
        st.session_state["last_username"] = username.strip()
        # Test user hardcoded (preload demo data only for this user)
        if username.strip() == "TheMarketingScientist" and pw == "TheMarketingScientist":
            st.session_state["authed"] = True
            st.session_state["auth_user"] = "TheMarketingScientist"
            st.session_state["client_name"] = client.strip() or "Demo Client"
            if logo_url.strip():
                st.session_state["company_logo_url"] = logo_url.strip()
            elif st.secrets.get("COMPANY_LOGO_URL", "").strip():
                st.session_state["company_logo_url"] = st.secrets.get("COMPANY_LOGO_URL").strip()
            st.session_state["preload_demo"] = True  # only for test user
            return True

        # Standard password auth
        expected = st.secrets.get("APP_PASSWORD", "").strip()
        if not expected:
            st.error("Server missing APP_PASSWORD in Secrets. Alternatively use the test user.")
            return False
        if pw == expected:
            st.session_state["authed"] = True
            st.session_state["auth_user"] = username.strip() or "password-user"
            st.session_state["client_name"] = client.strip() or "Client"
            if logo_url.strip():
                st.session_state["company_logo_url"] = logo_url.strip()
            elif st.secrets.get("COMPANY_LOGO_URL", "").strip():
                st.session_state["company_logo_url"] = st.secrets.get("COMPANY_LOGO_URL").strip()
            st.session_state["preload_demo"] = False  # do NOT preload for non-test users
            return True
        else:
            st.error("Invalid password.")
    return False

def _render_login_auth0() -> bool:
    """
    Minimal Auth0 Lite-style flow using streamlit-auth0-component (optional dependency).
    Secrets needed:
      AUTH0_DOMAIN, AUTH0_CLIENT_ID, AUTH0_AUDIENCE (optional), AUTH0_REDIRECT_URI (optional)
    """
    _brand_login_header()
    try:
        from st_auth0_component import login_button  # pip install streamlit-auth0-component
    except Exception:
        st.warning("Auth0 mode selected, but 'streamlit-auth0-component' is not installed. Falling back to password.")
        return _render_login_password()

    domain = st.secrets.get("AUTH0_DOMAIN", "")
    client_id = st.secrets.get("AUTH0_CLIENT_ID", "")
    redirect_uri = st.secrets.get("AUTH0_REDIRECT_URI", "")
    audience = st.secrets.get("AUTH0_AUDIENCE", "")

    with st.container(border=True):
        st.subheader("Sign in with Auth0")
        token = login_button(domain=domain, client_id=client_id, redirect_uri=redirect_uri, audience=audience)
        client = st.text_input("Client name (will show on the dashboard)", value=st.session_state.get("client_name",""))
        logo_url = st.text_input("Company logo URL (optional PNG/JPG/SVG)", value=st.session_state.get("company_logo_url",""))

    if token:
        st.session_state["authed"] = True
        st.session_state["auth_user"] = "auth0-user"
        st.session_state["client_name"] = client.strip() or "Client"
        if logo_url.strip():
            st.session_state["company_logo_url"] = logo_url.strip()
        elif st.secrets.get("COMPANY_LOGO_URL", "").strip():
            st.session_state["company_logo_url"] = st.secrets.get("COMPANY_LOGO_URL").strip()
        st.session_state["preload_demo"] = False  # do NOT preload for Auth0 by default
        return True
    return False

def auth_gate() -> bool:
    """
    Renders a login screen if not authenticated.
    Returns True if authenticated, otherwise False.
    """
    if st.session_state.get("authed"):
        return True

    mode = _auth_mode()
    ok = False
    with st.container():
        if mode == "AUTH0":
            ok = _render_login_auth0()
        else:
            ok = _render_login_password()
    return ok


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
            elif ch == '\\\\': esc = True
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
        text.replace("\\\\u201c", '"')
            .replace("\\\\u201d", '"')
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
        jlike = re.sub(r"(?<!\\\\\\\\)'", '\"', candidate)
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
    text = re.sub(rf"\\[({attrs_alt})\\s*:[^\\]]*\\]", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"\\((?:{attrs_alt})[^\\)]*\\)", "", text, flags=re.IGNORECASE)
    text = re.sub(rf"\\{{(?:{attrs_alt})[^\\}}]*\\}}", "", text, flags=re.IGNORECASE)
    # Prefix labels like "Leadership: ..." or "Quality - ..."
    text = re.sub(rf"\\b({attrs_alt})\\s*[:\\-–]\\s*.*?$", "", text, flags=re.IGNORECASE|re.MULTILINE)
    # Collapse extra spaces
    return re.sub(r"\\s{2,}", " ", text).strip()


def rewrite_copy_to_targets(api_key: str, text: str, target_levels: dict) -> str:
    """
    Rewrite text or transcript to *aim* for numeric target levels for each attribute in [0,1].
    - If input is Spanish, respond in Spanish.
    - If input looks like a transcript, return a transcript with timestamps/speaker labels preserved.
    - Avoid explicit attribute tags in the output.
    """
    # Normalize targets
    norm_targets = {}
    for k, v in target_levels.items():
        kk = k.replace(" ", "_").replace("/", "_")
        if kk not in ATTRS and kk.title() in ATTRS:
            kk = kk.title()
        try:
            val = float(v)
        except Exception:
            val = 0.5
        norm_targets[kk] = max(0.0, min(1.0, val))
    targets_str = ", ".join([f"{k}:{v:.2f}" for k, v in norm_targets.items()])

    cleaned = _clean_attribute_tags(_strip_code_fences(text or "")).strip()
    if not cleaned:
        return ""

    # Detect language and structure
    is_es = _is_spanish(cleaned)
    is_tx = _looks_like_transcript(cleaned)

    # Build user instruction
    lang_line = "Responde en Español." if is_es else "Respond in English."
    if is_tx:
        structure = (
            "This is a transcript. Improve clarity, flow, and persuasion while preserving the transcript structure, "
            "timestamps, and speaker labels. If a timestamped line exists, keep timestamps and align them with the original granularity. "
            "Do not remove timestamps; you may adjust punctuation and phrasing only."
        )
    else:
        structure = (
            "This is marketing copy. Improve clarity, flow, and persuasion for the specified channel, "
            "keeping the same language and avoiding explicit attribute tags."
        )

    user_content = f"""{lang_line}
Aim the tone to these target attribute levels in [0,1]: {targets_str}.
{structure}

ORIGINAL:
{cleaned}

Return only the improved text (no explanations, no JSON, no tags)."""

    def run(model, api_key_inner, uc):
        out_text = _call_messages(api_key_inner, SYSTEM_REWRITE, uc, model, temperature=0.3, max_tokens=900)
        clean = _clean_attribute_tags(_strip_code_fences(out_text).strip())
        if not clean:
            return cleaned
        # Be conservative: keep length similar
        words = clean.split()
        if len(words) > 0 and len(cleaned.split()) > 0:
            cap = max(90, int(len(cleaned.split()) * 1.3))
            clean = " ".join(words[:cap])
        return clean

    rewrite, _ = _with_fallback(run, api_key, user_content)
    return rewrite or cleaned


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

def _records_to_channel_attr_medians(
    entities: list[str] | None = None,
    variants: list[str] | None = None
) -> pd.DataFrame:
    _init_records()
    rows = []
    for r in st.session_state["score_records"]:
        if entities and r["entity"] not in entities:
            continue
        if variants and r["variant"] not in variants:
            continue
        channel = r["channel"]
        sc = r["scores"]
        for attr in ATTRS:
            rows.append({
                "Channel": channel,
                "Attribute": _pretty_attr(attr),
                "Score": float(sc[attr]["score"])
            })

    if not rows:
        return pd.DataFrame(columns=["Channel"] + [_pretty_attr(a) for a in ATTRS])

    df = pd.DataFrame(rows)
    med = df.groupby(["Channel", "Attribute"], as_index=False)["Score"].median()
    pivot = med.pivot(index="Channel", columns="Attribute", values="Score").reset_index()
    return pivot


def _append_trend_rows(entity: str, channel: str, variant: str, scores: dict):
    """
    Append per-attribute trend rows into monthly_attr_trends using the CURRENT month.
    Only appends real data (no synthetic). Safe for repeated calls.
    """
    ts = pd.Timestamp.today().normalize()
    month = ts.strftime("%Y-%m")
    rows = []
    for attr in ATTRS:
        try:
            val = float(scores[attr]["score"])
        except Exception:
            continue
        rows.append({
            "Entity": entity,
            "Variant": variant,
            "Channel": channel,
            "Month": month,
            "Attribute": _pretty_attr(attr),
            "Score": max(0.0, min(1.0, val)),
        })
    df_new = pd.DataFrame(rows)
    if "monthly_attr_trends" in st.session_state and isinstance(st.session_state["monthly_attr_trends"], pd.DataFrame):
        st.session_state["monthly_attr_trends"] = pd.concat(
            [st.session_state["monthly_attr_trends"], df_new],
            ignore_index=True
        )
    else:
        st.session_state["monthly_attr_trends"] = df_new


def _heatmap(fig_df: pd.DataFrame, title: str = "Attribute Importance Heatmap"):
    if fig_df.empty or len(fig_df.columns) <= 1:
        st.info("Not enough scored items to build a heatmap yet. Score at least one item.")
        return
    z = fig_df.drop(columns=["Channel"]).values
    x = list(fig_df.columns[1:])
    y = list(fig_df["Channel"])
    hm = go.Figure(data=go.Heatmap(
        colorscale=[
            [0.0, 'rgb(128,0,128)'],   # purple (low)
            [0.5, 'rgb(255,255,255)'], # white (mid)
            [1.0, 'rgb(0,0,128)']      # navy (high)
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
    st.subheader("Attribute Correlation Explorer")
    st.caption("See how attributes move together across all scored items (entity × channel × variant).")

    _init_records()
    if not st.session_state["score_records"]:
        # Only preload demo for the test user
        if st.session_state.get('preload_demo'):
            _seed_full_demo_heatmap()  # ensure some demo data exists

    df_long = _records_to_long_df(st.session_state["score_records"])

    # If there are no scored records yet, show a friendly message and exit
    if df_long.empty:
        st.info("No scored items yet. Score at least one item to view correlations.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        ent_sel = st.multiselect("Entities", sorted(df_long["entity"].dropna().unique().tolist()), default=sorted(df_long["entity"].dropna().unique().tolist()))
    with col2:
        ch_sel  = st.multiselect("Channels", sorted(df_long["channel"].dropna().unique().tolist()), default=sorted(df_long["channel"].dropna().unique().tolist()))
    with col3:
        var_sel = st.multiselect("Variants", sorted(df_long["variant"].dropna().unique().tolist()), default=sorted(df_long["variant"].dropna().unique().tolist()))

    df_filt = df_long[
        df_long["entity"].isin(ent_sel) &
        df_long["channel"].isin(ch_sel) &
        df_long["variant"].isin(var_sel)
    ].copy()

    corr = _attribute_correlation(df_filt)
    if corr.empty:
        st.info("Not enough variation to compute correlations (need ≥2 attributes with non-zero variance).")
        return

    fig = px.imshow(
        corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, color_continuous_scale=BRAND_DIVERGING,
        text_auto=True, aspect="auto"
    )
    fig.update_layout(
        margin=dict(l=60, r=30, t=30, b=60),
        coloraxis_colorbar=dict(title="Corr", tickformat=".2f")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top pairs
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    dfp = pd.DataFrame(pairs, columns=["Attribute A", "Attribute B", "Correlation"]).sort_values("Correlation", ascending=False)
    pos = dfp.head(5).reset_index(drop=True)
    neg = dfp.tail(5).sort_values("Correlation").reset_index(drop=True)

    st.markdown("#### Strongest positive relationships")
    st.dataframe(pos.style.format({"Correlation": "{:.2f}"}), use_container_width=True)
    st.markdown("#### Strongest negative relationships")
    st.dataframe(neg.style.format({"Correlation": "{:.2f}"}), use_container_width=True)

    # Downloads
    st.download_button(
        "Download correlation matrix (CSV)",
        data=corr.reset_index().rename(columns={"index":"attribute"}).to_csv(index=False).encode("utf-8"),
        file_name="attribute_correlation.csv",
        mime="text/csv"
    )

# ---------------- UI ----------------
st.set_page_config(page_title="Brand Lift", layout="wide")
inject_css()

# --- Require auth before proceeding ---
if not auth_gate():
    st.stop()



hdr_left, hdr_right = st.columns([4, 1.2])
with hdr_left:
    client_name = st.session_state.get("client_name", "Client")
    st.markdown(f"<h1 class='app-title'>Brand Lift — {client_name}</h1>", unsafe_allow_html=True)
with hdr_right:
    logo_url = st.session_state.get("company_logo_url", st.secrets.get("COMPANY_LOGO_URL", "").strip())
    if logo_url:
        try:
            st.image(logo_url, use_container_width=True)
        except Exception:
            pass
    else:
        logo_path = Path("logo.png")
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

# Sidebar account controls
st.sidebar.markdown("### Account")
st.sidebar.caption(f"User: **{st.session_state.get('auth_user', '—')}**")
if st.sidebar.button("Log out"):
    _auth_reset()
    st.experimental_rerun()


api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
if not api_key:
    st.info('Add your key in Settings → Secrets as:\\nANTHROPIC_API_KEY = "sk-ant-..."')

st.markdown("**Attributes**")
st.markdown('<div class="badges">' + " ".join(
    [f'<span class="badge" style="background:{BRAND_NAVY}">{_pretty_attr("Leadership")}</span>',
     f'<span class="badge" style="background:{BRAND_BLUE}">{_pretty_attr("Ease_of_Use")}</span>',
     f'<span class="badge" style="background:{BRAND_PURPLE}">{_pretty_attr("Quality")}</span>',
     f'<span class="badge" style="background:#64348B">{_pretty_attr("Luxury")}</span>',
     f'<span class="badge" style="background:#2B386A">{_pretty_attr("Cost_Benefit")}</span>',
     f'<span class="badge" style="background:{COMP_TEAL}">{_pretty_attr("Trust")}</span>']
) + "</div>", unsafe_allow_html=True)

text = st.text_area("Paste ad copy or transcript:", height=160,
    placeholder="Example: Introducing the next-generation sedan—hand-finished interiors, seamless app control, and advanced safety.")

client_url = st.text_input("Client creative URL (optional)", placeholder="https://...")
client_channel = st.selectbox("Client channel", CHANNELS, index=0)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    score_clicked = st.button("Score", use_container_width=True)
# REMOVED the multiselect & inline Improve button from c2 as requested
with c2:
    pass
with c3:
    comp_cols = st.columns([1.2, 2.8])
    with comp_cols[0]:
        comp_name = st.text_input("Competitor label", value="Competitor A")
    with comp_cols[1]:
        comp_url = st.text_input("Competitor ad/creative URL (optional)", placeholder="https://...")
    comp_text = st.text_area("Competitor copy/transcript (optional)", height=120, placeholder="Paste competitor copy/transcript to score (optional)")
    comp_channel = st.selectbox("Competitor channel", CHANNELS, index=0)
    score_comp = st.button("Score Competitor", use_container_width=True)


# --- Desired Attribute Targets (sliders) ---
with st.expander("Desired attribute targets", expanded=True):
    desired_targets = {}
    cols = st.columns(3)
    for i, a in enumerate(ATTRS):
        with cols[i % 3]:
            pretty = _pretty_attr(a)
            # Default targets are moderate-strong to create a "wow" effect on first demo
            desired_targets[pretty] = st.slider(pretty, 0.0, 1.0, 0.78, 0.01, help="Target perceived level for this attribute")
    st.caption("These targets drive the improvement step. The rewrite will **avoid explicit attribute tags** and aim for these levels in tone and content.")

# MOVE the Improve button right below the sliders
improve_clicked = st.button("Improve & Rescore", use_container_width=True)


if score_clicked:
    if not api_key:
        st.error("No API key found in Secrets.")
    elif not client_channel:
        st.warning("Please select a channel before scoring.")
    elif not text.strip():
        st.warning("Please paste some text.")
    else:
        with st.spinner("Scoring…"):
            base_scores = score_text(api_key, text)
            st.session_state["base_text"] = text
            st.session_state["base_scores"] = base_scores
            st.session_state["client_channel"] = client_channel
            _upsert_record("Client", client_channel, "Original", base_scores)
        _append_trend_rows("Client", client_channel, "Original", base_scores)


if improve_clicked:
    source_text = (text.strip() or st.session_state.get("improved_text") or st.session_state.get("base_text", "")).strip()
    if not api_key:
        st.error("No API key found in Secrets.")
    elif not client_channel:
        st.warning("Please select a channel before scoring.")
    elif not source_text:
        st.warning("Paste text and click Score first.")
    else:
        # Build numeric targets from sliders
        with st.spinner("Generating improved copy to match targets…"):
            improved_text = rewrite_copy_to_targets(api_key, source_text, desired_targets)
        with st.spinner("Scoring improved copy…"):
            improved_scores = score_text(api_key, improved_text)
        st.session_state["improved_text"] = improved_text
        st.session_state["improved_scores"] = improved_scores
        if "base_scores" not in st.session_state and text.strip():
            st.session_state["base_text"] = text.strip()
            st.session_state["base_scores"] = score_text(api_key, st.session_state["base_text"])
            _upsert_record("Client", client_channel, "Original", st.session_state["base_scores"])
        _upsert_record("Client", client_channel, "Improved", improved_scores)
        _append_trend_rows("Client", client_channel, "Improved", improved_scores)

if "score_comp" in locals() and score_comp:
    st.session_state["competitor_name"] = comp_name or "Competitor"
    st.session_state["competitor_url"] = comp_url.strip()
    if not api_key:
        st.error("No API key found in Secrets.")
    elif not client_channel:
        st.warning("Please select a channel before scoring.")
    elif (comp_text or "").strip():
        if not comp_channel:
            st.warning("Please select a competitor channel before scoring.")
        else:
            with st.spinner("Scoring competitor…"):
                s = score_text(api_key, comp_text)
                st.session_state["competitor_scores"] = s
                _upsert_record(st.session_state["competitor_name"], comp_channel, "Competitor", s)
    else:
        st.warning("Paste competitor copy/transcript to score.")

base_scores = st.session_state.get("base_scores")
improved_scores = st.session_state.get("improved_scores")
comp_scores = st.session_state.get("competitor_scores")
comp_label = st.session_state.get("competitor_name", "Competitor")
comp_link  = st.session_state.get("competitor_url", "")

if base_scores or improved_scores or comp_scores:
    st.subheader("Comparison")
    cols_to_use = [bool(base_scores), bool(improved_scores), bool(comp_scores)]
    n = sum(cols_to_use) or 1
    cols = st.columns(n)

    idx = 0
    if base_scores:
        with cols[idx]:
            st.markdown("**Original Copy**")
            st.write(st.session_state.get("base_text",""))
            if client_url:
                st.markdown(f"[Open creative]({client_url})")
                try:
                    lower_cli = client_url.lower()
                    if lower_cli.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                        st.image(client_url, use_container_width=True)
                    elif lower_cli.endswith(('.mp4', '.webm', '.ogg')) or 'youtube.com' in lower_cli or 'youtu.be' in lower_cli or 'vimeo.com' in lower_cli:
                        st.video(client_url)
                except Exception:
                    pass
            st.session_state["client_channel"] = client_channel
            st.plotly_chart(radar(base_scores, "Original Scores", "rgba(68, 93, 167, 0.45)", "rgba(68, 93, 167, 1.0)"), use_container_width=True)
            st.dataframe(scores_table(base_scores), use_container_width=True)
        idx += 1

    if improved_scores:
        with cols[idx]:
            st.markdown("**Improved Copy**")
            st.write(st.session_state.get("improved_text",""))
            st.plotly_chart(radar(improved_scores, "Improved Scores", "rgba(107, 56, 148, 0.45)", "rgba(107, 56, 148, 1.0)"), use_container_width=True)
            st.dataframe(scores_table(improved_scores), use_container_width=True)
        idx += 1

    if comp_scores:
        with cols[idx]:
            st.markdown(f"**{comp_label}**")
            if comp_link:
                lower = comp_link.lower()
                st.markdown(f"[Open creative]({comp_link})")
                try:
                    if lower.endswith(('.png','.jpg','.jpeg','.webp','.gif')):
                        st.image(comp_link, use_container_width=True)
                    elif lower.endswith(('.mp4','.webm','.ogg')) or 'youtube.com' in lower or 'youtu.be' in lower or 'vimeo.com' in lower:
                        st.video(comp_link)
                except Exception:
                    pass
            st.plotly_chart(radar(comp_scores, f"{comp_label} Scores", "rgba(42, 169, 161, 0.45)", "rgba(42, 169, 161, 1.0)"), use_container_width=True)
            st.dataframe(scores_table(comp_scores), use_container_width=True)


if improved_scores and comp_scores:
    # Compare competitor vs the *improved* copy
    lagging = []
    for k in ATTRS:
        client_val = float(improved_scores[k]["score"])
        comp_val = float(comp_scores[k]["score"])
        delta = round(comp_val - client_val, 2)
        if delta > 0.05:
            lagging.append((_pretty_attr(k), delta))

    if lagging:
        st.markdown(
            f"""
            <div style="background:#FFF4CC; border:1px solid #E2C268;
                        border-radius:8px; padding:0.8rem; margin-top:1rem;">
                <div style="font-weight:700; color:{BRAND_NAVY}; margin-bottom:0.5rem;">
                    Competitive Gap Analysis (vs Improved Copy)
                </div>
                {''.join([f"<div>⚠️ Still lagging in <b>{attr}</b> by {delta:+.2f}</div>" for attr, delta in lagging])}
            </div>
            """, unsafe_allow_html=True)

        if st.button("Improve again to close competitive gaps", use_container_width=True):
            # Start from the improved text
            source_text = st.session_state.get("improved_text") or st.session_state.get("base_text","")
            if source_text:
                # Build target levels: for lagging attrs, aim slightly above competitor (margin 0.06)
                comp_targets = { _pretty_attr(k): min(1.0, float(comp_scores[k]["score"]) + 0.06) for k in ATTRS }
                # Merge with user sliders: keep the higher of the two for lagging attributes
                merged_targets = {}
                for k in ATTRS:
                    pretty = _pretty_attr(k)
                    user_t = desired_targets.get(pretty, 0.78)
                    # If attribute is in lagging list, bump target to max(user, comp+margin)
                    if any(pretty == name for name, _ in lagging):
                        merged_targets[pretty] = max(user_t, comp_targets[pretty])
                    else:
                        merged_targets[pretty] = user_t
                with st.spinner("Rewriting to close gaps…"):
                    improved_text2 = rewrite_copy_to_targets(api_key, source_text, merged_targets)
                with st.spinner("Scoring new improved copy…"):
                    improved_scores2 = score_text(api_key, improved_text2)
                st.session_state["improved_text"] = improved_text2
                st.session_state["improved_scores"] = improved_scores2
                _upsert_record("Client", client_channel, "Improved (Gap Close)", improved_scores2)
                st.success("Generated an even stronger version aimed to close competitive gaps!")


# ---------------- Heatmap View ----------------

st.subheader("Attribute Importance Heatmap")
st.caption("Color-coded median scores of each attribute across channels from all scored items (client, improved, and competitors).")

def _seed_full_demo_heatmap():
    _init_records()
    if st.session_state["score_records"]:
        return
    for i, ch in enumerate(CHANNELS):
        _upsert_record("Client", ch, "Original", random_competitor_scores(seed=100 + i))
        _upsert_record("Client", ch, "Improved", random_competitor_scores(seed=200 + i))
        _upsert_record("Competitor A", ch, "Competitor", random_competitor_scores(seed=300 + i))

_init_records()
if st.session_state.get('preload_demo'):
    _seed_full_demo_heatmap()

all_entities = sorted({r["entity"] for r in st.session_state["score_records"]} | {"Client"})
all_variants = sorted({r["variant"] for r in st.session_state["score_records"]} | {"Original", "Improved", "Competitor"})

fcol1, fcol2 = st.columns([2,2])
with fcol1:
    sel_entities = st.multiselect("Entities", options=all_entities, default=all_entities,
                                  help="Filter which entities to include (e.g., Client, Competitor A)")
with fcol2:
    sel_variants = st.multiselect("Variants", options=all_variants, default=all_variants,
                                  help="Filter Original / Improved / Competitor entries")

heatmap_df = _records_to_channel_attr_medians(entities=sel_entities or None, variants=sel_variants or None)
_heatmap(heatmap_df, title="Attribute Importance Heatmap (Median by Channel)")

# ---------------- Channel Trends Over Time ----------------
st.subheader("Channel Trends Over Time")
st.caption("Channel attribute trends over time. For non-demo users, shows only actual scores over time.")

def _seed_demo_trends():
    """Create 12-month synthetic time series per (Entity, Variant, Channel, Attribute) using random walks."""
    if "monthly_attr_trends" in st.session_state:
        df = st.session_state["monthly_attr_trends"]
        if isinstance(df, pd.DataFrame) and set(["Entity","Variant"]).issubset(df.columns):
            return
        else:
            # Bad/missing schema -> reset
            if "monthly_attr_trends" in st.session_state:
                del st.session_state["monthly_attr_trends"]

    _init_records()
    # Only seed demo if the test-user preload is active
    if not st.session_state["score_records"] and not st.session_state.get('preload_demo'):
        return
    if not st.session_state["score_records"] and st.session_state.get('preload_demo'):
        _seed_full_demo_heatmap()

    months = pd.date_range(
        end=pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0),
        periods=12,
        freq="M",
    ).strftime("%Y-%m")
    entities = sorted({r["entity"] for r in st.session_state["score_records"]} | {"Client"})
    variants = sorted({r["variant"] for r in st.session_state["score_records"]} | {"Original","Improved","Competitor"})

    def random_walk_series(seed: int, start: float) -> list[float]:
        import random as _rnd
        _rnd.seed(seed)
        vals = [max(0.0, min(1.0, start))]
        sigma = 0.16
        for _ in range(1, 12):
            step = _rnd.gauss(0.0, sigma)
            nxt = vals[-1] + step
            if nxt < 0.0: nxt = -nxt
            if nxt > 1.0: nxt = 2.0 - nxt
            vals.append(max(0.0, min(1.0, nxt)))
        vmin, vmax = min(vals), max(vals)
        span = vmax - vmin
        if span < 0.65:
            mean = sum(vals) / len(vals)
            expand = (0.80 / max(span, 1e-6))
            vals = [mean + (v - mean) * expand for v in vals]
            vals = [min(1.0, max(0.0, (2.0 - v) if v > 1.0 else (-v if v < 0.0 else v))) for v in vals]
        vmin, vmax = min(vals), max(vals)
        if vmax < 0.92:
            add = 0.92 - vmax
            vals = [min(1.0, v + add * 0.6) for v in vals]
        if vmin > 0.08:
            sub = vmin - 0.08
            vals = [max(0.0, v - sub * 0.6) for v in vals]
        return [round(v, 3) for v in vals]

    rows = []
    for ent in entities:
        for var in variants:
            for ch in CHANNELS:
                for attr in ATTRS:
                    seed_val = abs(hash(f"rw:{ent}:{var}:{ch}:{attr}")) % (2**32)
                    start = 0.05 + (seed_val % 9000) / 9000.0 * 0.90
                    series = random_walk_series(seed_val, start)
                    for m, v in zip(months, series):
                        rows.append({
                            "Entity": ent,
                            "Variant": var,
                            "Channel": ch,
                            "Month": m,
                            "Attribute": _pretty_attr(attr),
                            "Score": v,
                        })
    st.session_state["monthly_attr_trends"] = pd.DataFrame(rows)


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

trend_df = st.session_state.get("monthly_attr_trends")
can_plot_trends = isinstance(trend_df, pd.DataFrame) and not trend_df.empty

trend_channel = st.selectbox("Select channel", CHANNELS, index=0, key="trend_channel")

if can_plot_trends:
    try:
        ef = set(sel_entities) if 'sel_entities' in locals() and sel_entities else set(sorted(trend_df["Entity"].unique()))
        vf = set(sel_variants) if 'sel_variants' in locals() and sel_variants else set(sorted(trend_df["Variant"].unique()))
        filtered = trend_df[(trend_df["Entity"].isin(ef)) & (trend_df["Variant"].isin(vf)) & (trend_df["Channel"] == trend_channel)]
    except Exception:
        filtered = trend_df[trend_df["Channel"] == trend_channel]

    if not filtered.empty:
        agg = (filtered.groupby(["Month", "Attribute"], as_index=False)["Score"].median())
        _plot_channel_trends(agg)
    else:
        st.info("No trend data for the selected channel/filters yet.")
else:
    st.info("No trend data yet. Score items or use the test user to see demo trends.")

# ---------------- Attribute Correlation Explorer ----------------
render_correlation_section()
