
import json
import random
import httpx
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import re

# ---------- Brand Theme ----------
BRAND_BLUE   = "#445DA7"  # Original
BRAND_PURPLE = "#6B3894"  # Improved
BRAND_NAVY   = "#2E3C71"
COMP_TEAL    = "#2AA9A1"  # Competitor

# Semi-transparent fills
blue_fill    = "rgba(68, 93, 167, 0.45)"
blue_line    = "rgba(68, 93, 167, 1.0)"
purple_fill  = "rgba(107, 56, 148, 0.45)"
purple_line  = "rgba(107, 56, 148, 1.0)"
teal_fill    = "rgba(42, 169, 161, 0.45)"
teal_line    = "rgba(42, 169, 161, 1.0)"

# ---------- Config ----------
TITLE_COLOR = BRAND_NAVY  # change to BRAND_BLUE or BRAND_PURPLE if you prefer

# EXACT modern channels as requested
CHANNELS = ["CTV", "DOOH", "Youtube", "TikTok", "Google Ads", "Instagram", "X"]

ATTRS = ["Leadership", "Ease_of_Use", "Quality", "Luxury", "Cost_Benefit", "Trust"]

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
    # Remove ```json ... ``` or ``` ... ``` fences if present
    text = text.strip()
    if text.startswith("```"):
        # drop first fence line
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else ""
        # drop ending fence
        if text.rstrip().endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()

def _extract_json_substring(text: str) -> str | None:
    """Return the first top-level JSON object/array substring using brace matching.
    Handles strings and escapes to avoid premature closing.
    """
    text = _strip_code_fences(text)
    # Try a direct load first
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Find first { or [
    start_idx = None
    for i, ch in enumerate(text):
        if ch in '{[':
            start_idx = i
            break
    if start_idx is None:
        return None

    stack = [text[start_idx]]
    i = start_idx + 1
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in '{[':
                stack.append(ch)
            elif ch in '}]':
                if not stack:
                    return None
                top = stack[-1]
                if (top == '{' and ch == '}') or (top == '[' and ch == ']'):
                    stack.pop()
                    if not stack:
                        # Found matching end
                        candidate = text[start_idx:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            # keep scanning in case there is another valid block later
                            pass
                else:
                    # mismatched brace, abort
                    return None
        i += 1
    return None

def _parse_json_block(text: str) -> dict:
    # Most robust JSON extractor: code-fence removal, brace matching, and smart-quote normalization
    if not isinstance(text, str):
        raise RuntimeError("Model did not return text content.")
    # Normalize common smart quotes to straight quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('“','"').replace('”','"').replace('’',"'").replace('‘',"'")
    # Remove trailing junk like \"}...note\" after the JSON
    candidate = _extract_json_substring(text)
    if candidate is None:
        raise RuntimeError("Model did not return JSON. Output was: " + text[:400])
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Last-resort: try to replace single quotes with double quotes if it looks like JSON-ish
        jlike = re.sub(r"(?<!\\)'", '"', candidate)
        try:
            return json.loads(jlike)
        except Exception:
            raise RuntimeError(f"Could not parse JSON. Error: {e}. Output was: {text[:400]}")

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
    # Some responses have multiple content blocks. Join all text blocks.
    blocks = data.get("content", [])
    texts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text" and "text" in b:
            texts.append(b["text"])
    return "\n".join(texts) if texts else (blocks[0]["text"] if blocks and isinstance(blocks[0], dict) and "text" in blocks[0] else "")

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
        # Fallbacks if the model didn't send valid JSON:
        # 1) If output looks like plain text, use it directly (trim to ~70 words)
        clean = _strip_code_fences(out_text).strip()
        if clean:
            words = clean.split()
            return " ".join(words[:70])
        # 2) Last resort: return original
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
            # fallback: split lines/bullets if model didn't follow JSON
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
    vals = [scores[k]["score"] for k in ATTRS]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + vals[:1],
        theta=ATTRS + ATTRS[:1],
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
            "Attribute": k.replace("_", " "),
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
            "Attribute": k.replace("_", " "),
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
        if entities and r["entity"] not in entities:
            continue
        if variants and r["variant"] not in variants:
            continue
        channel = r["channel"]
        sc = r["scores"]
        for attr in ATTRS:
            rows.append({"Channel": channel, "Attribute": attr.replace("_", " "), "Score": float(sc[attr]["score"])})
    if not rows:
        return pd.DataFrame(columns=["Channel"] + [a.replace("_", " ") for a in ATTRS])
    df = pd.DataFrame(rows)
    med = df.groupby(["Channel", "Attribute"], as_index=False)["Score"].median()
    pivot = med.pivot(index="Channel", columns="Attribute", values="Score").reindex(CHANNELS, fill_value=None)
    pivot = pivot[[a.replace("_"," ") for a in ATTRS]]
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

# ---------------- UI ----------------
st.set_page_config(page_title="Brand Lift", layout="wide")
inject_css()

hdr_left, hdr_right = st.columns([4,1])
with hdr_left:
    st.markdown("<h1 class='app-title'>Brand Lift</h1>", unsafe_allow_html=True)
with hdr_right:
    logo_path = Path("logo.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
if not api_key:
    st.info('Add your key in Settings → Secrets as:\nANTHROPIC_API_KEY = "sk-ant-..."')

st.markdown("**Attributes**")
st.markdown('<div class="badges">' + " ".join(
    [f'<span class="badge" style="background:{BRAND_NAVY}">Leadership</span>',
     f'<span class="badge" style="background:{BRAND_BLUE}">Ease of Use</span>',
     f'<span class="badge" style="background:{BRAND_PURPLE}">Quality</span>',
     f'<span class="badge" style="background:#64348B">Luxury</span>',
     f'<span class="badge" style="background:#2B386A">Cost/Benefit</span>',
     f'<span class="badge" style="background:#2AA9A1">Trust</span>']
) + "</div>", unsafe_allow_html=True)

text = st.text_area("Paste ad copy or transcript:", height=160,
    placeholder="Example: Introducing the next-generation sedan—hand-finished interiors, seamless app control, and advanced safety.")

client_url = st.text_input("Client creative URL (optional)", placeholder="https://...")
client_channel = st.selectbox("Client channel", CHANNELS, index=0)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    score_clicked = st.button("Score", use_container_width=True)
with c2:
    targets = st.multiselect("Improve attributes", ATTRS, default=["Leadership","Quality"])
    improve_clicked = st.button("Improve & Rescore", use_container_width=True)
with c3:
    comp_cols = st.columns([1.2, 2.8])
    with comp_cols[0]:
        comp_name = st.text_input("Competitor label", value="Competitor A")
    with comp_cols[1]:
        comp_url = st.text_input("Competitor ad/creative URL (optional)", placeholder="https://...")
    comp_text = st.text_area("Competitor copy/transcript (optional)", height=120, placeholder="Paste competitor copy/transcript to score (optional)")
    comp_channel = st.selectbox("Competitor channel", CHANNELS, index=0)
    score_comp = st.button("Score Competitor", use_container_width=True)

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

if improve_clicked:
    source_text = text.strip() or st.session_state.get("base_text", "")
    if not api_key:
        st.error("No API key found in Secrets.")
    elif not client_channel:
        st.warning("Please select a channel before scoring.")
    elif not source_text:
        st.warning("Paste text and click Score first.")
    else:
        with st.spinner("Generating improved copy…"):
            improved_text = rewrite_copy(api_key, source_text, targets)
        with st.spinner("Scoring improved copy…"):
            improved_scores = score_text(api_key, improved_text)
        st.session_state["improved_text"] = improved_text
        st.session_state["improved_scores"] = improved_scores
        if "base_scores" not in st.session_state:
            st.session_state["base_text"] = source_text
            st.session_state["base_scores"] = score_text(api_key, source_text)
            _upsert_record("Client", client_channel, "Original", st.session_state["base_scores"])
        _upsert_record("Client", client_channel, "Improved", improved_scores)

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
            st.plotly_chart(radar(base_scores, "Original Scores", blue_fill, blue_line), use_container_width=True)
            st.dataframe(scores_table(base_scores), use_container_width=True)
        idx += 1

    if improved_scores:
        with cols[idx]:
            st.markdown("**Improved Copy**")
            st.write(st.session_state.get("improved_text",""))
            st.plotly_chart(radar(improved_scores, "Improved Scores", purple_fill, purple_line), use_container_width=True)
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
            st.plotly_chart(radar(comp_scores, f"{comp_label} Scores", teal_fill, teal_line), use_container_width=True)
            st.dataframe(scores_table(comp_scores), use_container_width=True)

if base_scores and improved_scores:
    st.subheader("Score Changes (Original → Improved)")
    st.dataframe(delta_table(base_scores, improved_scores), use_container_width=True)

# ---------------- Heatmap View ----------------
st.subheader("Attribute Importance Heatmap")
st.caption("Color-coded median scores of each attribute across channels from all scored items (client, improved, and competitors).")

_init_records()

def _seed_full_demo_heatmap():
    _init_records()
    if st.session_state["score_records"]:
        return
    for i, ch in enumerate(CHANNELS):
        _upsert_record("Client", ch, "Original", random_competitor_scores(seed=100 + i))
        _upsert_record("Client", ch, "Improved", random_competitor_scores(seed=200 + i))
        _upsert_record("Competitor A", ch, "Competitor", random_competitor_scores(seed=300 + i))

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
st.caption("12-month synthetic median attribute trends, per selected channel (demo data).")




def _seed_demo_trends():
    """Create 12-month synthetic time series per (Entity, Variant, Channel, Attribute) using random walks.
    This lets Heatmap selectors (Entities, Variants) also filter the trends.
    """
    if "monthly_attr_trends" in st.session_state:
        # If the old DF lacks Entity/Variant (legacy), force a rebuild
        df = st.session_state["monthly_attr_trends"]
        if not set(["Entity","Variant"]).issubset(df.columns):
            del st.session_state["monthly_attr_trends"]
        else:
            return

    # Ensure score_records exist (for available Entity/Variant labels)
    _init_records()
    if not st.session_state["score_records"]:
        # seed heatmap demo (Client Original/Improved + Competitor)
        _seed_full_demo_heatmap()

    months = pd.date_range(
        end=pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0),
        periods=12,
        freq="M",
    ).strftime("%Y-%m")

    # Derive unique Entities and Variants from current records
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
            if nxt < 0.0:
                nxt = -nxt
            if nxt > 1.0:
                nxt = 2.0 - nxt
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
                            "Attribute": attr.replace("_"," "),
                            "Score": v,
                        })

    st.session_state["monthly_attr_trends"] = pd.DataFrame(rows)

def _plot_channel_trends(df_channel: pd.DataFrame):
    # Expect tidy DF filtered to one channel: cols = Channel, Month, Attribute, Score
    # Pivot for convenience in table view; plot as multi-line by attribute.
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

    # Also show a compact wide table for quick copy/export
    wide = df_channel.pivot(index="Month", columns="Attribute", values="Score").reset_index()
    st.dataframe(wide, use_container_width=True)

# Seed once per session
_seed_demo_trends()


# Selector and plot
trend_channel = st.selectbox("Select channel", CHANNELS, index=0, key="trend_channel")
trend_df = st.session_state["monthly_attr_trends"]

# Apply Heatmap filters if available (sel_entities, sel_variants)
try:
    ef = set(sel_entities) if sel_entities else set(sorted(trend_df["Entity"].unique()))
    vf = set(sel_variants) if sel_variants else set(sorted(trend_df["Variant"].unique()))
    filtered = trend_df[(trend_df["Entity"].isin(ef)) & (trend_df["Variant"].isin(vf)) & (trend_df["Channel"] == trend_channel)]
except NameError:
    # Fallback if variables not in scope
    filtered = trend_df[trend_df["Channel"] == trend_channel]

# Aggregate across selected Entities/Variants -> median per Month x Attribute
if not filtered.empty:
    agg = (filtered.groupby(["Month", "Attribute"], as_index=False)["Score"].median())
else:
    agg = filtered.copy()

_plot_channel_trends(agg)

