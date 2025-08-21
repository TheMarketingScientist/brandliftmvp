
import json
import random
import httpx
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

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
ATTRS = ["Leadership", "Ease_of_Use", "Quality", "Luxury", "Cost_Benefit"]

PREFERRED_MODEL = "claude-3-5-sonnet-latest"
FALLBACK_MODEL  = "claude-3-5-haiku-latest"
API_BASE = "https://api.anthropic.com/v1/messages"

SYSTEM_SCORE = """You are a brand perception rater. Score ad copy on 5 attributes:
Leadership, Ease_of_Use, Quality, Luxury, Cost_Benefit.
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

def _parse_json_block(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise RuntimeError("Model did not return JSON. Output was: " + text[:200])

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
    import httpx
    with httpx.Client(timeout=60) as client:
        r = client.post(API_BASE, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code} from Anthropic ({model}) — {e.response.text}")
    data = r.json()
    return data["content"][0]["text"]

def _with_fallback(func, *args, **kwargs):
    try:
        return func(PREFERRED_MODEL, *args, **kwargs), PREFERRED_MODEL
    except RuntimeError as e:
        s = str(e)
        if ("HTTP 403" in s) or ("HTTP 404" in s) or ("not_found" in s) or ("model not found" in s):
            return func(FALLBACK_MODEL, *args, **kwargs), FALLBACK_MODEL
        raise

def score_text(api_key: str, text: str) -> dict:
    user_content = f"Text:\\n{text}\\n\\nOutput JSON schema:\\n{json.dumps(_schema())}"
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
    instr = (
        "Rewrite the text to increase the perception of: " + t +
        ". Preserve original meaning, keep tone professional, avoid hype. Max 70 words. "
        'Return JSON: {"rewrite":"..."}'
    )
    def run(model, api_key_inner, prompt_text):
        out_text = _call_messages(api_key_inner, SYSTEM_REWRITE, prompt_text, model, temperature=0.4, max_tokens=400)
        data = _parse_json_block(out_text)
        return data.get("rewrite", "").strip()
    rewrite, _ = _with_fallback(run, api_key, instr + "\\n\\nOriginal:\\n" + text)
    return rewrite or text

def propose_new_ideas(api_key: str, base_text: str, targets: list[str], n: int = 2) -> list[str]:
    t = ", ".join(targets) if targets else "Leadership"
    prompt = (
        "Given this ad copy, propose " + str(n) + " distinct alternative ideas that could outperform a competitor "
        "on the following target attributes: " + t + ". "
        "Keep each idea under 35 words. Return JSON: {\\"ideas\\": [\\"...\\", \\"...\\"]}\\n\\n"
        "Original:\\n" + base_text
    )
    def run(model, api_key_inner, uc):
        out_text = _call_messages(api_key_inner, SYSTEM_IDEA, uc, model, temperature=0.6, max_tokens=500)
        data = _parse_json_block(out_text)
        return data.get("ideas", [])
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
        phrase = random.choice(COMP_PHRASES[k])
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

def trailing_badges(improved: dict, competitor: dict, threshold: float = 0.03) -> str:
    trailing = []
    for k in ATTRS:
        gap = float(competitor[k]["score"]) - float(improved[k]["score"])
        if gap > threshold:
            name = k.replace("_"," ")
            trailing.append((name, gap))
    if not trailing:
        return ""
    spans = []
    for name, gap in sorted(trailing, key=lambda x: -x[1]):
        spans.append(f'<span class="pill-warn">{name} (+{gap:.2f})</span>')
    return '<div class="warn-box"><div class="warn-title">Still behind competitor on</div>' + " ".join(spans) + "</div>"

def show_competitor_media(url: str):
    """Preview image/video if possible; always show a clickable link."""
    if not url:
        return
    lower = url.lower()
    st.markdown(f"[Open creative]({url})")
    try:
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            st.image(url, use_container_width=True)
        elif lower.endswith((".mp4", ".webm", ".ogg")) or "youtube.com" in lower or "youtu.be" in lower or "vimeo.com" in lower:
            st.video(url)
    except Exception:
        pass

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

# ---------------- UI ----------------
st.set_page_config(page_title="Brand Perception Scoring (MVP)", layout="wide")
inject_css()

# Header
hdr_left, hdr_right = st.columns([4,1])
with hdr_left:
    st.title("Brand Perception Scoring (MVP)")
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
     f'<span class="badge" style="background:#2B386A">Cost/Benefit</span>']
) + "</div>", unsafe_allow_html=True)

text = st.text_area(
    "Paste ad copy or transcript:",
    height=160,
    placeholder="Example: Introducing the next-generation sedan—hand-finished interiors, seamless app control, and advanced safety.",
)

# Action row
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
    add_comp = st.button("Add Competitor", use_container_width=True)

# Handle actions
if score_clicked:
    if not api_key:
        st.error("No API key found in Secrets.")
    elif not text.strip():
        st.warning("Please paste some text.")
    else:
        with st.spinner("Scoring…"):
            base_scores = score_text(api_key, text)
        st.session_state["base_text"] = text
        st.session_state["base_scores"] = base_scores

if improve_clicked:
    source_text = text.strip() or st.session_state.get("base_text", "")
    if not api_key:
        st.error("No API key found in Secrets.")
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

if add_comp:
    st.session_state["competitor_name"] = comp_name or "Competitor"
    st.session_state["competitor_url"] = comp_url.strip()
    st.session_state["competitor_scores"] = random_competitor_scores()

# ------- Comparison view (Original vs Improved vs Competitor) -------
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
                # show link and preview if possible
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

# Delta table for Original vs Improved
if base_scores and improved_scores:
    st.subheader("Score Changes (Original → Improved)")
    st.dataframe(delta_table(base_scores, improved_scores), use_container_width=True)

# Trailing attributes vs competitor
def trailing_badges(improved: dict, competitor: dict, threshold: float = 0.03) -> str:
    trailing = []
    for k in ATTRS:
        gap = float(competitor[k]["score"]) - float(improved[k]["score"])
        if gap > threshold:
            name = k.replace("_"," ")
            trailing.append((name, gap))
    if not trailing:
        return ""
    spans = []
    for name, gap in sorted(trailing, key=lambda x: -x[1]):
        spans.append(f'<span class="pill-warn">{name} (+{gap:.2f})</span>')
    return '<div class="warn-box"><div class="warn-title">Still behind competitor on</div>' + " ".join(spans) + "</div>"

if improved_scores and comp_scores:
    st.markdown("""
    <style>
    .warn-box { background: #FFF4CC; border: 1px solid #E2C268; border-radius: 8px;
                padding: 0.6rem; margin-top: 0.5rem; margin-bottom: 0.5rem; }
    .warn-title { font-weight: 700; color: #2E3C71; margin-bottom: 0.25rem; }
    .pill-warn { display:inline-block; padding: 0.2rem 0.5rem; border-radius: 999px;
                 background: white; border: 1px solid #E2C268; color: #2E3C71;
                 font-size: 0.85rem; margin-right: 0.35rem; margin-bottom: 0.35rem; }
    </style>
    """, unsafe_allow_html=True)
    html = trailing_badges(improved_scores, comp_scores, threshold=0.03)
    if html:
        st.markdown(html, unsafe_allow_html=True)
        # Pick top trailing attributes for proposed ideas
        trailing_list = sorted(
            [(k.replace("_"," "), float(comp_scores[k]["score"]) - float(improved_scores[k]["score"])) for k in ATTRS],
            key=lambda x: -x[1]
        )
        targets_auto = [name for name, gap in trailing_list if gap > 0.03][:2]
        with st.expander("Consider a different idea (generate alternatives)"):
            if st.button("Propose New Idea", use_container_width=True):
                ideas = propose_new_ideas(api_key, st.session_state.get("improved_text") or st.session_state.get("base_text",""), targets_auto or ["Leadership"])
                for i, idea in enumerate(ideas, start=1):
                    st.markdown(f"**Idea {i}**")
                    st.write(idea)
