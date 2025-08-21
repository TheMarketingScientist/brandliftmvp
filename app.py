
import json
import httpx
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ---------- Config ----------
ATTRS = ["Leadership", "Ease_of_Use", "Quality", "Luxury", "Cost_Benefit"]

PREFERRED_MODEL = "claude-3-5-sonnet-latest"
FALLBACK_MODEL  = "claude-3-5-haiku-latest"   # or "claude-3-haiku-20240307"
API_BASE = "https://api.anthropic.com/v1/messages"

SYSTEM_SCORE = """You are a brand perception rater. Score ad copy on 5 attributes:
Leadership, Ease_of_Use, Quality, Luxury, Cost_Benefit.
Each score is a float in [0,1]. Anchors: 0.2 = weak, 0.5 = moderate, 0.8 = strong.
Return STRICT JSON with each attribute as {"score": float, "evidence": "short phrase (<=12 words)"}.
No extra text.
"""
SYSTEM_REWRITE = "You are a precise brand copy editor focused on targeted brand attributes."
# ----------------------------

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

def _call_messages(api_key: str, system: str, user_content: str, model: str) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 700,
        "temperature": 0.2,
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
    return data["content"][0]["text"]

def _with_fallback(func, *args, **kwargs):
    """Run a call with Sonnet first, then Haiku if 403/404/not found. No warnings are shown."""
    try:
        return func(PREFERRED_MODEL, *args, **kwargs), PREFERRED_MODEL
    except RuntimeError as e:
        s = str(e)
        if ("HTTP 403" in s) or ("HTTP 404" in s) or ("not_found" in s) or ("model not found" in s):
            return func(FALLBACK_MODEL, *args, **kwargs), FALLBACK_MODEL
        raise

def score_text(api_key: str, text: str) -> dict:
    user_content = f"Text:\n{text}\n\nOutput JSON schema:\n{json.dumps(_schema())}"
    def run(model, api_key_inner, uc):
        out_text = _call_messages(api_key_inner, SYSTEM_SCORE, uc, model)
        scores = _parse_json_block(out_text)
        for k in ATTRS:
            s = float(scores[k]["score"])
            scores[k]["score"] = max(0.0, min(1.0, s))
        return scores
    scores, _used = _with_fallback(run, api_key, user_content)
    return scores

def rewrite_copy(api_key: str, text: str, targets: list[str]) -> str:
    t = ", ".join(targets) if targets else "Leadership"
    instr = (
        "Rewrite the text to increase the perception of: " + t +
        ". Preserve original meaning, keep tone professional, avoid hype. Max 70 words. "
        'Return JSON: {"rewrite":"..."}'
    )
    def run(model, api_key_inner, prompt_text):
        out_text = _call_messages(api_key_inner, SYSTEM_REWRITE, prompt_text, model)
        data = _parse_json_block(out_text)
        return data.get("rewrite", "").strip()
    rewrite, _used = _with_fallback(run, api_key, instr + "\n\nOriginal:\n" + text)
    return rewrite or text

def radar(scores: dict, title: str):
    vals = [scores[k]["score"] for k in ATTRS]
    fig = go.Figure(data=go.Scatterpolar(r=vals + vals[:1], theta=ATTRS + ATTRS[:1], fill="toself"))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(range=[0, 1])), showlegend=False,
                      margin=dict(l=40, r=40, t=40, b=40))
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

# ---------------- UI ----------------
st.set_page_config(page_title="Brand Perception Scoring (MVP)", layout="wide")
st.title("Brand Perception Scoring (MVP)")

api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
if not api_key:
    st.info('Add your key in Settings → Secrets as:\nANTHROPIC_API_KEY = "sk-ant-..."')

text = st.text_area(
    "Paste ad copy or transcript:",
    height=160,
    placeholder="Example: Introducing the next-generation sedan—hand-finished interiors, seamless app control, and advanced safety.",
)

cols = st.columns(2)
with cols[0]:
    score_clicked = st.button("Score", use_container_width=True)
with cols[1]:
    targets = st.multiselect("Improve attributes", ATTRS, default=["Leadership","Quality"])
    improve_clicked = st.button("Improve & Rescore", use_container_width=True)

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
        # Ensure base exists (score if needed)
        if "base_scores" not in st.session_state:
            st.session_state["base_text"] = source_text
            st.session_state["base_scores"] = score_text(api_key, source_text)

# ------- Comparison view (Original vs Improved) -------
base_scores = st.session_state.get("base_scores")
improved_scores = st.session_state.get("improved_scores")

if base_scores and improved_scores:
    st.subheader("Comparison")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Original Copy**")
        st.write(st.session_state.get("base_text", ""))
        st.plotly_chart(radar(base_scores, "Original Scores"), use_container_width=True)
        st.dataframe(scores_table(base_scores), use_container_width=True)

    with c2:
        st.markdown("**Improved Copy**")
        st.write(st.session_state.get("improved_text", ""))
        st.plotly_chart(radar(improved_scores, "Improved Scores"), use_container_width=True)
        st.dataframe(scores_table(improved_scores), use_container_width=True)

    st.subheader("Score Changes")
    st.dataframe(delta_table(base_scores, improved_scores), use_container_width=True)

elif base_scores and not improved_scores:
    st.subheader("Perception Scores")
    st.plotly_chart(radar(base_scores, "Original Scores"), use_container_width=True)
    st.dataframe(scores_table(base_scores), use_container_width=True)
