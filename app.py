
import json
import httpx
import streamlit as st
import plotly.graph_objects as go

# ---------- Config ----------
ATTRS = ["Leadership", "Ease_of_Use", "Quality", "Luxury", "Cost_Benefit"]

# Use the current/alias IDs to avoid "model not found" errors.
PREFERRED_MODEL = "claude-3-5-sonnet-latest"
FALLBACK_MODEL  = "claude-3-5-haiku-latest"   # or "claude-3-haiku-20240307" if needed

API_BASE = "https://api.anthropic.com/v1/messages"
SYSTEM = """You are a brand perception rater. Score ad copy on 5 attributes:
Leadership, Ease_of_Use, Quality, Luxury, Cost_Benefit.
Each score is a float in [0,1]. Anchors: 0.2 = weak, 0.5 = moderate, 0.8 = strong.
Return STRICT JSON with each attribute as {"score": float, "evidence": "short phrase (<=12 words)"}.
No extra text.
"""
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


def _invoke(api_key: str, text: str, model: str) -> dict:
    """Call Anthropic Messages API and return parsed score JSON."""
    prompt = f"Text:\n{text}\n\nOutput JSON schema:\n{json.dumps(_schema())}"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 600,
        "temperature": 0.2,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(API_BASE, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Surface body so you can see exact Anthropic error without digging in logs
            raise RuntimeError(f"HTTP {e.response.status_code} from Anthropic ({model}) — {e.response.text}")

    data = r.json()
    out_text = data["content"][0]["text"]

    # Parse strict JSON; if model added stray text, try extracting JSON block.
    try:
        scores = json.loads(out_text)
    except json.JSONDecodeError:
        start, end = out_text.find("{"), out_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            scores = json.loads(out_text[start : end + 1])
        else:
            raise RuntimeError("Model did not return JSON. Output was: " + out_text[:200])

    # Clamp to [0,1]
    for k in ATTRS:
        s = float(scores[k]["score"])
        scores[k]["score"] = max(0.0, min(1.0, s))

    return scores


def score_text(api_key: str, text: str) -> tuple[dict, str]:
    """
    Try preferred model; on 403/404/not-found or similar, automatically fall back.
    Returns (scores, model_used).
    """
    try:
        return _invoke(api_key, text, PREFERRED_MODEL), PREFERRED_MODEL
    except RuntimeError as e:
        err = str(e)
        # Fallback on common access/name issues
        if ("HTTP 403" in err) or ("HTTP 404" in err) or ("not_found" in err) or ("model not found" in err):
            st.warning("Preferred model unavailable. Retrying with fallback…")
            return _invoke(api_key, text, FALLBACK_MODEL), FALLBACK_MODEL
        raise


def radar(scores: dict, title: str):
    vals = [scores[k]["score"] for k in ATTRS]
    fig = go.Figure(
        data=go.Scatterpolar(r=vals + vals[:1], theta=ATTRS + ATTRS[:1], fill="toself")
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(range=[0, 1])),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


# ---------------- UI ----------------
st.set_page_config(page_title="Brand Perception Scoring (MVP)", page_icon="📈", layout="centered")
st.title("📈 Brand Perception Scoring (MVP)")

api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()
if api_key:
    st.caption(f"✅ API key detected (length {len(api_key)}).")
else:
    st.info('Add your key in **Settings → Secrets** as:\n\nANTHROPIC_API_KEY = "sk-ant-..."')

text = st.text_area(
    "Paste ad copy or transcript:",
    height=180,
    placeholder="e.g., Introducing the next-generation sedan—hand-finished interiors, seamless app control, and advanced safety.",
)

if st.button("Score"):
    if not api_key:
        st.error("No API key found in Secrets.")
    elif not text.strip():
        st.warning("Please paste some text.")
    else:
        with st.spinner("Scoring…"):
            try:
                scores, used_model = score_text(api_key, text)
            except Exception as e:
                st.error(str(e))
            else:
                st.caption(f"Model used: {used_model}")
                st.plotly_chart(radar(scores, "Perception Scores"), use_container_width=True)
                st.json(scores)
