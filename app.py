import json
import httpx
import streamlit as st
import plotly.graph_objects as go

# ---------- Config ----------
ATTRS = ["Leadership","Ease_of_Use","Quality","Luxury","Cost_Benefit"]
PREFERRED_MODEL = "claude-3-5-sonnet-20240620"
FALLBACK_MODEL = "claude-3-haiku-20240307"
API_BASE = "https://api.anthropic.com/v1/messages"
SYSTEM = """You are a brand perception rater. Score ad copy on 5 attributes:
Leadership, Ease_of_Use, Quality, Luxury, Cost_Benefit.
Each score is a float in [0,1]. Anchors: 0.2=weak, 0.5=moderate, 0.8=strong.
Return STRICT JSON with each attribute as {\"score\": float, \"evidence\": \"short phrase\"}.
No extra text.
"""
# ----------------------------

def _invoke(api_key: str, text: str, model: str) -> dict:
    schema = {
      "type":"object",
      "properties":{k:{
        "type":"object",
        "properties":{"score":{"type":"number"},"evidence":{"type":"string"}},
        "required":["score","evidence"]
      } for k in ATTRS},
      "required": ATTRS,
      "additionalProperties": False
    }
    prompt = f"Text:\n{text}\n\nOutput JSON schema:\n{json.dumps(schema)}"
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
        "messages": [{"role":"user","content": prompt}],
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(API_BASE, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Surface the API error body to the UI for fast debugging
            body = e.response.text
            raise RuntimeError(f"HTTP {e.response.status_code} from Anthropic ({model}) — {body}")
    data = r.json()
    out_text = data["content"][0]["text"]
    try:
        scores = json.loads(out_text)
    except json.JSONDecodeError:
        start, end = out_text.find("{"), out_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            scores = json.loads(out_text[start:end+1])
        else:
            raise RuntimeError("Model did not return JSON. Output was: " + out_text[:200])
    for k in ATTRS:
        s = float(scores[k]["score"])
        scores[k]["score"] = max(0.0, min(1.0, s))
    return scores

def score_text(api_key: str, text: str) -> dict:
    try:
        return _invoke(api_key, text, PREFERRED_MODEL)
    except RuntimeError as e:
        if "HTTP 403" in str(e) or "model_not_found" in str(e):
            st.warning("Access to Sonnet may be restricted. Retrying with Haiku…")
            return _invoke(api_key, text, FALLBACK_MODEL)
        raise

def radar(scores: dict, title: str):
    vals = [scores[k]["score"] for k in ATTRS]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals+vals[:1], theta=ATTRS+ATTRS[:1], fill='toself'))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(range=[0,1])),
                      showlegend=False, margin=dict(l=40,r=40,t=40,b=40))
    return fig

# ---------------- UI ----------------
st.set_page_config(page_title="Brand Lift MVP", page_icon="📈", layout="centered")
st.title("📈 Brand Perception Scoring (MVP)")

api_key = st.secrets.get("ANTHROPIC_API_KEY", "").strip()

if api_key:
    st.caption(f"✅ API key detected (length {len(api_key)}).")
else:
    st.info("Add your key in Settings → Secrets as:\nANTHROPIC_API_KEY = \"sk-ant-...\"")

text = st.text_area("Paste ad copy or transcript:", height=180,
                    placeholder="e.g., Introducing the next-generation sedan—hand-finished interiors, seamless app control, and advanced safety.")

if st.button("Score"):
    if not api_key:
        st.error("No API key found in Secrets.")
    elif not text.strip():
        st.warning("Please paste some text.")
    else:
        with st.spinner("Scoring…"):
            try:
                scores = score_text(api_key, text)
            except Exception as e:
                st.error(str(e))
            else:
                st.plotly_chart(radar(scores, "Perception Scores"), use_container_width=True)
                st.json(scores)
