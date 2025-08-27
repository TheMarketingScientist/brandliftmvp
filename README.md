# 📈 Brand Lift MVP

Minimal Streamlit app to score ad copy or transcripts across brand perception attributes:
- Leadership
- Ease of Use
- Quality
- Luxury
- Cost/Benefit
- Trust

## 🚀 Quick Start (Local)

1) **Install**
```bash
pip install -r requirements.txt
```

2) **Add your API key to Streamlit secrets**
Create a file at `.streamlit/secrets.toml` (note the leading dot directory):
```toml
ANTHROPIC_API_KEY = "your-key-here"
```

3) **Run**
```bash
streamlit run app.py
```

## ☁️ Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and create a new app pointing to `app.py`.
3. In your app's **Settings → Secrets**, add:
   ```
   ANTHROPIC_API_KEY = "your-key-here"
   ```
4. Deploy — you'll get a live URL.

## 🧩 Notes
- Uses Anthropic Claude via the Messages API. You can swap to another provider by changing the `call_anthropic` function.
- Outputs strict JSON with a score (0–1) and a short evidence phrase per attribute.
- Radar chart visualizes the five scores.
