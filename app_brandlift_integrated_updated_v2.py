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
with st.expander("🎯 Desired attribute targets", expanded=True):
    desired_targets = {}
    cols = st.columns(3)
    for i, a in enumerate(ATTRS):
        with cols[i % 3]:
            pretty = _pretty_attr(a)
            # Default targets are moderate-strong to create a "wow" effect on first demo
            desired_targets[pretty] = st.slider(pretty, 0.0, 1.0, 0.78, 0.01, help="Target perceived level for this attribute")
    st.caption("These targets drive the improvement step. The rewrite will **avoid explicit attribute tags** and aim for these levels in tone and content.")

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
        # desired_targets has pretty labels; convert inside helper
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
    """Create 12-month synthetic time series per (Entity, Variant, Channel, Attribute) using random walks."""
    if "monthly_attr_trends" in st.session_state:
        df = st.session_state["monthly_attr_trends"]
        if not set(["Entity","Variant"]).issubset(df.columns):
            del st.session_state["monthly_attr_trends"]
        else:
            return
    _init_records()
    if not st.session_state["score_records"]:
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
