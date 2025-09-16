
# app_brandlift_integrated_auth_fixed_v8.py
# Streamlit Brand Lift MVP with Supabase Auth, Admin Users, Org Settings, and DB persistence
# Notes:
# - Secrets are loaded via cfg(): ENV → st.secrets → .streamlit/app_config.toml → defaults.
# - On Streamlit Cloud, set secrets once in the app settings; they persist across deploys.
# - Locally, put secrets in .streamlit/secrets.toml (never commit real keys).

import os
from pathlib import Path
import json
import time
import streamlit as st
import tomllib  # Python 3.11+
from supabase import create_client, Client

# -----------------------------
# Config loader
# -----------------------------
_CFG = {}
_cfg_path = Path(".streamlit/app_config.toml")
if _cfg_path.exists():
    with open(_cfg_path, "rb") as f:
        _CFG = tomllib.load(f)

def cfg(name: str, default=None):
    """Order: ENV → st.secrets → app_config.toml → fallback"""
    v = os.getenv(name)
    if v is not None:
        return v
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return _CFG.get(name, default)

# -----------------------------
# Secrets / keys
# -----------------------------
SUPABASE_URL = cfg("SUPABASE_URL")
SUPABASE_ANON_KEY = cfg("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = cfg("SUPABASE_SERVICE_ROLE_KEY", None)
APP_BASE_URL = cfg("APP_BASE_URL", "")
ANTHROPIC_API_KEY = cfg("ANTHROPIC_API_KEY")

# --- Brand palette (from v6) ---
BRAND_BLUE   = "#445DA7"  # Original
BRAND_PURPLE = "#6B3894"  # Improved
BRAND_NAVY   = "#2E3C71"
COMP_TEAL    = "#2AA9A1"  # Competitor
HEATMAP_COLORSCALE = [
    [0.0, 'rgb(128,0,128)'],  # purple low
    [0.5, 'rgb(255,255,255)'],# white mid
    [1.0, 'rgb(0,0,128)']     # navy high
]
CORR_COLORSCALE = HEATMAP_COLORSCALE  # keep same look

# Fail fast for critical secrets used at runtime
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Missing Supabase URL or ANON key. Set them once in Streamlit → Secrets or .streamlit/secrets.toml locally.")
    st.stop()
if not ANTHROPIC_API_KEY:
    st.error("Missing ANTHROPIC_API_KEY. Add it to Streamlit → Secrets or .streamlit/secrets.toml locally.")
    st.stop()

# Optional, for local password fallback (kept for compatibility)
AUTH_MODE         = cfg("AUTH_MODE", "PASSWORD")
APP_PASSWORD      = cfg("APP_PASSWORD", "ChangeMe")
COMPANY_LOGO_URL  = cfg("COMPANY_LOGO_URL", "")

# -----------------------------
# Supabase client
# -----------------------------
@st.cache_resource
def sb_client(kind: str = "anon") -> Client:
    if kind == "service":
        if not SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError("SERVICE ROLE key is required for admin actions (set once in Secrets).")
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# -----------------------------
# Session helpers
# -----------------------------
def init_session():
    ss = st.session_state
    ss.setdefault("sb_session", None)
    ss.setdefault("sb_user", None)
    ss.setdefault("org", None)       # dict: {id, name, logo_url}
    ss.setdefault("org_id", None)
    ss.setdefault("role", None)

def maybe_refresh_session():
    ss = st.session_state
    try:
        sess = ss.get("sb_session")
        if sess and getattr(sess, "refresh_token", None):
            new = sb_client().auth.refresh_session(sess)
            ss["sb_session"] = new.session
            ss["sb_user"] = new.user
    except Exception:
        pass

def _fetch_active_membership(uid: str):
    res = sb_client().table("memberships")         .select("organization_id, role, status")         .eq("auth_user_id", uid).eq("status", "active")         .limit(1).single().execute()
    if not res.data:
        return None, None
    return res.data["organization_id"], res.data["role"]

def _fetch_org(org_id: str):
    res = sb_client().table("organizations")         .select("id, name, logo_url").eq("id", org_id)         .single().execute()
    return res.data

# -----------------------------
# Auth views & guards
# -----------------------------
def login_view():
    st.title("Sign in")
    email = st.text_input("Email", key="sb_email")
    pwd = st.text_input("Password", type="password", key="sb_pwd")
    c1, c2 = st.columns(2)
    if c1.button("Sign in"):
        try:
            out = sb_client().auth.sign_in_with_password({"email": email, "password": pwd})
            st.session_state.sb_session = out.session
            st.session_state.sb_user = out.user
            _post_login_bootstrap()
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
    if c2.button("Forgot password?"):
        try:
            sb_client().auth.reset_password_for_email(email)
            st.info("Password reset email sent (if the email exists).")
        except Exception as e:
            st.error(f"Could not send reset email: {e}")

def _post_login_bootstrap():
    ss = st.session_state
    u = ss.sb_user
    if not u:
        return
    # Email verification gate
    if not getattr(u, "email_confirmed_at", None):
        st.warning("Please verify your email to continue.")
        if st.button("Resend verification email"):
            try:
                sb_client().auth.resend({"email": u.email, "type": "signup"})
                st.success("Verification email resent.")
            except Exception as e:
                st.error(f"Resend failed: {e}")
        st.stop()

    org_id, role = _fetch_active_membership(u.id)
    if not org_id:
        st.error("No active membership found for this user.")
        st.stop()
    ss.org_id, ss.role = org_id, role
    ss.org = _fetch_org(org_id)


def require_auth():
    init_session()
    # If we're handling a password recovery link, route there and stop.
    try:
        q = _get_query_params()
    except Exception:
        q = {}
    if (q.get('type') == 'recovery') or q.get('code') or (q.get('access_token') and q.get('refresh_token')):
        _password_recovery_view()

    maybe_refresh_session()
    if not st.session_state.get("sb_user"):
        login_view()
        st.stop()
    if not st.session_state.get("org_id") or not st.session_state.get("role"):
        _post_login_bootstrap()
        if not st.session_state.get("org_id"):
            st.stop()


def require_role(allowed: set[str]):
    require_auth()
    if st.session_state.role not in allowed:
        st.error("You are not authorized to view this page.")
        st.stop()

def logout_button():
    if st.sidebar.button("Logout"):
        try:
            sb_client().auth.sign_out()
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()

def profile_menu():
    if not st.session_state.get("sb_user"):
        return
    u = st.session_state.sb_user
    org = st.session_state.get("org") or {}
    st.sidebar.markdown(f"**{org.get('name','')}**")
    st.sidebar.markdown(f"{getattr(u,'email','')} · _{st.session_state.get('role','')}_")
    if not getattr(u, "email_confirmed_at", None):
        st.sidebar.warning("Unverified email")
    logout_button()

# -----------------------------
# Admin pages (defined BEFORE router)
# -----------------------------
def _set_status(email: str, new_status: str, org_id: str, svc: Client):
    try:
        u = svc.auth.admin.list_users(email)
        if not u or not u.data.get("users"):
            st.error("User not found.")
            return
        uid = u.data["users"][0]["id"]
        sb_client().table("memberships")             .update({"status": new_status})             .match({"organization_id": org_id, "auth_user_id": uid}).execute()
        st.success(f"Status set to {new_status}.")
        st.rerun()
    except Exception as e:
        st.error(f"Status update failed: {e}")

def page_admin_users():
    require_role({"owner","admin"})
    st.title("Admin · Users")
    org_id = st.session_state.org_id
    svc = sb_client("service")

    # Load org memberships
    mem = sb_client().table("memberships")         .select("auth_user_id, role, status, created_at")         .eq("organization_id", org_id).execute().data or []

    # Map auth users (email, last signin)
    rows = []
    for m in mem:
        try:
            u = svc.auth.admin.get_user_by_id(m["auth_user_id"])
            email = u.user.email
            last_sign_in_at = getattr(u.user, "last_sign_in_at", None)
        except Exception:
            email, last_sign_in_at = "(unknown)", None
        rows.append({
            "email": email,
            "auth_user_id": m["auth_user_id"],
            "role": m["role"],
            "status": m["status"],
            "last_sign_in": last_sign_in_at,
        })
    st.dataframe(rows, hide_index=True)

    st.subheader("Invite user")
    with st.form("invite_form", clear_on_submit=True):
        email = st.text_input("Email")
        role = st.selectbox("Role", ["admin","analyst","viewer","client"])
        submitted = st.form_submit_button("Send invite")
        if submitted:
            try:
                svc.auth.admin.invite_user_by_email(email)
                st.success("Invite sent. When the user accepts, attach membership below.")
            except Exception as e:
                st.error(f"Invite failed: {e}")

    st.subheader("Attach or update membership")
    with st.form("attach_form"):
        email2 = st.text_input("User email")
        role2 = st.selectbox("New role", ["owner","admin","analyst","viewer","client"], key="role2")
        status2 = st.selectbox("Status", ["active","invited","disabled"], key="status2")
        if st.form_submit_button("Upsert membership"):
            try:
                user = svc.auth.admin.list_users(email2)
                auth_id = None
                if user and getattr(user, "data", None) and user.data.get("users"):
                    auth_id = user.data["users"][0]["id"]
                if not auth_id:
                    st.error("No auth user for that email yet. Ask them to accept invite or sign up.")
                else:
                    sb_client().table("memberships").upsert({
                        "organization_id": org_id,
                        "auth_user_id": auth_id,
                        "role": role2,
                        "status": status2
                    }, on_conflict="organization_id,auth_user_id").execute()
                    st.success("Membership upserted.")
                    st.rerun()
            except Exception as e:
                st.error(f"Upsert failed: {e}")

    st.subheader("User actions")
    email3 = st.text_input("Target email")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Disable"):
        _set_status(email3, "disabled", org_id, svc)
    if c2.button("Enable"):
        _set_status(email3, "active", org_id, svc)
    if c3.button("Resend invite"):
        try:
            svc.auth.admin.invite_user_by_email(email3)
            st.success("Invite resent.")
        except Exception as e:
            st.error(f"Resend failed: {e}")
    if c4.button("Trigger password reset"):
        try:
            sb_client().auth.reset_password_for_email(email3)
            st.success("Reset email sent.")
        except Exception as e:
            st.error(f"Reset failed: {e}")

    st.subheader("Remove from org")
    email4 = st.text_input("Remove email")
    if st.button("Remove"):
        try:
            u = svc.auth.admin.list_users(email4)
            if u and u.data.get("users"):
                uid = u.data["users"][0]["id"]
                sb_client().table("memberships")                     .delete().match({"organization_id": org_id, "auth_user_id": uid}).execute()
                st.success("Membership removed.")
                st.rerun()
            else:
                st.error("User not found.")
        except Exception as e:
            st.error(f"Remove failed: {e}")

def page_org_settings():
    require_role({"owner","admin"})
    st.title("Organization Settings")
    org = st.session_state.get("org") or {}
    st.markdown(f"**Org:** {org.get('name','')}")

    current = org.get("logo_url")
    if current:
        st.image(current, width=180, caption="Current logo")

    file = st.file_uploader("Upload logo (PNG/SVG)", type=["png","svg"])
    if file and st.button("Upload/Replace"):
        path = f"{st.session_state.org_id}/logo.png"
        try:
            sb_client().storage.from_("org-assets").upload(
                path=path,
                file=file.getvalue(),
                file_options={"contentType": "image/png", "upsert": True},
            )
            public_url = sb_client().storage.from_("org-assets").get_public_url(path).get("publicUrl")
            public_url = f"{public_url}?v={int(time.time())}"
            sb_client().table("organizations").update({"logo_url": public_url})                 .eq("id", st.session_state.org_id).execute()
            st.session_state["org"] = _fetch_org(st.session_state.org_id)
            st.success("Logo updated.")
            st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")

# -----------------------------
# Persistence helpers for scoring
# -----------------------------
def db_insert_scored_item(entity_label: str, channel: str, variant: str,
                          text: str, creative_url: str | None, scores: dict):
    require_auth()
    org_id = st.session_state.org_id
    uid = (st.session_state.sb_user or {}).__dict__.get("id") if hasattr(st.session_state.sb_user, "__dict__") else None
    if uid is None and hasattr(st.session_state.sb_user, "id"):
        uid = st.session_state.sb_user.id

    ins = sb_client().table("scored_items").insert({
        "organization_id": org_id,
        "auth_user_id": uid,
        "entity_label": entity_label,
        "channel": channel,
        "variant": variant,
        "text": text,
        "creative_url": creative_url,
        "scores": scores
    }).execute()
    item_id = ins.data[0]["id"]

    rows = []
    for attr, payload in scores.items():
        if isinstance(payload, dict):
            sc = payload.get("score")
            ev = payload.get("evidence")
        else:
            sc, ev = payload, None
        rows.append({
            "scored_item_id": item_id,
            "organization_id": org_id,
            "entity_label": entity_label,
            "channel": channel,
            "variant": variant,
            "attribute": attr,
            "score": sc,
            "evidence": ev
        })
    if rows:
        sb_client().table("scored_item_attributes").insert(rows).execute()

def load_heatmap_df():
    require_auth()
    res = sb_client().table("v_channel_attribute_medians")         .select("*").eq("organization_id", st.session_state.org_id).execute()
    import pandas as pd
    return pd.DataFrame(res.data or [])

def load_trends_df():
    require_auth()
    res = sb_client().table("v_monthly_attr_trends")         .select("*").eq("organization_id", st.session_state.org_id).execute()
    import pandas as pd
    return pd.DataFrame(res.data or [])

# -----------------------------
# Main UI Router
# -----------------------------
require_auth()
profile_menu()

logo_url = (st.session_state.get("org") or {}).get("logo_url") or COMPANY_LOGO_URL
if logo_url:
    st.sidebar.image(logo_url, width=160)

page = st.sidebar.selectbox("Go to", ["Dashboard","Admin · Users","Org Settings"])

if page == "Admin · Users":
    page_admin_users()
    st.stop()
elif page == "Org Settings":
    page_org_settings()
    st.stop()

def _heatmap(fig_df: pd.DataFrame, title: str = "Attribute Importance Heatmap"):
    if fig_df.empty or len(fig_df.columns) <= 1 or "Channel" not in fig_df.columns:
        st.info("Not enough scored items to build a heatmap yet. Score at least one item.")
        return

    # matrix and axes
    z = fig_df.drop(columns=["Channel"]).values
    x = list(fig_df.columns.drop("Channel"))
    y = list(fig_df["Channel"])

    # use your constant OR inline the list (pick one)
    hm = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        zmin=0.0, zmax=1.0,
        colorscale=HEATMAP_COLORSCALE,   # <-- or replace with the 3-stop literal below
        colorbar=dict(title="Median Score"),
        hovertemplate="Channel: %{y}<br>Attribute: %{x}<br>Median: %{z:.2f}<extra></extra>"
    ))
    # If you prefer the explicit literal instead of the constant:
    # colorscale=[
    #     [0.0, 'rgb(128,0,128)'],   # purple (low)
    #     [0.5, 'rgb(255,255,255)'], # white  (mid)
    #     [1.0, 'rgb(0,0,128)']      # navy   (high)
    # ],

    hm.update_layout(title=title, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(hm, use_container_width=True)
    st.dataframe(fig_df, use_container_width=True, hide_index=True)

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
    st.plotly_chart(fig, width='stretch')
    wide = df_channel.pivot(index="Month", columns="Attribute", values="Score").reset_index()
    st.dataframe(wide, width='stretch')

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


st.subheader("Attribute Importance Heatmap")
hm_raw = load_heatmap_df()
if hm_raw.empty:
    st.info("No scored data yet. Score something to populate the heatmap.")
else:
    # pivot to Channel × Attribute with median values
    hm_raw = hm_raw.rename(columns={"channel":"Channel", "attribute":"Attribute", "median_score":"Score"})
    hm_raw["Attribute"] = hm_raw["Attribute"].apply(lambda s: s.replace("_"," ").replace("Cost_Benefit","Cost/Benefit"))
    pivot = hm_raw.pivot(index="Channel", columns="Attribute", values="Score").reset_index()
    _heatmap(pivot, title="Attribute Importance Heatmap (Median by Channel)")

# Trends from DB view# -----------------------------
# Dashboard (Brand Lift)

st.markdown("<h1>Brand Lift</h1>", unsafe_allow_html=True)

# Heatmap from DB view

st.subheader("Monthly trends (median by month)")
tr_raw = load_trends_df()
if tr_raw.empty:
    st.info("No monthly trend data yet.")
else:
    channels = sorted(tr_raw["Channel"].unique())
    trend_channel = st.selectbox("Select channel", channels, index=0, key="trend_channel")
    filtered = tr_raw[tr_raw["Channel"] == trend_channel][["Month","Attribute","Score"]]
    if filtered.empty:
        st.info("No trend data for the selected channel.")
    else:
        _plot_channel_trends(filtered)

