
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


import plotly.graph_objects as go
import anthropic
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
RECOVERY_BRIDGE_URL = cfg("RECOVERY_BRIDGE_URL", "")
ANTHROPIC_API_KEY = cfg("ANTHROPIC_API_KEY")

# ---------- Brand Theme ----------
BRAND_BLUE   = "#445DA7"  # Original
BRAND_PURPLE = "#6B3894"  # Improved
BRAND_NAVY   = "#2E3C71"
COMP_TEAL    = "#2AA9A1"  # Competitor

# Exact modern channels (kept for display consistency)
CHANNELS = ["CTV", "DOOH", "Youtube", "TikTok", "Google Ads", "Instagram", "X"]

# Internal attribute keys (stable). UI shows prettified labels with spaces/slashes.
ATTRS = ["Leadership", "Ease_of_Use", "Quality", "Luxury", "Cost_Benefit", "Trust"]

# Pretty label mapping for display
def _pretty_attr(a: str) -> str:
    return a.replace("_", " ").replace("Benefit", "Benefit").replace("Cost Benefit", "Cost/Benefit")

# Fail checks → switch to tolerant flags (v7-style for demo friendliness)
HAS_SB = bool(SUPABASE_URL and SUPABASE_ANON_KEY)
HAS_ANTH = bool(ANTHROPIC_API_KEY)
if not HAS_SB:
    st.warning("Supabase is not configured. Running in PASSWORD demo mode. Admin & DB features are disabled.")
if not HAS_ANTH:
    st.info("ANTHROPIC_API_KEY is not set. You can browse the UI, but scoring won't run until you add the key.")
if not ANTHROPIC_API_KEY:
    st.warning('ANTHROPIC_API_KEY is missing. Scoring buttons will be disabled until you add the key.')
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
# Mode selection (v7-style tolerant auth)
# -----------------------------
DEMO_MODE = (AUTH_MODE.upper() == "PASSWORD") or (not HAS_SB)
AUTH_MODE_RESOLVED = "PASSWORD_DEMO" if DEMO_MODE else "SUPABASE"

# Password demo gate (no Supabase)

# Password demo gate (no Supabase) — now with v6-style test account preload
def _password_demo_gate():
    st.sidebar.subheader("Demo access")
    username = st.sidebar.text_input("Username (try: TheMarketingScientist)", key="demo_user")
    pw = st.sidebar.text_input("App password", type="password", key="pwd_demo")
    client = st.sidebar.text_input("Client name (shows in header)", key="demo_client", value=st.session_state.get("client_name","Demo Client"))
    logo_url = st.sidebar.text_input("Company logo URL (optional)", key="demo_logo", value=st.session_state.get("company_logo_url",""))
    ok = st.sidebar.button("Enter demo") or st.session_state.get("demo_authed", False)

    if ok:
        # Special test account behavior (from v6): username+password both "TheMarketingScientist"
        if (username.strip() == "TheMarketingScientist" and pw == "TheMarketingScientist") or st.session_state.get("demo_authed", False):
            st.session_state["demo_authed"] = True
            st.session_state["authed"] = True
            st.session_state["auth_user"] = "TheMarketingScientist"
            st.session_state["client_name"] = client.strip() or "Demo Client"
            if logo_url.strip():
                st.session_state["company_logo_url"] = logo_url.strip()
            st.session_state["preload_demo"] = True  # seed demo data
            return True
        # Generic password mode (APP_PASSWORD)
        expected = APP_PASSWORD
        if pw == expected:
            st.session_state["demo_authed"] = True
            st.session_state["authed"] = True
            st.session_state["auth_user"] = username.strip() or "password-user"
            st.session_state["client_name"] = client.strip() or "Client"
            if logo_url.strip():
                st.session_state["company_logo_url"] = logo_url.strip()
            st.session_state["preload_demo"] = False  # no seeding for generic demo users
            return True

        st.sidebar.error("Wrong credentials")
    return False
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
            try:
                _post_login_bootstrap()
                st.rerun()
            except Exception:
                st.info("Password updated. Please sign in with your new password.")
                try:
                    st.experimental_set_query_params()
                except Exception:
                    pass
                st.stop()
        except Exception as e:
            st.error(f"Login failed: {e}")
if c2.button("Forgot password?"):
    if not email:
        st.warning("Enter your email above first.")
    else:
        try:
            # Prefer your GitHub Pages bridge; fall back to app URL
            redirect = RECOVERY_BRIDGE_URL or APP_BASE_URL

            kwargs = {}
            if redirect:
                kwargs["options"] = {"redirect_to": redirect}

            sb_client().auth.reset_password_for_email(email, **kwargs)
            st.success("If that email exists, we sent a reset link.")
            if not redirect:
                st.info("Tip: set RECOVERY_BRIDGE_URL or APP_BASE_URL so the email link returns to your app.")
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
    """
    Unified guard:
      1) Initialize session state
      2) If URL indicates a password recovery flow, render the recovery view and stop
      3) If DEMO_MODE → handle password gate
      4) Else require Supabase session + membership
    """
    init_session()

    # (1) Handle password recovery links early
    try:
        q = _get_query_params()
    except Exception:
        q = {}

    if (q.get('type') == 'recovery') or q.get('code') or (q.get('access_token') and q.get('refresh_token')):
        _password_recovery_view()
        st.stop()

    # (2) Refresh session if possible (no-op if not applicable)
    maybe_refresh_session()

    # (3) Demo mode: simple password gate
    if DEMO_MODE:
        if not _password_demo_gate():
            st.stop()
        return

    # (4) Supabase auth required
    if not st.session_state.get("sb_user"):
        login_view()
        st.stop()
    if not st.session_state.get("org_id") or not st.session_state.get("role"):
        _post_login_bootstrap()
        if not st.session_state.get("org_id"):
            st.stop()

def require_role(allowed: set[str]):
    if DEMO_MODE:
        st.error("Admin/role-restricted pages are disabled in PASSWORD demo mode.")
        st.stop()
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
    if DEMO_MODE:
        st.sidebar.markdown("**Demo Mode**")
        st.sidebar.markdown("demo@brandlift.local · _viewer_")
        return
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
    if DEMO_MODE:
        import pandas as pd
        df = st.session_state.get('demo_heatmap')
        return df if df is not None else pd.DataFrame()
    require_auth()
    res = sb_client().table("v_channel_attribute_medians").select("*").eq("organization_id", st.session_state.org_id).execute()
    import pandas as pd
    return pd.DataFrame(res.data or [])

def load_trends_df():
    if DEMO_MODE:
        import pandas as pd
        df = st.session_state.get('demo_trends')
        return df if df is not None else pd.DataFrame()
    require_auth()
    res = sb_client().table("v_monthly_attr_trends").select("*").eq("organization_id", st.session_state.org_id).execute()
    import pandas as pd
    return pd.DataFrame(res.data or [])

# -----------------------------
# Main UI Router
if DEMO_MODE:
    # Only Dashboard available in demo
    page = "Dashboard"
    if COMPANY_LOGO_URL:
        st.sidebar.image(COMPANY_LOGO_URL, width=160)
    st.sidebar.info("Demo mode: Admin & Org Settings disabled.")
else:
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
    st.plotly_chart(hm, width='stretch')
    st.dataframe(fig_df, width='stretch')

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

def scores_table(scores: dict):
    rows = []
    for k in ATTRS:
        rows.append({
            "Attribute": _pretty_attr(k),
            "Score": round(float(scores[k]["score"]), 2),
            "Key phrase": scores[k]["evidence"]
        })
    return pd.DataFrame(rows)

def delta_table(base: dict, improved: dict):
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


# ---------------- LLM Scoring Helpers ----------------
PREFERRED_MODEL = "claude-3-5-sonnet-latest"
FALLBACK_MODEL = "claude-3-opus-latest"

SYSTEM_SCORE = """You are a brand perception rater. Score ad copy on 6 attributes:

"""
SYSTEM_REWRITE = "You are a precise brand copy editor focused on targeted brand attributes."
SYSTEM_IDEA = "You are a senior creative strategist. Propose concise alternative ad copy ideas."

def rewrite_to_targets(api_key: str, base_text: str, targets: list[str]) -> str:
    t = ", ".join(targets) if targets else "Leadership"
    instr = f"""Rewrite the following ad copy to improve: {t}.
Keep the tone and product intact. 55 words max.
Return only the rewritten copy, with no preamble or explanations.
Original:
{base_text}"""
    def run(model, api_key_inner, uc):
        out_text = _call_messages(api_key_inner, SYSTEM_REWRITE, uc, model)
        return out_text.strip()
    rewrite, _ = _with_fallback(run, api_key, instr)
    return rewrite or base_text

def propose_new_ideas(api_key: str, base_text: str, targets: list[str], n: int = 2) -> list[str]:
    t = ", ".join(targets) if targets else "Leadership"
    prompt = f"""Given this ad copy, propose {n} distinct alternative ideas that could outperform a competitor
on the following target attributes: {t}.
Keep each idea under 35 words. Return each idea as a separate paragraph.
Original:
{base_text}"""
    def run(model, api_key_inner, uc):
        return _call_messages(api_key_inner, SYSTEM_IDEA, uc, model)
    out, _ = _with_fallback(run, api_key, prompt)
    # Split on blank lines or bullet-like markers, keep non-empty lines
    ideas = [s.strip("-• ").strip() for s in out.split("\n") if s.strip()]
    # Deduplicate and keep first n
    seen, dedup = set(), []
    for idea in ideas:
        if idea not in seen:
            seen.add(idea)
            dedup.append(idea)
        if len(dedup) >= n:
            break
    return dedup
# Leadership, Ease_of_Use, Quality, Luxury, Cost_Benefit, Trust.
# Each score is a float in [0,1]. Anchors: 0.2 = weak, 0.5 = moderate, 0.8 = strong.
# Return STRICT JSON with each attribute as {"score": float, "evidence": "short phrase (<=12 words)"}.
# No extra text.
# """

def _schema():
    return {
        "type": "object",
        "properties": {k: {"type":"object","properties":{
            "score":{"type":"number"}, "evidence":{"type":"string"}
        }} for k in ATTRS},
        "required": ATTRS
    }

def _parse_json_block(text: str) -> dict:
    import json, re
    # try to find first JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise RuntimeError("No JSON found in model output")
    obj = json.loads(m.group(0))
    return obj

def _call_messages(api_key: str, system_prompt: str, user_content: str, model: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1200,
        system=system_prompt,
        messages=[{"role":"user", "content": user_content}]
    )
    parts = []
    for p in msg.content:
        if hasattr(p, "text"):
            parts.append(p.text)
        elif isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text",""))
    return "\n".join(parts)

def _with_fallback(func, *args, **kwargs):
    try:
        return func(PREFERRED_MODEL, *args, **kwargs), PREFERRED_MODEL
    except Exception as e:
        s = str(e)
        if ("403" in s) or ("404" in s) or ("model not found" in s.lower()):
            return func(FALLBACK_MODEL, *args, **kwargs), FALLBACK_MODEL
        raise

def score_text(api_key: str, text: str) -> dict:
    user_content = f"""Text:
{text}

Output JSON schema:
{{json.dumps(_schema())}}"""
    def run(model, api_key_inner, uc):
        out_text = _call_messages(api_key_inner, SYSTEM_SCORE, uc, model)
        scores = _parse_json_block(out_text)
        for k in ATTRS:
            s = float(scores[k]["score"])
            scores[k]["score"] = max(0.0, min(1.0, s))
            if "evidence" not in scores[k] or not scores[k]["evidence"]:
                scores[k]["evidence"] = ""
        return scores
    scores, _ = _with_fallback(run, api_key, user_content)
    return scores


# ---------------- Persistence of scored items ----------------
def db_insert_scored_item(entity_label: str, channel: str, variant: str,
                          text_value: str, creative_url: str | None, scores: dict):
    # Only in full Supabase mode
    if DEMO_MODE or not HAS_SB:
        return
    require_auth()
    org_id = st.session_state.org_id
    uid = getattr(st.session_state.get("sb_user"), "id", None)
    try:
        ins = sb_client().table("scored_items").insert({
            "organization_id": org_id,
            "auth_user_id": uid,
            "entity_label": entity_label,
            "channel": channel,
            "variant": variant,
            "text": text_value,
            "creative_url": creative_url,
            "scores": scores
        }).execute()
        item_id = ins.data[0]["id"] if ins and ins.data else None
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
    except Exception as e:
        st.warning(f"Could not persist score to DB: {e}")


# ---------------- Demo data seeding (in-memory, demo mode only) ----------------
def _seed_demo_data():
    """Create small demo heatmap/trend datasets for PASSWORD_DEMO with preload_demo=True."""
    if not st.session_state.get("preload_demo"):
        return
    if st.session_state.get("demo_seeded"):
        return
    import pandas as pd
    # Heatmap-style medians by Channel × Attribute
    demo_rows_hm = []
    demo_channels = CHANNELS[:4] if len(CHANNELS) >= 4 else CHANNELS
    demo_attrs = ATTRS
    import random
    rng = random.Random(42)
    for ch in demo_channels:
        base = rng.uniform(0.45, 0.65)
        for a in demo_attrs:
            val = max(0.0, min(1.0, base + rng.uniform(-0.12, 0.18)))
            demo_rows_hm.append({"Channel": ch, "Attribute": _pretty_attr(a), "Score": round(val, 3)})
    st.session_state["demo_heatmap"] = pd.DataFrame(demo_rows_hm)

    # Trends: last 6 months synthetic per selected channel
    from datetime import datetime, timedelta
    today = datetime.utcnow().replace(day=1)
    months = []
    for i in range(5, -1, -1):
        m = (today.replace(day=1) - timedelta(days=30*i)).strftime("%Y-%m")
        months.append(m)
    demo_rows_tr = []
    for ch in demo_channels[:2]:
        for a in demo_attrs[:4]:
            base = rng.uniform(0.4, 0.7)
            for idx, m in enumerate(months):
                drift = (idx - len(months)/2) * 0.02
                val = max(0.0, min(1.0, base + drift + rng.uniform(-0.05, 0.05)))
                demo_rows_tr.append({"Month": m, "Attribute": _pretty_attr(a), "Score": round(val, 3), "Channel": ch})
    st.session_state["demo_trends"] = pd.DataFrame(demo_rows_tr)
    st.session_state["demo_seeded"] = True

# -----------------------------
# Dashboard (Brand Lift)

st.markdown(f"<h1>Brand Lift — {st.session_state.get('client_name', 'Demo')}</h1>", unsafe_allow_html=True)
if DEMO_MODE and st.session_state.get("preload_demo"):
    _seed_demo_data()


# ================= Scoring & Competitor Comparison =================
st.subheader("Score your copy & compare to a competitor")

# Context selectors for persistence/labeling
with st.expander("Context (used for labels & DB persistence)"):
    ccol1, ccol2 = st.columns([2,1])
    entity_label = ccol1.text_input("Entity / Brand label", value=st.session_state.get("entity_label", "My Brand"))
    channel_sel = ccol2.selectbox("Channel", CHANNELS, index=0, key="bl_channel_sel")
    st.session_state["entity_label"] = entity_label


col1, col2 = st.columns(2)
with col1:
    base_text = st.text_area("Your ad copy", key="bl_base_text", height=160, placeholder="Paste your ad copy here...")
    if st.button("Score my copy", use_container_width=True):
        if not ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY is missing. Add it to secrets or environment.")
        elif not base_text.strip():
            st.warning("Please paste some text to score.")
        else:
            with st.spinner("Scoring your copy..."):
                st.session_state['scores_base'] = score_text(ANTHROPIC_API_KEY, base_text.strip())
        db_insert_scored_item(entity_label, st.session_state.get('bl_channel_sel','CTV'), 'Original', base_text.strip(), None, st.session_state['scores_base'])

with col2:
    comp_text = st.text_area("Competitor copy (paste text)", key="bl_comp_text", height=160, placeholder="Paste competitor ad copy or transcript...")
    if st.button("Score competitor", use_container_width=True, key="btn_score_comp"):
        if not ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY is missing. Add it to secrets or environment.")
        elif not comp_text.strip():
            st.warning("Please paste competitor text to score.")
        else:
            with st.spinner("Scoring competitor copy..."):
                st.session_state['scores_comp'] = score_text(ANTHROPIC_API_KEY, comp_text.strip())
        db_insert_scored_item(entity_label, st.session_state.get('bl_channel_sel','CTV'), 'Competitor', comp_text.strip(), None, st.session_state['scores_comp'])

# Show comparison once available
scores_base = st.session_state.get('scores_base')
scores_comp  = st.session_state.get('scores_comp')

if scores_base or scores_comp:
    st.markdown("### Visual comparison")
    c1, c2 = st.columns(2)
    if scores_base:
        fig1 = radar(scores_base, "Your Copy", fill_color=BRAND_BLUE+"33", line_color=BRAND_BLUE)
        c1.plotly_chart(fig1, use_container_width=True)
    if scores_comp:
        fig2 = radar(scores_comp, "Competitor", fill_color=COMP_TEAL+"33", line_color=COMP_TEAL)
        c2.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Tables")
    tcol1, tcol2 = st.columns(2)
    if scores_base:
        tcol1.dataframe(scores_table(scores_base), use_container_width=True, hide_index=True)
    if scores_base and scores_comp:
        tcol2.dataframe(delta_table(scores_base, scores_comp), use_container_width=True, hide_index=True)

    # Competitive gaps
    if scores_base and scores_comp:
        worse = []
        for k in ATTRS:
            if float(scores_comp[k]["score"]) > float(scores_base[k]["score"]) + 0.02:
                worse.append(( _pretty_attr(k),
                               round(float(scores_base[k]["score"]),2),
                               round(float(scores_comp[k]["score"]),2)))
        if worse:
            st.markdown("#### Competitive gaps")
            for name, b, c in worse:
                st.write(f"- {name}: you {b} vs competitor {c}")
        else:
            st.markdown("#### Competitive gaps")
            st.success("No clear gaps detected. You're matching or beating the competitor on most attributes.")

# Heatmap from DB view

# ================= Improve (Rewrite & New Ideas) =================
with st.expander("Improve your copy"):
    targets = st.multiselect("Target attributes to improve", [_pretty_attr(a) for a in ATTRS])
    # Map pretty labels back to keys
    target_keys = [t.replace(" ", "_").replace("Cost/Benefit","Cost_Benefit") for t in targets]
    colA, colB = st.columns(2)
    with colA:
        if st.button("Rewrite toward targets", use_container_width=True, key="btn_rewrite"):
            if not ANTHROPIC_API_KEY:
                st.error("ANTHROPIC_API_KEY is missing.")
            elif not st.session_state.get("bl_base_text") and not 'base_text' in locals():
                st.warning("Score or enter base text first.")
            else:
                base = st.session_state.get("bl_base_text") or ""
                with st.spinner("Generating rewrite..."):
                    rewritten = rewrite_to_targets(ANTHROPIC_API_KEY, base, target_keys)
                st.text_area("Rewritten copy", value=rewritten, height=160, key="improved_text")
                if st.button("Score improved rewrite", key="btn_score_improved"):
                    if not rewritten.strip():
                        st.warning("Nothing to score.")
                    else:
                        st.session_state['scores_improved'] = score_text(ANTHROPIC_API_KEY, rewritten.strip())
                        # Persist to DB if available
                        try:
                            db_insert_scored_item(
                                st.session_state.get('entity_label', 'My Brand'),
                                st.session_state.get('bl_channel_sel','CTV'),
                                "Improved",
                                rewritten.strip(),
                                None,
                                st.session_state['scores_improved']
                            )
                        except Exception as e:
                            st.warning(f"Could not persist score to DB: {e}")
                        st.success("Improved rewrite scored and saved (if DB enabled).")
    with colB:
        if st.button("Propose new ideas (2)", use_container_width=True, key="btn_ideas"):
            if not ANTHROPIC_API_KEY:
                st.error("ANTHROPIC_API_KEY is missing.")
            else:
                base = st.session_state.get("bl_base_text") or ""
                with st.spinner("Brainstorming..."):
                    ideas = propose_new_ideas(ANTHROPIC_API_KEY, base, target_keys, n=2)
                for i, idea in enumerate(ideas, 1):
                    st.markdown(f"**Idea {i}:** {idea}")

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

# Trends from DB view
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

def _seed_demo_trends():
    if not DEMO_MODE:
        return
    import pandas as pd, random
    if st.session_state.get("monthly_attr_trends") is not None:
        return
    rng = random.Random(123)
    from datetime import datetime, timedelta
    today = datetime.utcnow().replace(day=1)
    months = [(today - timedelta(days=30*i)).strftime("%Y-%m") for i in range(5,-1,-1)]
    rows = []
    for ch in CHANNELS[:3]:
        for a in ATTRS[:5]:
            base = rng.uniform(0.45, 0.65)
            for idx, m in enumerate(months):
                drift = (idx - len(months)/2) * 0.02
                val = max(0.0, min(1.0, base + drift + rng.uniform(-0.05, 0.05)))
                rows.append({"Month": m, "Attribute": _pretty_attr(a), "Score": round(val, 3), "Channel": ch, "Entity": "Demo", "Variant": "Original"})
    st.session_state["monthly_attr_trends"] = pd.DataFrame(rows)

def render_correlation_section():
    return
