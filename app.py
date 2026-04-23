"""
CLUTCH – Elite FPL Strategy Dashboard
Dark-mode Streamlit UI with Win Probability Gauge, Assassin Feed, Presser Summary.
"""

import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor

import fpl_api
import simulator as sim
import nlp_utils  # uses OpenRouter (meta-llama/llama-3.3-70b-instruct)

# Resolve API key: Streamlit Cloud secrets → env var → empty (shows input field)
def _resolve_api_key() -> str:
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.getenv("OPENROUTER_API_KEY", "")


@st.cache_data(ttl=300)
def _fetch_fpl_data(gw: int) -> tuple[dict, pd.DataFrame, dict]:
    """Fetch bootstrap once per 5-minute window; derive players_df and gw_info from it."""
    bs      = fpl_api.get_bootstrap()
    pdf     = fpl_api.get_players_df(bs)
    gw_info = fpl_api.get_gw_info(gw, bs)
    return bs, pdf, gw_info


@st.cache_data(ttl=300)
def _fetch_standings(league_id: int) -> list[dict]:
    return fpl_api.get_top_n_managers(league_id, n=5)


@st.cache_data(ttl=300)
def _fetch_picks(manager_id: int, gw: int) -> list[int]:
    return fpl_api.get_manager_picks(manager_id, gw)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CLUTCH – FPL Strategy Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colors and base theme (backgroundColor, primaryColor, textColor) are set in
# .streamlit/config.toml. Only inject CSS that config.toml cannot express.
DARK_CSS = """
<style>
  .block-container { padding: 1.5rem 2rem; }
  h1 { font-family: 'Courier New', monospace; letter-spacing: 3px; }
  h2, h3 { color: #b0bec5; }
  .metric-card {
    background: #1a1a2e; border: 1px solid #00e5ff33;
    border-radius: 12px; padding: 1rem; text-align: center;
  }
  .assassin-row { border-left: 3px solid #ff1744; padding-left: 8px; margin-bottom: 4px; }
  .stButton>button { background: #00e5ff22; border: 1px solid #00e5ff; color: #00e5ff; }
  .stButton>button:hover { background: #00e5ff44; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key in ("bootstrap", "players_df", "sim_results", "assassins", "pressers", "gw_info",
            "transfer_result", "target_profile", "leader_profile", "sim_cfg", "captain_id"):
    if key not in st.session_state:
        st.session_state[key] = None


def _valid_df(key: str, required_cols: tuple = ()) -> "pd.DataFrame | None":
    """Return session DataFrame only if it's a non-empty DataFrame with expected columns."""
    val = st.session_state.get(key)
    if not isinstance(val, pd.DataFrame) or val.empty:
        return None
    if required_cols and not all(c in val.columns for c in required_cols):
        return None
    return val

# ---------------------------------------------------------------------------
# Sidebar – Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚡ CLUTCH Config")
    target_id  = st.number_input("Your Manager ID", min_value=1, value=1234567, step=1)
    league_id  = st.number_input("Mini-League ID",  min_value=1, value=314,     step=1)
    gameweek   = st.number_input("Current GW",      min_value=1, max_value=38, value=32, step=1)
    remaining  = st.number_input("GWs Remaining",   min_value=1, max_value=20, value=6,  step=1)
    n_iter     = st.select_slider("MC Iterations", options=[1000, 5000, 10000], value=10000)

    st.markdown("---")
    st.markdown("### Presser Analyzer")
    gw_info = st.session_state.gw_info
    if gw_info:
        mins = gw_info.get("minutes_to_deadline")
        tag  = "🟥 DGW" if gw_info["is_dgw"] else "🟨 BGW" if gw_info["is_bgw"] else "🟩 Normal"
        urgency_color = "#ff1744" if mins and mins < 60 else "#ff9800" if mins and mins < 360 else "#00e676"
        deadline_str  = gw_info.get("deadline_human", "")
        mins_str      = f"{mins}m left" if mins is not None and mins > 0 else "LOCKED"
        st.markdown(
            f"<div style='background:#1a1a2e;border:1px solid {urgency_color}33;"
            f"border-radius:8px;padding:6px 10px;font-size:12px;margin-bottom:8px'>"
            f"<b style='color:{urgency_color}'>GW{gw_info['gw']} {tag}</b><br>"
            f"<span style='color:#b0bec5'>⏰ {deadline_str}</span><br>"
            f"<span style='color:{urgency_color}'>{mins_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    presser_player = st.text_input("Player name", placeholder="e.g. Salah")
    presser_pid    = st.number_input("Player ID (FPL)", min_value=0, value=0, step=1)
    presser_quote  = st.text_area("Paste press conference quote", height=100)
    anthropic_key  = st.text_input("OpenRouter API Key", type="password",
                                    value=_resolve_api_key())
    run_presser    = st.button("Analyze Quote")

    st.markdown("---")
    # BGW Alert — count how many of the target squad are blanking
    _gw   = st.session_state.gw_info
    _tp2  = st.session_state.target_profile
    _pdf2 = _valid_df("players_df", ("id", "team", "web_name"))
    if _gw and _gw.get("is_bgw") and _tp2 and _pdf2 is not None:
        bgw_t    = _gw.get("bgw_teams", set())
        bgw_sq   = _pdf2[
            (_pdf2["id"].isin(_tp2.squad_player_ids)) & (_pdf2["team"].isin(bgw_t))
        ]
        bgw_cnt  = len(bgw_sq)
        bgw_names = ", ".join(bgw_sq["web_name"].tolist()[:4])
        if bgw_cnt:
            st.markdown(
                f"<div style='background:#ff174422;border:1px solid #ff1744;"
                f"border-radius:8px;padding:8px 12px;margin-bottom:8px'>"
                f"<b style='color:#ff1744'>❌ BGW Alert</b><br>"
                f"<span style='color:#ffcdd2;font-size:12px'>"
                f"You have <b>{bgw_cnt}</b> blanking player{'s' if bgw_cnt>1 else ''} "
                f"this GW: {bgw_names}{'...' if bgw_cnt>4 else ''}.<br>"
                f"Use the Transfer Lab to find replacements.</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Captain Selector")
    _tp  = st.session_state.target_profile
    _pdf = _valid_df("players_df", ("id", "ppg", "web_name"))
    if _tp is not None and _pdf is not None:
        _squad_cap = _pdf[_pdf["id"].isin(_tp.squad_player_ids)].sort_values("ppg", ascending=False)
        _cap_names = _squad_cap["web_name"].tolist()
        _cap_ids   = _squad_cap["id"].tolist()
        _cur_cap   = st.session_state.captain_id
        _def_idx   = _cap_ids.index(_cur_cap) if _cur_cap in _cap_ids else 0
        _cap_sel   = st.selectbox("Choose your captain", _cap_names, index=_def_idx, key="cap_sel")
        st.session_state.captain_id = int(_cap_ids[_cap_names.index(_cap_sel)])
    else:
        st.caption("Run CLUTCH once to unlock captain selector.")

    st.markdown("---")
    run_btn = st.button("🚀 RUN CLUTCH", use_container_width=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# ⚡ C L U T C H")
st.caption("Elite FPL Strategy Engine – Monte Carlo | Assassin | Coach-Speak AI")

col_gauge, col_assassin, col_presser = st.tabs(["📊 Win Probability", "🎯 Assassin Feed", "🧠 Presser Intel"])

# ---------------------------------------------------------------------------
# Helper: Win Probability Gauge
# ---------------------------------------------------------------------------
def draw_gauge(win_prob: float, target_name: str, leader_name: str) -> go.Figure:
    color = "#ff1744" if win_prob < 20 else "#ff9800" if win_prob < 50 else "#00e676"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=win_prob,
        title={"text": f"Win Prob vs {leader_name}", "font": {"color": "#b0bec5"}},
        number={"suffix": "%", "font": {"color": color, "size": 48}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar":  {"color": color},
            "bgcolor": "#1a1a2e",
            "steps": [
                {"range": [0,  20], "color": "#200000"},
                {"range": [20, 50], "color": "#1a1500"},
                {"range": [50,100], "color": "#001a0d"},
            ],
            "threshold": {"line": {"color": "#00e5ff", "width": 3}, "value": 50},
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d0d", font_color="#e0e0e0",
        height=300, margin=dict(t=40, b=0, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: Assassin Table
# ---------------------------------------------------------------------------
def draw_assassin_table(df: pd.DataFrame) -> go.Figure:
    top = df.head(10).copy()

    def _name_label(row):
        bgw_flag = " ❌" if row.get("has_bgw", False) else ""
        return f"{row['web_name']}{bgw_flag}"

    top["display_name"] = top.apply(_name_label, axis=1)

    fig = go.Figure(go.Table(
        header=dict(
            values=["Player", "xGI", "Own%", "Rivals Own", "Action"],
            fill_color="#1a1a2e", font=dict(color="#00e5ff", size=12),
            align="left",
        ),
        cells=dict(
            values=[
                top["display_name"],
                top["xGI"].round(2),
                top["ownership_pct"].round(1),
                top["rival_ownership_count"],
                top["target_owns"].map({True: "✅ Hold", False: "🔴 BUY"}),
            ],
            fill_color=[["#0d0d0d", "#111122"] * 10],
            font=dict(color="#e0e0e0", size=11),
            align="left",
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d0d", margin=dict(t=10, b=0, l=0, r=0), height=320
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: GW xP Sparkline
# ---------------------------------------------------------------------------
def draw_sparkline(result: dict) -> go.Figure:
    gws = list(range(1, result["remaining_gws"] + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gws, y=result["target_gw_xp"], mode="lines+markers",
        name=result["target_name"], line=dict(color="#00e5ff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=gws, y=result["leader_gw_xp"], mode="lines+markers",
        name=result["leader_name"], line=dict(color="#ff1744", width=2, dash="dash"),
    ))
    fig.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111122",
        font_color="#b0bec5", height=200,
        margin=dict(t=20, b=30, l=30, r=10),
        legend=dict(bgcolor="#0d0d0d"),
        xaxis=dict(title="GW", color="#555"),
        yaxis=dict(title="xP", color="#555"),
    )
    return fig


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------
if run_btn:
    with st.spinner("Fetching FPL data..."):
        try:
            bs, pdf, gw_info = _fetch_fpl_data(gameweek)
            st.session_state.bootstrap  = bs
            st.session_state.players_df = pdf
            st.session_state.gw_info    = gw_info

            # Manager info + target picks — cached per (manager_id, gw)
            t_info  = fpl_api.get_manager_info(target_id)
            t_name  = f"{t_info['player_first_name']} {t_info['player_last_name']}"
            t_pts   = t_info["summary_overall_points"]
            t_picks = _fetch_picks(target_id, gameweek)

            # League standings — cached per league_id
            top5     = _fetch_standings(league_id)
            rivals   = [m for m in top5 if m["entry"] != target_id]
            leader_m = top5[0] if top5[0]["entry"] != target_id else top5[1]

            # Build rival profiles — _fetch_picks is cached so each manager_id
            # is fetched once and reused everywhere in this run
            def _profile(m: dict) -> sim.ManagerProfile:
                picks = _fetch_picks(m["entry"], gameweek)
                return sim.ManagerProfile(
                    manager_id=m["entry"],
                    name=m["player_name"],
                    current_points=m["total"],
                    squad_player_ids=picks,
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                rival_profiles = list(pool.map(_profile, rivals))

            # Extract rival picks from already-built profiles (zero extra requests)
            rival_picks_dict = {p.manager_id: p.squad_player_ids for p in rival_profiles}

            # leader_profile already exists in rival_profiles — no extra fetch
            leader_profile = next(
                p for p in rival_profiles if p.manager_id == leader_m["entry"]
            )

            # Assassin feed — all picks forwarded, no internal re-fetching
            assassins = fpl_api.compute_ownership_gap(
                target_id=target_id, league_id=league_id,
                gameweek=gameweek, players_df=pdf,
                bgw_teams=gw_info.get("bgw_teams", set()),
                top5_standings=top5,
                target_picks=t_picks,
                rival_picks=rival_picks_dict,
            )
            st.session_state.assassins = assassins

            # Auto-select captain as highest-PPG squad player if not manually chosen
            stored_cap = st.session_state.captain_id
            if stored_cap not in t_picks:
                cap_candidates = pdf[pdf["id"].isin(t_picks)]
                stored_cap = int(cap_candidates.loc[cap_candidates["ppg"].idxmax(), "id"]) if not cap_candidates.empty else None
                st.session_state.captain_id = stored_cap

            target_profile = sim.ManagerProfile(
                manager_id=target_id,
                name=t_name,
                current_points=t_pts,
                squad_player_ids=t_picks,
                captain_id=stored_cap,
            )

            # DGW/BGW player sets come from gw_info (already fetched above)
            dgw_teams = gw_info.get("dgw_teams", set())
            bgw_teams = gw_info.get("bgw_teams", set())
            dgw_pids  = set(pdf[pdf["team"].isin(dgw_teams)]["id"].tolist())
            bgw_pids  = set(pdf[pdf["team"].isin(bgw_teams)]["id"].tolist())

            cfg = sim.SimulationConfig(
                n_iterations=n_iter,
                remaining_gws=remaining,
                dgw_player_ids=dgw_pids,
                bgw_player_ids=bgw_pids,
            )
            result = sim.simulate_season(target_profile, leader_profile, pdf, cfg)
            st.session_state.sim_results     = result
            st.session_state.target_profile  = target_profile
            st.session_state.leader_profile  = leader_profile
            st.session_state.sim_cfg         = cfg

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            url    = e.response.url         if e.response is not None else ""
            if status == 404:
                if f"/entry/{target_id}" in url:
                    st.error(
                        f"Manager ID **{target_id}** was not found on FPL. "
                        f"Check your ID at fantasy.premierleague.com/entry/{target_id}/history."
                    )
                elif f"/leagues-classic/{league_id}" in url:
                    st.error(
                        f"League ID **{league_id}** was not found. "
                        "Verify the ID in the mini-league URL on the FPL website."
                    )
                else:
                    st.error("An FPL resource returned 404. Check your Manager ID and League ID.")
            elif status == 429:
                st.error("FPL API rate limit reached. Wait 60 seconds then try again.")
            else:
                st.error(f"FPL API returned HTTP {status}. Try again in a moment.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Presser analysis (independent of main run)
if run_presser and presser_quote and presser_player:
    with st.spinner("Analyzing press conference..."):
        try:
            pdf      = st.session_state.players_df
            gw_info  = st.session_state.gw_info or fpl_api.get_gw_info(gameweek)
            picks    = fpl_api.get_manager_picks(target_id, gameweek) if pdf is not None else []
            res      = nlp_utils.interpret_presser(
                quote=presser_quote,
                manager_id=target_id,
                player_name=presser_player,
                squad_player_ids=picks,
                player_id=int(presser_pid) if presser_pid else None,
                players_df=pdf,
                gw_context=gw_info,
                api_key=anthropic_key or None,
            )
            st.session_state.pressers = res
        except Exception as e:
            st.error(f"Presser error: {e}")

# ---------------------------------------------------------------------------
# Render panels
# ---------------------------------------------------------------------------
with col_gauge:
    st.markdown("### Win Probability")
    res = st.session_state.sim_results
    if res:
        st.plotly_chart(draw_gauge(res["win_probability"], res["target_name"], res["leader_name"]),
                        use_container_width=True)
        st.plotly_chart(draw_sparkline(res), use_container_width=True)
        if res.get("dgw_active"):
            st.markdown(
                "<div style='background:#ff980022;border:1px solid #ff9800;"
                "border-radius:8px;padding:6px 12px;margin-bottom:8px;"
                "color:#ff9800;font-size:13px'>⚡ <b>Double GW</b> — "
                "players with 2 fixtures get two independent score draws</div>",
                unsafe_allow_html=True,
            )
        if res.get("bgw_active"):
            st.markdown(
                "<div style='background:#ff174422;border:1px solid #ff1744;"
                "border-radius:8px;padding:6px 12px;margin-bottom:8px;"
                "color:#ff6e6e;font-size:13px'>❌ <b>Blank GW</b> — "
                "blanking players contribute 0 to the simulation</div>",
                unsafe_allow_html=True,
            )
        cap_name = res.get("target_captain", "—")
        cap_ppg  = res.get("target_captain_ppg", 0)
        win_se   = res.get("win_prob_se", 0)
        st.markdown(
            f"<div style='background:#1a1a2e;border:1px solid #ffd60033;"
            f"border-radius:10px;padding:8px 14px;margin-bottom:8px;font-size:13px'>"
            f"<span style='color:#ffd600'>★ Captain: <b>{cap_name}</b></span> "
            f"&nbsp;·&nbsp; PPG {cap_ppg:.2f} "
            f"&nbsp;·&nbsp; Win Prob SE ±{win_se:.2f}%"
            f"</div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='metric-card'>"
            f"<b>Current Gap</b>: {res['current_gap']} pts<br>"
            f"<b>Projected Gap</b>: {res['expected_gap']:.1f} pts<br>"
            f"<b>Simulations</b>: {res['n_iterations']:,}"
            f"</div>", unsafe_allow_html=True
        )
    else:
        st.info("Configure sidebar and click **RUN CLUTCH**.")

with col_assassin:
    st.markdown("### Assassin Differentials")
    adf = st.session_state.assassins
    if adf is not None and not adf.empty:
        st.plotly_chart(draw_assassin_table(adf), use_container_width=True)
        buy_targets = adf[~adf["target_owns"]].head(3)
        if not buy_targets.empty:
            st.markdown("**Top Transfer Targets**")
            for _, row in buy_targets.iterrows():
                st.markdown(
                    f"<div class='assassin-row'>🔴 <b>{row['web_name']}</b> "
                    f"– xGI {row['xGI']:.2f} | {row['ownership_pct']:.1f}% owned | "
                    f"{int(row['rival_ownership_count'])} rivals</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Assassin feed loads after RUN CLUTCH.")

with col_presser:
    st.markdown("### Presser Intelligence")
    pr = st.session_state.pressers
    if pr:
        if pr.get("fallback"):
            st.markdown(
                "<div style='background:#ff174422;border:1px solid #ff1744;"
                "border-radius:8px;padding:6px 12px;margin-bottom:8px;"
                "color:#ff6e6e;font-size:13px'>⚠️ <b>Heuristic estimate</b> — "
                "AI analysis unavailable. Check your API key or connection.</div>",
                unsafe_allow_html=True,
            )
        likelihood = pr.get("start_likelihood", "?")
        action     = pr.get("transfer_action", "?")
        conf       = pr.get("confidence", "?")
        color_map  = {"Hold": "#00e676", "Buy": "#00e5ff", "Sell": "#ff1744", "Monitor": "#ff9800"}
        action_col = color_map.get(action, "#e0e0e0")

        st.markdown(
            f"<div class='metric-card'>"
            f"<h2 style='color:{action_col}'>{action}</h2>"
            f"<b>Start Likelihood:</b> {likelihood}%<br>"
            f"<b>Confidence:</b> {conf}<br>"
            f"<b>In Squad:</b> {'✅' if pr.get('in_squad') else '❌'}"
            f"</div>", unsafe_allow_html=True
        )

        st.markdown(f"\n**Impact Summary**\n\n{pr.get('impact_summary', '')}")

        signals = pr.get("key_signals", [])
        risks   = pr.get("risk_flags", [])
        if signals:
            st.markdown("**Signals:** " + " · ".join(signals))
        if risks:
            st.markdown("**Risks:** " + " · ".join(risks))

        # Likelihood meter
        fig_bar = go.Figure(go.Bar(
            x=[likelihood], y=["Start Likelihood"], orientation="h",
            marker_color="#00e5ff" if likelihood >= 70 else "#ff9800" if likelihood >= 40 else "#ff1744",
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0d0d0d", plot_bgcolor="#111122",
            font_color="#b0bec5", height=100,
            xaxis=dict(range=[0, 100], color="#555"),
            yaxis=dict(color="#555"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Paste a press conference quote in the sidebar and click **Analyze Quote**.")

# ---------------------------------------------------------------------------
# Transfer Lab — full-width what-if engine
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Transfer Lab — What-If Simulator")

pdf             = _valid_df("players_df", ("id", "ppg", "xGI", "element_type", "ownership_pct"))
target_profile  = st.session_state.target_profile
leader_profile  = st.session_state.leader_profile
sim_cfg         = st.session_state.sim_cfg

POSITION_LABELS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

if pdf is None or target_profile is None:
    st.info("Run CLUTCH first to unlock the Transfer Lab.")
else:
    squad_df = pdf[pdf["id"].isin(target_profile.squad_player_ids)].copy()
    squad_df["label"] = squad_df.apply(
        lambda r: f"{r['web_name']} ({POSITION_LABELS.get(r['element_type'], '?')}) — {r['ppg']} PPG", axis=1
    )

    with st.expander("⚙️ Advanced Settings"):
        inertia_threshold = st.slider(
            "Minimum points gain to count as a real upgrade",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            key="inertia_slider",
            help=(
                "Transfers below this threshold are labelled 'Lateral Move' — "
                "not worth burning a free transfer or taking a hit. "
                "This is a judgment call, not a calibrated constant: "
                "tune it to match how much of a PPG edge you personally require "
                "before pulling the trigger on a transfer."
            ),
        )
        st.caption(
            "⚠️ This threshold is not calibrated against the per-player simulation model. "
            "Start at 1.0 and adjust: lower it if you transfer freely, raise it if you prefer stability."
        )

    lab_col1, lab_col2, lab_col3 = st.columns([1.4, 1.4, 1.2])

    with lab_col1:
        st.markdown("**Player OUT** (from your squad)")
        out_label = st.selectbox(
            "Select player to remove",
            options=squad_df["label"].tolist(),
            label_visibility="collapsed",
            key="transfer_out",
        )
        out_row     = squad_df[squad_df["label"] == out_label].iloc[0]
        out_id      = int(out_row["id"])
        out_pos     = int(out_row["element_type"])
        out_ppg_val = float(out_row["ppg"])

        st.markdown(
            f"<div class='metric-card' style='margin-top:8px'>"
            f"<b>{out_row['web_name']}</b><br>"
            f"PPG: {out_ppg_val:.2f} &nbsp;|&nbsp; xGI: {out_row.get('xGI', 0):.2f}<br>"
            f"Pos: {POSITION_LABELS.get(out_pos, '?')} &nbsp;|&nbsp; Own: {out_row['ownership_pct']:.1f}%"
            f"</div>", unsafe_allow_html=True
        )

    with lab_col2:
        st.markdown("**Player IN** (same position)")
        non_squad_df = pdf[
            (~pdf["id"].isin(target_profile.squad_player_ids)) &
            (pdf["element_type"] == out_pos)
        ].copy().sort_values("xGI", ascending=False)

        non_squad_df["label"] = non_squad_df.apply(
            lambda r: f"{r['web_name']} ({r['short_name']}) — {r['ppg']} PPG | xGI {r['xGI']:.2f}", axis=1
        )

        in_label = st.selectbox(
            "Select player to bring in",
            options=non_squad_df["label"].tolist(),
            label_visibility="collapsed",
            key="transfer_in",
        )
        in_row     = non_squad_df[non_squad_df["label"] == in_label].iloc[0]
        in_id      = int(in_row["id"])
        in_ppg_val = float(in_row["ppg"])

        st.markdown(
            f"<div class='metric-card' style='margin-top:8px'>"
            f"<b>{in_row['web_name']}</b><br>"
            f"PPG: {in_ppg_val:.2f} &nbsp;|&nbsp; xGI: {in_row.get('xGI', 0):.2f}<br>"
            f"Pos: {POSITION_LABELS.get(out_pos, '?')} &nbsp;|&nbsp; Own: {in_row['ownership_pct']:.1f}%"
            f"</div>", unsafe_allow_html=True
        )

    with lab_col3:
        st.markdown("**Verdict**")
        run_transfer = st.button("⚡ Run What-If", use_container_width=True, key="run_transfer")
        if run_transfer:
            with st.spinner(f"Simulating {out_row['web_name']} → {in_row['web_name']}..."):
                tr = sim.simulate_transfer(
                    target=target_profile,
                    leader=leader_profile,
                    players_df=pdf,
                    player_out_id=out_id,
                    player_in_id=in_id,
                    cfg=sim_cfg,
                    inertia_threshold=inertia_threshold,
                )
                st.session_state.transfer_result = tr

        tr = st.session_state.transfer_result
        if tr:
            delta   = tr["win_prob_delta"]
            verdict = tr["verdict"]
            v_color = {
                "MAKE IT":      "#00e676",
                "WORTH IT":     "#69f0ae",
                "INCONCLUSIVE": "#78909c",
                "HOLD":         "#ff9800",
                "AVOID":        "#ff1744",
                "LATERAL MOVE": "#546e7a",
            }.get(verdict, "#e0e0e0")

            noise_floor = tr.get("noise_floor", 0)
            win_se      = tr.get("win_prob_se", 0)

            cap_note = ""
            if tr.get("captain_promoted") and tr.get("promoted_captain"):
                cap_note = (
                    f"<div style='margin-top:8px;padding:6px 8px;"
                    f"background:#ffd60022;border:1px solid #ffd600;"
                    f"border-radius:6px;font-size:11px;color:#ffd600;text-align:left'>"
                    f"★ Captain auto-promoted to <b>{tr['promoted_captain']}</b> "
                    f"(transferred out your previous captain)"
                    f"</div>"
                )

            st.markdown(
                f"<div class='metric-card'>"
                f"<h2 style='color:{v_color};margin:0'>{verdict}</h2>"
                f"<span style='font-size:2rem;color:{v_color}'>"
                f"{'▲' if delta > 0 else '▼' if delta < 0 else '►'} {abs(delta):.1f}%</span><br>"
                f"<span style='color:#b0bec5;font-size:12px'>Win Δ &nbsp;·&nbsp; "
                f"SE ±{win_se:.2f}% &nbsp;·&nbsp; noise floor {noise_floor:.2f}%</span>"
                f"{cap_note}"
                f"</div>", unsafe_allow_html=True
            )

            # ── Tabs: Win Probability  |  Raw Points (Causal Impact) ──────────
            # Auto-select Raw Points tab when win prob is near zero (bug fix)
            low_prob = tr["win_prob_after"] < 1.0
            tab_wp, tab_xp = st.tabs(["📊 Win Probability", "📈 Raw Points"])

            with tab_wp:
                if low_prob:
                    _sr  = st.session_state.sim_results or {}
                    _gap = abs(_sr.get("current_gap", 0))
                    st.warning(
                        f"Win probability is effectively 0% — the points gap "
                        f"({_gap} pts) is too large for a single transfer to bridge. "
                        f"Switch to **Raw Points** for actionable insight.",
                        icon="⚠️",
                    )
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    name="Before", x=["Win Prob"], y=[tr["win_prob_before"]],
                    marker_color="#455a64",
                    text=[f"{tr['win_prob_before']}%"], textposition="outside",
                ))
                fig_compare.add_trace(go.Bar(
                    name="After", x=["Win Prob"], y=[tr["win_prob_after"]],
                    marker_color=v_color,
                    text=[f"{tr['win_prob_after']}%"], textposition="outside",
                ))
                fig_compare.update_layout(
                    paper_bgcolor="#0d0d0d", plot_bgcolor="#111122",
                    font_color="#b0bec5", height=200, barmode="group",
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(bgcolor="#0d0d0d"),
                    yaxis=dict(range=[0, max(100, tr["win_prob_after"] + 5)], color="#555"),
                    xaxis=dict(color="#555"),
                )
                st.plotly_chart(fig_compare, use_container_width=True)

            with tab_xp:
                total_gain = tr["total_xp_gain"]
                gw_labels  = [f"GW+{i+1}" for i in range(len(tr["xp_gain_per_gw"]))]
                cumulative = []
                running    = 0.0
                for g in tr["xp_gain_per_gw"]:
                    running += g
                    cumulative.append(round(running, 2))

                gain_color = "#00e676" if total_gain >= 0 else "#ff1744"
                st.metric(
                    label="Clutch Gain (total raw xP)",
                    value=f"{total_gain:+.1f} pts",
                    delta=f"{'Upgrade' if total_gain > 0 else 'Downgrade'} — "
                          f"{'Above' if not tr['is_lateral'] else 'Below'} inertia threshold",
                )

                fig_xp = go.Figure()
                fig_xp.add_trace(go.Bar(
                    x=gw_labels, y=tr["xp_gain_per_gw"],
                    marker_color=[gain_color if v >= 0 else "#ff1744" for v in tr["xp_gain_per_gw"]],
                    name="xP gain per GW",
                ))
                fig_xp.add_trace(go.Scatter(
                    x=gw_labels, y=cumulative,
                    mode="lines+markers", name="Cumulative",
                    line=dict(color="#00e5ff", width=2, dash="dot"),
                ))
                fig_xp.add_hline(
                    y=inertia_threshold,
                    line_dash="dash", line_color="#ff9800",
                    annotation_text=f"Your threshold ({inertia_threshold:.1f} pts — adjust in Advanced Settings)",
                    annotation_position="top right",
                )
                fig_xp.update_layout(
                    paper_bgcolor="#0d0d0d", plot_bgcolor="#111122",
                    font_color="#b0bec5", height=220,
                    margin=dict(t=10, b=30, l=30, r=10),
                    legend=dict(bgcolor="#0d0d0d"),
                    xaxis=dict(color="#555"), yaxis=dict(color="#555"),
                )
                st.plotly_chart(fig_xp, use_container_width=True)

            # PPG + xGI impact strip
            ppg_d = tr["ppg_delta"]
            xgi_d = round(tr["in_xgi"] - tr["out_xgi"], 2)
            st.markdown(
                f"<div style='display:flex;gap:12px;margin-top:4px'>"
                f"<div class='metric-card' style='flex:1'>"
                f"<b>PPG shift</b><br>"
                f"<span style='color:{'#00e676' if ppg_d > 0 else '#ff1744'};font-size:1.3rem'>"
                f"{'▲' if ppg_d > 0 else '▼'} {abs(ppg_d):.2f}</span></div>"
                f"<div class='metric-card' style='flex:1'>"
                f"<b>xGI shift</b><br>"
                f"<span style='color:{'#00e676' if xgi_d > 0 else '#ff1744'};font-size:1.3rem'>"
                f"{'▲' if xgi_d > 0 else '▼'} {abs(xgi_d):.2f}</span></div>"
                f"<div class='metric-card' style='flex:1'>"
                f"<b>Gap Δ</b><br>"
                f"<span style='color:{'#00e676' if tr['gap_delta'] < 0 else '#ff1744'};font-size:1.3rem'>"
                f"{'▼' if tr['gap_delta'] < 0 else '▲'} {abs(tr['gap_delta']):.1f} pts</span></div>"
                f"</div>", unsafe_allow_html=True
            )

st.markdown("---")
st.markdown(
    "<div style='background:#1a1a2e;border:1px solid #00e5ff33;border-radius:12px;"
    "padding:14px 20px;font-size:14px'>"
    "🎬 <b>Content Creator?</b> &nbsp; Open <b>Content Creator</b> in the sidebar "
    "to generate Egyptian Arabic YouTube scripts and social captions from your CLUTCH data."
    "</div>",
    unsafe_allow_html=True,
)
