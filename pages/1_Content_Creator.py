"""
CLUTCH – Content Creator Mode
Egyptian Arabic YouTube script + social carousel caption generator.
Reads simulation results from session state set by the main CLUTCH engine.
"""

import os
import streamlit as st
import nlp_utils

st.set_page_config(
    page_title="CLUTCH – Content Creator",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
  .block-container { padding: 1.5rem 2rem; }
  .metric-card {
    background: #1a1a2e; border: 1px solid #00e5ff33;
    border-radius: 12px; padding: 1rem; text-align: center;
  }
</style>
""", unsafe_allow_html=True)


def _resolve_api_key() -> str:
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.getenv("OPENROUTER_API_KEY", "")


st.markdown("# 🎬 Content Creator Mode")
st.caption("Turn your CLUTCH analysis into Egyptian Arabic YouTube scripts and social captions.")

sim_res   = st.session_state.get("sim_results")
assassins = st.session_state.get("assassins")
gw_info   = st.session_state.get("gw_info")
tr_result = st.session_state.get("transfer_result")

if sim_res is None:
    st.info(
        "No simulation data found. Run CLUTCH on the **main page** first — "
        "the script generator needs your win probability and assassin results."
    )
    st.stop()

script_api_key = _resolve_api_key()

script_col, preview_col = st.columns([1, 1.6])

with script_col:
    st.markdown("**Session data loaded**")
    has_assassins = assassins is not None and not assassins.empty
    st.markdown(
        f"<div class='metric-card' style='font-size:13px;text-align:left'>"
        f"✅ Win Probability: <b>{sim_res['win_probability']}%</b><br>"
        f"✅ Gap vs {sim_res['leader_name']}: <b>{sim_res['current_gap']} pts</b><br>"
        f"✅ Assassin candidates: <b>{'Yes' if has_assassins else 'No — run CLUTCH again'}</b><br>"
        f"✅ Transfer tip: <b>{'Yes — ' + tr_result['player_out'] + ' → ' + tr_result['player_in'] if tr_result else 'None (run Transfer Lab first)'}</b><br>"
        f"✅ GW Type: <b>{'DGW' if gw_info and gw_info.get('is_dgw') else 'BGW' if gw_info and gw_info.get('is_bgw') else 'Normal'}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    cc_tab_script, cc_tab_caption = st.tabs(["🎬 Video Script", "📱 Social Caption"])

    with cc_tab_script:
        gen_btn = st.button("Generate 60-Second Script", use_container_width=True, key="gen_script")
        if gen_btn:
            with st.spinner("CLUTCH is writing your script..."):
                script_text = nlp_utils.generate_video_script(
                    sim_result=sim_res,
                    assassins_df=assassins,
                    manager_name=sim_res.get("target_name", "Manager"),
                    gw_context=gw_info,
                    transfer_result=tr_result,
                    api_key=script_api_key or None,
                )
                st.session_state["video_script"] = script_text

    with cc_tab_caption:
        platform = st.selectbox(
            "Platform", ["instagram", "tiktok", "twitter"], key="caption_platform"
        )
        cap_btn = st.button("Generate Social Caption", use_container_width=True, key="gen_caption")
        if cap_btn:
            with st.spinner("Crafting your carousel caption..."):
                caption_text = nlp_utils.generate_social_caption(
                    sim_result=sim_res,
                    assassins_df=assassins,
                    gw_context=gw_info,
                    transfer_result=tr_result,
                    platform=platform,
                    api_key=script_api_key or None,
                )
                st.session_state["social_caption"] = caption_text

with preview_col:
    st.markdown("**Generated Script**")
    script_text = st.session_state.get("video_script")
    if script_text:
        word_count   = len(script_text.split())
        read_secs    = max(30, min(90, int(word_count / 2.5)))
        is_arabic    = any("؀" <= c <= "ۿ" for c in script_text)
        script_dir   = "rtl" if is_arabic else "ltr"
        script_align = "right" if is_arabic else "left"
        st.markdown(
            f"<div style='background:#111122;border:1px solid #00e5ff33;border-radius:12px;"
            f"padding:16px 20px;font-size:15px;line-height:1.8;"
            f"direction:{script_dir};text-align:{script_align};"
            f"font-family:Georgia,serif;color:#e8e8e8'>{script_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#555;font-size:12px;margin-top:6px'>"
            f"~{word_count} words &nbsp;·&nbsp; ~{read_secs}s read time</div>",
            unsafe_allow_html=True,
        )
        st.text_area(
            "Copy-paste version",
            value=script_text,
            height=160,
            key="script_copy_box",
            label_visibility="collapsed",
        )
    else:
        st.info("Click **Generate Script** to create your Ammiya video script.")

    caption_text = st.session_state.get("social_caption")
    if caption_text:
        st.markdown("---")
        st.markdown("**Social Caption Preview**")
        slides = [s.strip() for s in caption_text.split("---") if s.strip()]
        for i, slide in enumerate(slides, 1):
            has_arabic = any("؀" <= c <= "ۿ" for c in slide)
            direction  = "rtl" if has_arabic else "ltr"
            st.markdown(
                f"<div style='background:#1a1a2e;border:1px solid #00e5ff22;"
                f"border-radius:8px;padding:10px 14px;margin-bottom:6px;"
                f"direction:{direction};font-size:14px;line-height:1.7'>"
                f"<span style='color:#555;font-size:11px'>Slide {i}</span><br>"
                f"{slide}</div>",
                unsafe_allow_html=True,
            )
        st.text_area(
            "Copy all slides",
            value=caption_text,
            height=120,
            key="caption_copy_box",
            label_visibility="collapsed",
        )
