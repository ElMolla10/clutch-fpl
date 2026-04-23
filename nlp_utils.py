"""
CLUTCH – The Coach-Speak Interpreter
Analyzes press conference quotes via Groq API (llama-3.3-70b).
Returns start likelihood (0-100) and a personalized impact summary
cross-referenced against the manager's actual squad.
"""

import os
import json
import re
import pandas as pd
from typing import Optional
from openai import OpenAI

# OpenRouter endpoint + model slug for Llama 3.3 70B via Groq backend
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.3-70b-instruct"


def _build_gw_context_block(gw_context: Optional[dict]) -> str:
    if not gw_context:
        return ""

    gw       = gw_context.get("gw", "?")
    deadline = gw_context.get("deadline_human", "unknown")
    mins     = gw_context.get("minutes_to_deadline")
    is_dgw   = gw_context.get("is_dgw", False)
    is_bgw   = gw_context.get("is_bgw", False)

    gw_type = "DOUBLE GAMEWEEK" if is_dgw else "BLANK GAMEWEEK" if is_bgw else "Normal Gameweek"

    urgency = ""
    if mins is not None:
        if mins <= 0:
            urgency = "DEADLINE HAS PASSED — transfers are locked."
        elif mins < 60:
            urgency = f"URGENT: only {mins} minutes until deadline — act now or hold."
        elif mins < 360:
            urgency = f"{mins} minutes to deadline — limited time to decide."
        else:
            hours = mins // 60
            urgency = f"{hours} hours to deadline — time to research before committing."

    return (
        f"\nGAMEWEEK CONTEXT:\n"
        f"  GW: {gw} ({gw_type})\n"
        f"  Deadline: {deadline}\n"
        f"  Timing: {urgency}\n"
        f"  Note: {'Double GW — players with 2 fixtures have amplified importance.' if is_dgw else ''}"
        f"{'Blank GW — some players have no fixture, prioritize certainty over upside.' if is_bgw else ''}"
    )


def _build_prompt(
    quote: str,
    player_name: str,
    in_squad: bool,
    player_stats: Optional[dict] = None,
    gw_context: Optional[dict] = None,
) -> str:
    squad_context = (
        "This player IS in the manager's squad (currently owned)."
        if in_squad
        else "This player is NOT in the manager's squad."
    )
    stats_block = ""
    if player_stats:
        stats_block = (
            f"\nPlayer season stats: xG={player_stats.get('xG', 'N/A')}, "
            f"xA={player_stats.get('xA', 'N/A')}, "
            f"PPG={player_stats.get('ppg', 'N/A')}, "
            f"Global ownership={player_stats.get('ownership_pct', 'N/A')}%"
        )

    gw_block = _build_gw_context_block(gw_context)

    return f"""You are an elite Fantasy Premier League analyst interpreting a manager's press conference quote.
Your advice must account for the gameweek timing — a transfer 2 hours before deadline is a very different \
risk to one made 3 days out.{gw_block}

PRESS CONFERENCE QUOTE:
\"\"\"{quote}\"\"\"

PLAYER: {player_name}
SQUAD STATUS: {squad_context}{stats_block}

Analyze the quote and respond with a JSON object (no markdown fences, no extra text) with exactly these fields:
{{
  "start_likelihood": <integer 0-100>,
  "confidence": "<Low|Medium|High>",
  "key_signals": ["<signal1>", "<signal2>"],
  "risk_flags": ["<flag1>"],
  "impact_summary": "<2-3 sentence personalized FPL advice that factors in deadline urgency and GW type>",
  "transfer_action": "<Hold|Buy|Sell|Monitor>"
}}

Be precise. Use numbers not vague language. If the deadline is imminent and the quote is ambiguous, \
recommend Monitor over a rushed transfer."""


def _ammiya_fallback(
    manager_id: int,
    player_name: str,
    in_squad: bool,
    reason: str = "api_error",
) -> dict:
    """Egyptian Arabic heuristic fallback when the LLM call fails."""
    squad_note = "وانت شايله" if in_squad else "مش في الفريق بتاعك"
    return {
        "start_likelihood": 50,
        "confidence": "Low",
        "key_signals": ["السيستم مش قادر يحلل دلوقتي"],
        "risk_flags": [f"fallback_reason={reason}"],
        "impact_summary": (
            f"السيستم مهنج شوية، بس غالباً اللعيب ده هيلعب أساسي — {player_name} {squad_note}. "
            "متعملش قرار كبير دلوقتي وإنت مش عارف الوضع."
        ),
        "transfer_action": "Monitor",
        "manager_id": manager_id,
        "player_name": player_name,
        "in_squad": in_squad,
        "fallback": True,
    }


def interpret_presser(
    quote: str,
    manager_id: int,
    player_name: str,
    squad_player_ids: list[int],
    player_id: Optional[int] = None,
    players_df: Optional[pd.DataFrame] = None,
    gw_context: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    Parse a press conference quote and return structured FPL intelligence.

    Args:
        quote:            Raw press conference text.
        manager_id:       FPL manager ID (used for context labeling).
        player_name:      Name of the player being discussed.
        squad_player_ids: List of player IDs in the manager's current squad.
        player_id:        FPL element ID for stat lookup.
        players_df:       Full players DataFrame from fpl_api.get_players_df().
        gw_context:       Dict from fpl_api.get_gw_info() — deadline, DGW/BGW flags, urgency.
        api_key:          OpenRouter API key (falls back to OPENROUTER_API_KEY env var).

    Returns dict with start_likelihood, impact_summary, transfer_action, etc.
    """
    key      = api_key or os.getenv("OPENROUTER_API_KEY", "")
    in_squad = player_id in squad_player_ids if player_id else False

    player_stats = None
    if players_df is not None and player_id is not None:
        row = players_df[players_df["id"] == player_id]
        if not row.empty:
            player_stats = row.iloc[0][["xG", "xA", "ppg", "ownership_pct"]].to_dict()

    if not key:
        return _ammiya_fallback(manager_id, player_name, in_squad, reason="no_key")

    client = OpenAI(api_key=key, base_url=OPENROUTER_BASE)
    prompt = _build_prompt(quote, player_name, in_squad, player_stats, gw_context)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            extra_headers={"HTTP-Referer": "https://clutch-fpl.app", "X-Title": "CLUTCH FPL"},
        )
        raw = response.choices[0].message.content.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group()) if match else _ammiya_fallback(
                manager_id, player_name, in_squad, reason="parse_error"
            )

    except Exception:
        return _ammiya_fallback(manager_id, player_name, in_squad, reason="api_error")

    result["manager_id"]  = manager_id
    result["player_name"] = player_name
    result["in_squad"]    = in_squad
    return result


def generate_video_script(
    sim_result: dict,
    assassins_df,          # pd.DataFrame — top assassin candidates
    manager_name: str,
    gw_context: Optional[dict] = None,
    transfer_result: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a 60-second Egyptian Arabic YouTube script from CLUTCH data.

    Tone: energetic FPL YouTuber, Egyptian Ammiya (عامية مصرية).
    Structure: hook → win prob → assassin move → transfer tip → CTA.
    Falls back to a templated Ammiya script if the API call fails.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    win_prob_raw = sim_result.get("win_probability", 0)
    leader_name  = sim_result.get("leader_name", "المتصدر")
    gap          = sim_result.get("current_gap", "?")
    gw           = gw_context.get("gw", "?") if gw_context else "?"
    gw_type      = (
        "دبل جيم ويك" if gw_context and gw_context.get("is_dgw")
        else "بلانك جيم ويك" if gw_context and gw_context.get("is_bgw")
        else "جيم ويك عادية"
    )

    # ── Fix 1: Hope Filter ────────────────────────────────────────────────────
    # Never open a script with a demoralising probability number.
    # Under 5% → swap the stat for a motivational frame.
    if isinstance(win_prob_raw, (int, float)) and win_prob_raw < 5:
        prob_line = (
            "المهمة صعبة والفرق كبير، "
            "بس الـ Assassin Engine لسه مطلع لنا ثغرات نقدر نلعب عليها!"
        )
        prob_instruction = (
            "لا تذكر نسبة احتمالية الفوز — استخدم الجملة دي بالحرف: "
            f'"{prob_line}"'
        )
    else:
        prob_line        = f"احتمالية الفوز على {leader_name}: {win_prob_raw}%"
        prob_instruction = "اذكر احتمالية الفوز بشكل دراماتيكي مع اسم المنافس."

    # ── Fix 2: Clutch Gain Pivot ─────────────────────────────────────────────
    # Use raw xP gain instead of the probability delta (which can be -0.1%).
    transfer_line = ""
    if transfer_result:
        xp_gain   = transfer_result.get("total_xp_gain", 0)
        gws_count = len(transfer_result.get("xp_gain_per_gw", [1]))
        transfer_line = (
            f"\nالتحويلة المقترحة: اطلع {transfer_result['player_out']} "
            f"وجيب {transfer_result['player_in']}.\n"
            f"استخدم هذه الجملة بالحرف: "
            f'"التحويلة دي هتفرق معانا بـ {abs(xp_gain):.1f} نقطة زيادة '
            f'على مدار الـ {gws_count} جولات الجاية."'
        )

    # Top 3 assassins
    assassin_lines = ""
    if assassins_df is not None and not assassins_df.empty:
        top3 = assassins_df[
            ~assassins_df.get("target_owns", pd.Series(True, index=assassins_df.index))
        ].head(3)
        if top3.empty:
            top3 = assassins_df.head(3)
        for _, row in top3.iterrows():
            bgw_note = " (بلانك — ابعد عنه!)" if row.get("has_bgw") else ""
            assassin_lines += (
                f"- {row['web_name']}{bgw_note}: xGA {row.get('xGA', 0):.2f}, "
                f"ملكيته {row['ownership_pct']:.1f}%, "
                f"بس {int(row.get('rival_ownership_count', 0))} من منافسيك شايلينه\n"
            )

    # ── Fix 3: Ammiya Polish ─────────────────────────────────────────────────
    # Explicit tone rules to kill robotic output.
    prompt = f"""أنت بتكتب سكريبت ليوتيوبر FPL مصري اسمه CLUTCH — صوته حماسي وواثق زي أي يوتيوبر مصري ناجح.

البيانات:
- جيم ويك {gw} ({gw_type})
- {prob_line}
- الفارق عن المتصدر: {gap} نقطة
- أقوى لاعبين تحت الرادار:
{assassin_lines}{transfer_line}

قواعد الكتابة (لازم تتبعها):
1. ابدأ بجملة hook مباشرة — مفيش "أهلاً" أو "يا جماعة" في الأول
2. {prob_instruction}
3. اذكر اسم اللاعب الـ Assassin بثقة + رقم xGA بتاعه
4. {f'استخدم جملة الـ Clutch Gain بالحرف.' if transfer_result else 'مفيش تحويلة — ركز على الـ Assassin.'}
5. اختم بـ CTA طبيعي مش رسمي — زي كلام صاحبك مش إعلان
6. الأسلوب: جمل قصيرة. مفيش فقرات طويلة. حوار مش خطبة.
7. مفيش ترجمة حرفية للإنجليزي — قول "الـ xGA" مش "المتوقع من الأهداف والتمريرات"
8. الطول: 130-160 كلمة بالضبط

اكتب السكريبت فقط — بدون عناوين أو تعليقات."""

    if not key:
        return _ammiya_script_fallback(prob_line, assassin_lines, leader_name, gw, transfer_line)

    try:
        client   = OpenAI(api_key=key, base_url=OPENROUTER_BASE)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.82,
            max_tokens=450,
            extra_headers={"HTTP-Referer": "https://clutch-fpl.app", "X-Title": "CLUTCH FPL"},
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _ammiya_script_fallback(prob_line, assassin_lines, leader_name, gw, transfer_line)


def _ammiya_script_fallback(
    prob_line: str, assassin_lines: str, leader_name: str, gw, transfer_line: str = ""
) -> str:
    first_assassin = assassin_lines.split("\n")[0].replace("- ", "") if assassin_lines else "لاعب تحت الرادار"
    xp_sentence    = ""
    if transfer_line:
        # Pull the quoted Clutch Gain sentence if present
        import re as _re
        m = _re.search(r'"(التحويلة دي[^"]+)"', transfer_line)
        xp_sentence = f"\n\n{m.group(1)}" if m else ""
    return (
        f"الجيم ويك {gw} — والـ CLUTCH Engine شغّال.\n\n"
        f"{prob_line}\n\n"
        f"الـ Assassin اللي المحرك لقاه:\n{first_assassin}\n"
        f"ملكيته ضعيفة — ومنافسيك مش شايفينه.{xp_sentence}\n\n"
        f"سيبلي كومنت بكابتن فريقك — وأنا هقولك رأيي على طول. 🔥"
    )


def generate_social_caption(
    sim_result: dict,
    assassins_df,
    gw_context: Optional[dict] = None,
    transfer_result: Optional[dict] = None,
    platform: str = "instagram",
    api_key: Optional[str] = None,
) -> str:
    """
    Generate a TikTok/Instagram carousel caption in Egyptian Arabic + English hashtags.
    Vibe: "آية واحدة قلبت موازين حياته..." adapted for FPL.
    Falls back to a templated caption if the API fails.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY", "")

    win_prob    = sim_result.get("win_probability", "?")
    gap         = sim_result.get("current_gap", "?")
    leader_name = sim_result.get("leader_name", "المتصدر")
    gw          = gw_context.get("gw", "?") if gw_context else "?"
    is_bgw      = gw_context.get("is_bgw", False) if gw_context else False
    is_dgw      = gw_context.get("is_dgw", False) if gw_context else False
    gw_label    = "BGW" if is_bgw else "DGW" if is_dgw else "GW"

    top_assassin = ""
    xp_gain      = ""
    if assassins_df is not None and not assassins_df.empty:
        top = assassins_df[~assassins_df.get("target_owns", pd.Series(True, index=assassins_df.index))].head(1)
        if not top.empty:
            r = top.iloc[0]
            top_assassin = f"{r['web_name']} (xGA {r.get('xGA', 0):.2f}, ملكية {r['ownership_pct']:.1f}%)"

    if transfer_result:
        xp_gain = f"+{transfer_result['total_xp_gain']:.1f} نقطة خام على الـ 5 جيم ويكس الجايين"

    platform_cta = {
        "instagram": "احفظ البوست ده وشاركه مع صاحبك اللي بيخسر! 👇",
        "tiktok":    "ادوس اللايك لو الـ CLUTCH حقق معاك! ❤️🔥",
        "twitter":   "RT لو الـ data بتتكلم أكتر من الـ gut feeling 🧠",
    }.get(platform, "شاركه مع فريقك! 🔥")

    prompt = f"""أنت خبير كتابة محتوى سوشيال ميديا بالعامية المصرية.
اكتب كابشن كاروسيل (TikTok/Instagram) من 5 سلايدز قصيرة — كل سلايد جملة أو جملتين.
الأسلوب: "آية واحدة قلبت موازين حياته..." — درامي، مثير، بيخلي الناس تكمل.

البيانات:
- الجيم ويك: {gw} ({gw_label})
- احتمالية الفوز: {win_prob}%
- الفارق عن {leader_name}: {gap} نقطة
- اللاعب الـ Assassin: {top_assassin or 'مش محدد'}
- مكسب التحويلة: {xp_gain or 'مش محسوب'}

قواعد:
1. سلايد 1: hook صادم ("تحويلة واحدة..." / "الإحصاء قال..." / "ده مش رأي...")
2. سلايد 2: المشكلة (الفارق + احتمالية الفوز)
3. سلايد 3: الـ Assassin Move + الأرقام
4. سلايد 4: الـ xP gain + لماذا هي قرار وليس تخمين
5. سلايد 5: CTA + hashtags إنجليزي

الـ hashtags لازم تشمل: #FPL #CLUTCH #FPLArabic #GW{gw} #FantasyPremierLeague
اكتب السلايدز فقط، بدون ترقيم أو عناوين، افصل بينهم بـ ---"""

    if not key:
        return _social_caption_fallback(win_prob, top_assassin, gw, gw_label, platform_cta)

    try:
        client   = OpenAI(api_key=key, base_url=OPENROUTER_BASE)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=350,
            extra_headers={"HTTP-Referer": "https://clutch-fpl.app", "X-Title": "CLUTCH FPL"},
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _social_caption_fallback(win_prob, top_assassin, gw, gw_label, platform_cta)


def _social_caption_fallback(
    win_prob, top_assassin: str, gw, gw_label: str, cta: str
) -> str:
    return (
        f"تحويلة واحدة ممكن تغير كل حاجة في الـ {gw_label}{gw}... 🧠⚡\n"
        f"---\n"
        f"الـ CLUTCH Engine بيقول احتمالية الفوز دلوقتي {win_prob}% بس.\n"
        f"الوضع صعب، بس مش مستحيل — والفرق في تفصيلة واحدة.\n"
        f"---\n"
        f"الـ Assassin Move: {top_assassin or 'لاعب تحت الرادار بأرقام مجنونة'} 🎯\n"
        f"ملكيته ضعيفة — ومنافسيك مش شايفينه.\n"
        f"---\n"
        f"ده مش رأي. ده إحصاء. 10,000 سيمولاشن قالوا نعم ✅\n"
        f"One transfer. One edge. That's CLUTCH.\n"
        f"---\n"
        f"{cta}\n"
        f"#FPL #CLUTCH #FPLArabic #GW{gw} #FantasyPremierLeague #FPLStrategy"
    )


def batch_interpret(
    quotes: list[dict],
    manager_id: int,
    squad_player_ids: list[int],
    players_df: Optional[pd.DataFrame] = None,
    gw_context: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Process multiple presser quotes in sequence.
    Each item in quotes: {"quote": str, "player_name": str, "player_id": int|None}
    """
    results = []
    for item in quotes:
        res = interpret_presser(
            quote=item["quote"],
            manager_id=manager_id,
            player_name=item["player_name"],
            squad_player_ids=squad_player_ids,
            player_id=item.get("player_id"),
            players_df=players_df,
            gw_context=gw_context,
            api_key=api_key,
        )
        results.append(res)
    return results
