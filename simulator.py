"""
CLUTCH – The Multiverse Engine (v2)

Player-level Monte Carlo: each player draws independently from N(ppg_i, σ_i).
Fixes vs v1:
  - True player-level variance: Haaland and a bench-warmer with the same PPG
    get completely different distributions.
  - DGW players get two independent draws (two fixtures = two scoring events).
  - BGW players are zeroed, not noise-bumped.
  - fixture_weights default is uniform 1.0 — no undocumented decay bias.
  - Verdict thresholds are calibrated to the standard error of the win-prob
    estimate; sub-noise deltas are labelled INCONCLUSIVE.
  - simulate_transfer computes leader xP once and reuses it (not four lookups).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# Position-based per-GW points standard deviation (GK/DEF/MID/FWD).
# Derived from typical FPL score distributions: GKs are more consistent,
# forwards have higher haul/blank variance.
_POSITION_SIGMA = {1: 2.5, 2: 3.5, 3: 4.5, 4: 5.5}
_DEFAULT_SIGMA  = 4.0


@dataclass
class ManagerProfile:
    manager_id:      int
    name:            str
    current_points:  int
    squad_player_ids: list[int]
    avg_ppg:         Optional[float] = None   # unused in v2; kept for API compat
    captain_id:      Optional[int]  = None


@dataclass
class SimulationConfig:
    n_iterations:   int   = 10_000
    remaining_gws:  int   = 5
    dgw_player_ids: set   = field(default_factory=set)  # 2 draws this GW
    bgw_player_ids: set   = field(default_factory=set)  # zeroed this GW
    # fixture_weights: explicit per-GW scalars.
    # None → uniform 1.0. Provide real difficulty data if you have it.
    fixture_weights: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fixture_weights(n_gws: int, weights: Optional[list[float]]) -> np.ndarray:
    if weights and len(weights) >= n_gws:
        return np.array(weights[:n_gws], dtype=float)
    return np.ones(n_gws, dtype=float)   # uniform — no hidden bias


def _player_sigma(player: pd.Series) -> float:
    """
    Per-player σ = position base + form-deviation bonus, clamped to a ceiling.

    form deviation captures rotation risk and hot/cold streaks:
    a player whose form (5-GW avg) diverges from season PPG is less
    predictable, so their distribution is wider.

    Ceiling: sigma is capped at max(ppg * 0.8, base * 1.5) to prevent
    malformed API data (e.g. form=25, ppg=3) from producing distributions
    that regularly sample negative scores or scores above 30 in one GW.
    """
    pos  = int(player.get("element_type", 3))
    base = _POSITION_SIGMA.get(pos, _DEFAULT_SIGMA)
    ppg  = float(player.get("ppg", 0) or 0)
    try:
        form = float(str(player.get("form", ppg) or ppg))
    except (ValueError, TypeError):
        form = ppg
    form_dev = abs(form - ppg) * 0.3
    sigma    = base + form_dev
    ceiling  = max(ppg * 0.8, base * 1.5)
    return max(1.0, min(sigma, ceiling))


def _select_starters(squad: pd.DataFrame, n: int = 11) -> pd.DataFrame:
    """
    Two-pass FPL-legal starting XI selection.

    Pass 1 — positional floors (FPL minimum formation requirements):
      1 GK, 3 DEF, 2 MID, 1 FWD — best by PPG within each position.
    Pass 2 — fill the remaining slots (up to n=11) greedily by PPG
      from whoever was not selected in pass 1, regardless of position.

    This prevents a PPG-sorted top-11 from picking, e.g., 5 MIDs and
    0 GKs when a squad is heavily weighted toward premium midfielders.
    """
    # element_type: 1=GK, 2=DEF, 3=MID, 4=FWD
    FLOORS = {1: 1, 2: 3, 3: 2, 4: 1}

    selected_idx: list[int] = []
    used: set[int] = set()

    for pos, count in FLOORS.items():
        pool = squad[squad["element_type"] == pos].sort_values("ppg", ascending=False)
        picks = pool.head(count)
        selected_idx.extend(picks.index.tolist())
        used.update(picks.index.tolist())

    # Pass 2: fill remaining slots from the rest of the squad by PPG
    remaining = squad[~squad.index.isin(used)].sort_values("ppg", ascending=False)
    flex_slots = n - len(selected_idx)
    if flex_slots > 0:
        selected_idx.extend(remaining.head(flex_slots).index.tolist())

    return squad.loc[selected_idx]


def _resolve_captain(starters: pd.DataFrame, captain_id: Optional[int]) -> tuple[int, str, float]:
    """Return (cap_id, cap_name, cap_ppg), auto-selecting highest PPG if unset."""
    if captain_id is not None and captain_id in starters["id"].values:
        row = starters[starters["id"] == captain_id].iloc[0]
    else:
        row = starters.loc[starters["ppg"].idxmax()]
    return int(row["id"]), str(row["web_name"]), float(row["ppg"])


def _simulate_squad_xp(
    squad_ids:  list[int],
    captain_id: Optional[int],
    players_df: pd.DataFrame,
    cfg:        SimulationConfig,
    rng:        np.random.Generator,
    n_iter:     int,
    fw:         np.ndarray,
) -> tuple[np.ndarray, str, float]:
    """
    Player-level Monte Carlo for one manager.

    For each starter:
      - Normal GW:  draws ~ N(ppg_i, σ_i), scaled by fixture weight.
      - BGW player: contribution = 0 (no fixture, no points).
      - DGW player: two independent draws summed (two scoring events).
      - Captain:    their draw is added a second time (2x multiplier).

    Returns:
      xp_matrix  shape (n_iter, n_gws)
      cap_name   str
      cap_ppg    float
    """
    squad    = players_df[players_df["id"].isin(squad_ids)].copy()
    n_gws    = len(fw)

    if squad.empty:
        return np.zeros((n_iter, n_gws)), "Unknown", 0.0

    starters              = _select_starters(squad)
    cap_id, cap_name, cap_ppg = _resolve_captain(starters, captain_id)
    xp_mat = np.zeros((n_iter, n_gws))

    for _, player in starters.iterrows():
        pid     = int(player["id"])
        ppg     = float(player.get("ppg", 0) or 0)
        sigma_i = _player_sigma(player)
        is_bgw  = bool(cfg.bgw_player_ids and pid in cfg.bgw_player_ids)
        is_dgw  = bool(cfg.dgw_player_ids and pid in cfg.dgw_player_ids)
        is_cap  = pid == cap_id

        if is_bgw:
            continue   # genuine zero — no fixture this week

        draws = rng.normal(ppg, sigma_i, size=(n_iter, n_gws)) * fw[None, :]

        if is_dgw:
            # Second independent fixture: fresh draw from same distribution
            draws += rng.normal(ppg, sigma_i, size=(n_iter, n_gws)) * fw[None, :]

        xp_mat += draws

        if is_cap:
            xp_mat += draws    # captain's points counted a second time

    return xp_mat, cap_name, cap_ppg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_season(
    target:     ManagerProfile,
    leader:     ManagerProfile,
    players_df: pd.DataFrame,
    cfg:        SimulationConfig = SimulationConfig(),
) -> dict:
    """
    True player-level Monte Carlo.
    Uses seed=42 for target squad, seed=43 for leader squad so the
    two distributions are independent but deterministic.
    """
    n  = cfg.n_iterations
    gws = cfg.remaining_gws
    fw  = _fixture_weights(gws, cfg.fixture_weights)

    t_xp, t_cap_name, t_cap_ppg = _simulate_squad_xp(
        target.squad_player_ids, target.captain_id, players_df, cfg,
        np.random.default_rng(seed=42), n, fw,
    )
    l_xp, l_cap_name, l_cap_ppg = _simulate_squad_xp(
        leader.squad_player_ids, leader.captain_id, players_df, cfg,
        np.random.default_rng(seed=43), n, fw,
    )

    t_final = target.current_points + t_xp.sum(axis=1)
    l_final = leader.current_points + l_xp.sum(axis=1)

    win_prob     = float((t_final >= l_final).mean() * 100)
    expected_gap = float((l_final - t_final).mean())
    current_gap  = leader.current_points - target.current_points

    # SE of the win-probability estimate: SE = sqrt(p(1-p)/n)
    p            = win_prob / 100
    se_win_prob  = float(np.sqrt(p * (1 - p) / n) * 100)

    return {
        "win_probability":    round(win_prob, 1),
        "win_prob_se":        round(se_win_prob, 2),
        "current_gap":        current_gap,
        "expected_gap":       round(expected_gap, 1),
        "target_final_mean":  round(float(t_final.mean()), 1),
        "leader_final_mean":  round(float(l_final.mean()), 1),
        "target_gw_xp":       [round(v, 1) for v in t_xp.mean(axis=0).tolist()],
        "leader_gw_xp":       [round(v, 1) for v in l_xp.mean(axis=0).tolist()],
        "target_name":        target.name,
        "leader_name":        leader.name,
        "n_iterations":       n,
        "remaining_gws":      gws,
        "dgw_active":         bool(cfg.dgw_player_ids),
        "bgw_active":         bool(cfg.bgw_player_ids),
        "target_captain":     t_cap_name,
        "target_captain_ppg": round(t_cap_ppg, 2),
        "leader_captain":     l_cap_name,
        "leader_captain_ppg": round(l_cap_ppg, 2),
    }


def simulate_transfer(
    target:             ManagerProfile,
    leader:             ManagerProfile,
    players_df:         pd.DataFrame,
    player_out_id:      int,
    player_in_id:       int,
    cfg:                SimulationConfig = SimulationConfig(),
    inertia_threshold:  float = 0.7,
) -> dict:
    """
    What-If transfer engine.

    Efficiency: leader xP is computed ONCE with seed=43 and reused for
    both baseline and modified comparisons. Target draws use seed=42
    in both cases so the only variable is squad composition.

    Verdict calibration: a delta is only labelled WORTH IT or better if
    it exceeds 2× the SE of the win-probability estimate — i.e. the shift
    is outside the statistical noise of the simulation.
    """
    n   = cfg.n_iterations
    fw  = _fixture_weights(cfg.remaining_gws, cfg.fixture_weights)

    # Leader xP — computed once, reused for both comparisons
    l_xp, l_cap_name, l_cap_ppg = _simulate_squad_xp(
        leader.squad_player_ids, leader.captain_id, players_df, cfg,
        np.random.default_rng(seed=43), n, fw,
    )
    l_final = leader.current_points + l_xp.sum(axis=1)

    def _run_target(profile: ManagerProfile) -> np.ndarray:
        xp, _, _ = _simulate_squad_xp(
            profile.squad_player_ids, profile.captain_id, players_df, cfg,
            np.random.default_rng(seed=42), n, fw,
        )
        return profile.current_points + xp.sum(axis=1)

    # Baseline
    t_base_final   = _run_target(target)
    base_win_prob  = float((t_base_final >= l_final).mean() * 100)

    # Modified squad
    new_squad_ids     = [p for p in target.squad_player_ids if p != player_out_id] + [player_in_id]
    new_cap_id        = target.captain_id
    captain_promoted  = False
    promoted_cap_name = None
    if target.captain_id == player_out_id:
        cands = players_df[players_df["id"].isin(new_squad_ids)]
        if not cands.empty:
            best         = cands.loc[cands["ppg"].idxmax()]
            new_cap_id   = int(best["id"])
            promoted_cap_name = str(best["web_name"])
        else:
            new_cap_id = None
        captain_promoted = True

    modified = ManagerProfile(
        manager_id=target.manager_id, name=target.name,
        current_points=target.current_points,
        squad_player_ids=new_squad_ids, captain_id=new_cap_id,
    )
    t_mod_final  = _run_target(modified)
    new_win_prob = float((t_mod_final >= l_final).mean() * 100)

    prob_delta = round(new_win_prob - base_win_prob, 1)
    gap_delta  = round(
        float((l_final - t_mod_final).mean()) - float((l_final - t_base_final).mean()), 1
    )

    # xP gain over remaining GWs (raw points, position-independent)
    xp_gain_per_gw = [round(
        (float(players_df[players_df["id"] == player_in_id]["ppg"].iloc[0])
         - float(players_df[players_df["id"] == player_out_id]["ppg"].iloc[0]))
        * float(fw[i]), 2
    ) for i in range(len(fw))]
    total_xp_gain = round(sum(xp_gain_per_gw), 2)
    is_lateral    = abs(total_xp_gain) < inertia_threshold

    # SE-calibrated verdict: refuse to call WORTH IT on a sub-noise delta
    p_new = new_win_prob / 100
    se    = float(np.sqrt(p_new * (1 - p_new) / n) * 100)
    noise_floor = 2 * se   # ~2σ significance threshold

    if is_lateral:
        verdict = "LATERAL MOVE"
    elif abs(prob_delta) <= noise_floor:
        verdict = "INCONCLUSIVE"
    elif prob_delta >= 2.0:
        verdict = "MAKE IT"
    elif prob_delta > noise_floor:
        verdict = "WORTH IT"
    elif prob_delta <= -2.0:
        verdict = "AVOID"
    else:
        verdict = "HOLD"

    def _prow(pid):
        rows = players_df[players_df["id"] == pid]
        return rows.iloc[0] if not rows.empty else pd.Series(
            {"web_name": str(pid), "ppg": 0.0, "xGI": 0.0, "ownership_pct": 0.0}
        )

    out_row = _prow(player_out_id)
    in_row  = _prow(player_in_id)

    return {
        "player_out":          str(out_row["web_name"]),
        "player_in":           str(in_row["web_name"]),
        "out_ppg":             round(float(out_row["ppg"]), 2),
        "in_ppg":              round(float(in_row["ppg"]),  2),
        "ppg_delta":           round(float(in_row["ppg"]) - float(out_row["ppg"]), 2),
        "out_xgi":             round(float(out_row.get("xGI", 0)), 2),
        "in_xgi":              round(float(in_row.get("xGI", 0)),  2),
        "win_prob_before":     round(base_win_prob, 1),
        "win_prob_after":      round(new_win_prob, 1),
        "win_prob_delta":      prob_delta,
        "win_prob_se":         round(se, 2),
        "noise_floor":         round(noise_floor, 2),
        "expected_gap_before": round(float((l_final - t_base_final).mean()), 1),
        "expected_gap_after":  round(float((l_final - t_mod_final).mean()),  1),
        "gap_delta":           gap_delta,
        "xp_gain_per_gw":      xp_gain_per_gw,
        "total_xp_gain":       total_xp_gain,
        "is_lateral":          is_lateral,
        "verdict":             verdict,
        "captain_promoted":    captain_promoted,
        "promoted_captain":    promoted_cap_name,
    }


def run_full_simulation(
    target:     ManagerProfile,
    rivals:     list[ManagerProfile],
    players_df: pd.DataFrame,
    cfg:        SimulationConfig = SimulationConfig(),
) -> list[dict]:
    results = [simulate_season(target, rival, players_df, cfg) for rival in rivals]
    return sorted(results, key=lambda r: r["win_probability"], reverse=True)
