"""
CLUTCH – The Multiverse Engine
Monte Carlo simulation (10 000 iterations) for remaining GWs.
xP = Season_Avg_PPG * Fixture_Weight + Gaussian_Noise
Returns Win Probability % to catch the league leader.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ManagerProfile:
    manager_id: int
    name: str
    current_points: int
    squad_player_ids: list[int]
    avg_ppg: Optional[float] = None       # overrides auto-calc if supplied
    captain_id: Optional[int] = None      # FPL element ID of the chosen captain


@dataclass
class SimulationConfig:
    n_iterations: int = 10_000
    remaining_gws: int = 5
    noise_std: float = 8.0                    # base Gaussian σ (points per GW)
    dgw_noise_std: float = 12.0               # σ during Double Gameweeks
    dgw_player_ids: set = field(default_factory=set)  # player IDs with 2 fixtures this GW
    dgw_threshold: int = 3                    # min DGW players to trigger elevated σ
    fixture_weights: Optional[list[float]] = None     # per-GW difficulty scalar


def _dynamic_sigma(squad_ids: list[int], cfg: "SimulationConfig") -> float:
    """Bump σ to dgw_noise_std when the squad has enough DGW players."""
    if not cfg.dgw_player_ids:
        return cfg.noise_std
    dgw_count = sum(1 for pid in squad_ids if pid in cfg.dgw_player_ids)
    return cfg.dgw_noise_std if dgw_count > cfg.dgw_threshold else cfg.noise_std


def _fixture_weights(n_gws: int, weights: Optional[list[float]]) -> np.ndarray:
    if weights and len(weights) >= n_gws:
        return np.array(weights[:n_gws], dtype=float)
    # Default: neutral difficulty (1.0) with slight decay toward end of season
    return np.linspace(1.0, 0.85, n_gws)


def _squad_avg_ppg(profile: ManagerProfile, players_df: pd.DataFrame) -> float:
    if profile.avg_ppg is not None:
        return profile.avg_ppg
    mask  = players_df["id"].isin(profile.squad_player_ids)
    squad = players_df[mask]
    if squad.empty:
        return 5.0
    return float(squad["ppg"].mean())


def _captain_split(
    profile: ManagerProfile, players_df: pd.DataFrame
) -> tuple[float, float, str]:
    """
    Split squad PPG into (non_captain_total_ppg, captain_ppg, captain_name).

    non_captain_total_ppg is the SUM of all non-captain players' PPG —
    used directly as the collective outfield base, not averaged, so the
    captain can be added back separately as 2× contribution.

    If no captain_id is set, auto-selects the highest-PPG squad player.
    """
    squad = players_df[players_df["id"].isin(profile.squad_player_ids)].copy()
    if squad.empty:
        return 50.0, 7.0, "Unknown"

    cap_id = profile.captain_id
    if cap_id is None or cap_id not in squad["id"].values:
        cap_id = int(squad.loc[squad["ppg"].idxmax(), "id"])

    cap_row   = squad[squad["id"] == cap_id].iloc[0]
    non_cap   = squad[squad["id"] != cap_id]
    cap_ppg   = float(cap_row["ppg"])
    cap_name  = str(cap_row["web_name"])
    non_total = float(non_cap["ppg"].sum()) if not non_cap.empty else 0.0

    return non_total, cap_ppg, cap_name


def simulate_season(
    target: ManagerProfile,
    leader: ManagerProfile,
    players_df: pd.DataFrame,
    cfg: SimulationConfig = SimulationConfig(),
) -> dict:
    """
    Run Monte Carlo to estimate P(target catches leader) over remaining GWs.
    Returns a results dict with win_probability and score distributions.
    """
    rng = np.random.default_rng(seed=42)
    n   = cfg.n_iterations
    gws = cfg.remaining_gws
    fw  = _fixture_weights(gws, cfg.fixture_weights)

    t_sigma = _dynamic_sigma(target.squad_player_ids, cfg)
    l_sigma = _dynamic_sigma(leader.squad_player_ids, cfg)

    # ── Captain-aware path ────────────────────────────────────────────────────
    # Outfield noise  σ   → collective variance of the 14 non-captain players
    # Captain noise   2σ  → Var(2X) = 4Var(X)  →  σ(2X) = 2σ(X)
    # Captain xP contribution = 2 × (cap_ppg × fw  +  noise(2σ))
    # ─────────────────────────────────────────────────────────────────────────
    t_non_cap, t_cap_ppg, t_cap_name = _captain_split(target, players_df)
    l_non_cap, l_cap_ppg, l_cap_name = _captain_split(leader, players_df)

    t_outfield_noise = rng.normal(0, t_sigma,     size=(n, gws))
    t_captain_noise  = rng.normal(0, t_sigma * 2, size=(n, gws))
    l_outfield_noise = rng.normal(0, l_sigma,     size=(n, gws))
    l_captain_noise  = rng.normal(0, l_sigma * 2, size=(n, gws))

    # Total GW xP = outfield contribution + doubled captain contribution
    t_gw_xp = (
        (t_non_cap * fw)[None, :] + t_outfield_noise
        + (2 * t_cap_ppg * fw)[None, :] + t_captain_noise
    )
    l_gw_xp = (
        (l_non_cap * fw)[None, :] + l_outfield_noise
        + (2 * l_cap_ppg * fw)[None, :] + l_captain_noise
    )

    # Cumulative season totals
    t_final = target.current_points + t_gw_xp.sum(axis=1)
    l_final = leader.current_points + l_gw_xp.sum(axis=1)

    win_mask     = t_final >= l_final
    win_prob     = float(win_mask.mean() * 100)
    expected_gap = float((l_final - t_final).mean())
    current_gap  = leader.current_points - target.current_points

    # Per-GW base xP for sparkline (deterministic means, no noise)
    t_gw_means = ((t_non_cap + 2 * t_cap_ppg) * fw).tolist()
    l_gw_means = ((l_non_cap + 2 * l_cap_ppg) * fw).tolist()

    return {
        "win_probability":    round(win_prob, 1),
        "current_gap":        current_gap,
        "expected_gap":       round(expected_gap, 1),
        "target_final_mean":  round(float(t_final.mean()), 1),
        "leader_final_mean":  round(float(l_final.mean()), 1),
        "target_gw_xp":       [round(v, 1) for v in t_gw_means],
        "leader_gw_xp":       [round(v, 1) for v in l_gw_means],
        "target_name":        target.name,
        "leader_name":        leader.name,
        "n_iterations":       n,
        "remaining_gws":      gws,
        "target_sigma":       t_sigma,
        "leader_sigma":       l_sigma,
        "dgw_active":         t_sigma == cfg.dgw_noise_std or l_sigma == cfg.dgw_noise_std,
        "target_captain":     t_cap_name,
        "target_captain_ppg": round(t_cap_ppg, 2),
        "leader_captain":     l_cap_name,
        "leader_captain_ppg": round(l_cap_ppg, 2),
    }


def simulate_transfer(
    target: ManagerProfile,
    leader: ManagerProfile,
    players_df: pd.DataFrame,
    player_out_id: int,
    player_in_id: int,
    cfg: SimulationConfig = SimulationConfig(),
    inertia_threshold: float = 0.7,
) -> dict:
    """
    What-If engine: swap one player in the target squad and measure
    the change in win probability vs the league leader.

    Uses the same rng seed as simulate_season so noise matrices are
    identical — only the squad PPG shifts, making the delta purely
    attributable to the transfer.

    Returns a dict with before/after win probability and the delta.
    """
    # Baseline (use existing sim if already run; recompute cleanly here)
    baseline = simulate_season(target, leader, players_df, cfg)

    # Build modified squad — if captain is transferred out, promote highest-PPG replacement
    new_squad_ids = [p for p in target.squad_player_ids if p != player_out_id] + [player_in_id]
    new_captain_id = target.captain_id
    if target.captain_id == player_out_id:
        # Auto-promote highest-PPG player from new squad as interim captain
        new_cap_rows = players_df[players_df["id"].isin(new_squad_ids)]
        new_captain_id = int(new_cap_rows.loc[new_cap_rows["ppg"].idxmax(), "id"]) if not new_cap_rows.empty else None

    modified_target = ManagerProfile(
        manager_id=target.manager_id,
        name=target.name,
        current_points=target.current_points,
        squad_player_ids=new_squad_ids,
        captain_id=new_captain_id,
    )

    new_result = simulate_season(modified_target, leader, players_df, cfg)

    # Player metadata
    def _player_row(pid: int) -> pd.Series:
        rows = players_df[players_df["id"] == pid]
        return rows.iloc[0] if not rows.empty else pd.Series({"web_name": str(pid), "ppg": 0.0, "xGA": 0.0, "ownership_pct": 0.0})

    out_row = _player_row(player_out_id)
    in_row  = _player_row(player_in_id)

    prob_delta = round(new_result["win_probability"] - baseline["win_probability"], 1)
    gap_delta  = round(new_result["expected_gap"] - baseline["expected_gap"], 1)

    # xP gain: raw expected points the new player adds over remaining GWs
    fw              = _fixture_weights(cfg.remaining_gws, cfg.fixture_weights)
    ppg_diff        = float(in_row["ppg"]) - float(out_row["ppg"])
    xp_gain_per_gw  = [round(ppg_diff * float(fw[i]), 2) for i in range(len(fw))]
    total_xp_gain   = round(sum(xp_gain_per_gw), 2)
    is_lateral      = abs(total_xp_gain) < inertia_threshold

    if is_lateral:
        verdict = "LATERAL MOVE"
    elif prob_delta >= 2.0:
        verdict = "MAKE IT"
    elif prob_delta >= 0.5:
        verdict = "WORTH IT"
    elif prob_delta >= -0.5:
        verdict = "HOLD"
    else:
        verdict = "AVOID"

    return {
        "player_out":          out_row["web_name"],
        "player_in":           in_row["web_name"],
        "out_ppg":             round(float(out_row["ppg"]), 2),
        "in_ppg":              round(float(in_row["ppg"]),  2),
        "ppg_delta":           ppg_diff,
        "out_xga":             round(float(out_row.get("xGA", 0)), 2),
        "in_xga":              round(float(in_row.get("xGA", 0)),  2),
        "win_prob_before":     baseline["win_probability"],
        "win_prob_after":      new_result["win_probability"],
        "win_prob_delta":      prob_delta,
        "expected_gap_before": baseline["expected_gap"],
        "expected_gap_after":  new_result["expected_gap"],
        "gap_delta":           gap_delta,
        "xp_gain_per_gw":      xp_gain_per_gw,
        "total_xp_gain":       total_xp_gain,
        "is_lateral":          is_lateral,
        "verdict":             verdict,
    }


def run_full_simulation(
    target: ManagerProfile,
    rivals: list[ManagerProfile],
    players_df: pd.DataFrame,
    cfg: SimulationConfig = SimulationConfig(),
) -> list[dict]:
    """Run simulation against every rival and return ranked results."""
    results = []
    for rival in rivals:
        res = simulate_season(target, rival, players_df, cfg)
        results.append(res)
    results.sort(key=lambda r: r["win_probability"], reverse=True)
    return results
