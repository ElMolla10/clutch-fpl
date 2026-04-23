"""
CLUTCH – The Overtake Handler
Fetches FPL bootstrap data, manager squad, and mini-league standings.

Fixes vs v1:
  - _get() uses a persistent Session with urllib3 Retry (3 attempts,
    exponential backoff, retries on 429/5xx). No more crash on transient 503.
  - compute_ownership_gap accepts bootstrap/players_df from the caller
    instead of re-fetching bootstrap on every run.
  - xG+xA column renamed xGI (expected Goal Involvement). xGA is a
    defensive metric (expected Goals Against) — wrong name for an
    attacking proxy.
  - get_players_df fills missing expected_goals/expected_assists with 0
    and warns, instead of silently substituting ppg.
"""

import logging
import warnings

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

BASE    = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "CLUTCH-FPL-Engine/1.0"}


# ---------------------------------------------------------------------------
# HTTP layer — persistent session with exponential backoff
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    retry   = Retry(
        total=3,
        backoff_factor=1,                          # sleeps: 1s, 2s, 4s
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,                     # let raise_for_status handle it
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


_SESSION = _make_session()


def _get(url: str) -> dict:
    r = _SESSION.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def get_bootstrap() -> dict:
    return _get(f"{BASE}/bootstrap-static/")


def get_gw_info(gw: int, bootstrap: Optional[dict] = None) -> dict:
    """
    Returns timing and type context for a given gameweek.
    Accepts an already-fetched bootstrap dict to avoid a redundant API call.
    """
    bs     = bootstrap or get_bootstrap()
    events = {e["id"]: e for e in bs.get("events", [])}
    event  = events.get(gw, {})

    deadline_iso   = event.get("deadline_time", "")
    deadline_human = ""
    minutes_left   = None

    if deadline_iso:
        dt             = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
        deadline_human = dt.strftime("%a %d %b, %H:%M UTC")
        delta          = dt - datetime.now(timezone.utc)
        minutes_left   = max(0, int(delta.total_seconds() // 60))

    try:
        fixtures    = _get(f"{BASE}/fixtures/?event={gw}")
        team_counts: dict[int, int] = {}
        for f in fixtures:
            for t in (f["team_h"], f["team_a"]):
                team_counts[t] = team_counts.get(t, 0) + 1
        all_teams = {t["id"] for t in bs.get("teams", [])}
        dgw_teams = {t for t, c in team_counts.items() if c > 1}
        bgw_teams = all_teams - set(team_counts.keys())
        is_dgw    = len(dgw_teams) > 0
        is_bgw    = len(bgw_teams) > 0
    except Exception:
        is_dgw = is_bgw = False
        dgw_teams = bgw_teams = set()

    return {
        "gw":                  gw,
        "deadline_iso":        deadline_iso,
        "deadline_human":      deadline_human,
        "minutes_to_deadline": minutes_left,
        "is_dgw":              is_dgw,
        "is_bgw":              is_bgw,
        "dgw_teams":           dgw_teams,
        "bgw_teams":           bgw_teams,
        "finished":            event.get("finished", False),
    }


def get_players_df(bootstrap: Optional[dict] = None) -> pd.DataFrame:
    """
    Build the master players DataFrame from bootstrap data.

    xGI (expected Goal Involvement) = xG + xA — an attacking proxy for
    FPL differential analysis. Not to be confused with xGA (expected
    Goals Against), which is a defensive metric.

    Missing stat fields are filled with 0 and a warning is emitted,
    rather than silently substituting an unrelated column (ppg).
    """
    bs      = bootstrap or get_bootstrap()
    players = pd.DataFrame(bs["elements"])
    teams   = pd.DataFrame(bs["teams"])[["id", "name", "short_name"]]
    players = players.merge(teams, left_on="team", right_on="id", suffixes=("", "_team"))

    players["ownership_pct"] = players["selected_by_percent"].astype(float)
    players["ppg"]           = players["points_per_game"].astype(float)

    # xG — warn and fill 0 if column absent (API may not expose it yet)
    if "expected_goals" in players.columns:
        players["xG"] = pd.to_numeric(players["expected_goals"], errors="coerce").fillna(0)
    else:
        warnings.warn(
            "FPL API did not return 'expected_goals'; xG set to 0 for all players.",
            RuntimeWarning, stacklevel=2,
        )
        players["xG"] = 0.0

    # xA — same treatment
    if "expected_assists" in players.columns:
        players["xA"] = pd.to_numeric(players["expected_assists"], errors="coerce").fillna(0)
    else:
        warnings.warn(
            "FPL API did not return 'expected_assists'; xA set to 0 for all players.",
            RuntimeWarning, stacklevel=2,
        )
        players["xA"] = 0.0

    # xGI = expected Goal Involvement (xG + xA) — attacking FPL proxy
    players["xGI"] = players["xG"] + players["xA"]

    # form as float for simulator sigma calculation
    players["form"] = pd.to_numeric(players.get("form", 0), errors="coerce").fillna(0)

    return players


# FDR → score multiplier. Calibrated to average FPL score distributions
# by difficulty tier; FDR 3 is neutral (1.00).
_FDR_MULTIPLIER: dict[int, float] = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.88, 5: 0.75}


def get_upcoming_opponents(
    team_id: int,
    from_gw: int,
    n_gws: int,
    bootstrap: Optional[dict] = None,
) -> tuple[list[int], list[int]]:
    """
    Return (gw_numbers, opponent_team_ids) for the next n_gws gameweeks.

    opponent_team_id is -1 when the team has no fixture in that GW (blank).
    DGW: only the first opponent is returned — caller can detect DGW separately.
    Any fetch failure returns -1 (neutral fallback) for that GW.
    """
    bs = bootstrap or get_bootstrap()
    gw_numbers:    list[int] = []
    opponent_ids:  list[int] = []

    for gw in range(from_gw, from_gw + n_gws):
        gw_numbers.append(gw)
        try:
            fixtures = _get(f"{BASE}/fixtures/?event={gw}")
            opp = -1
            for f in fixtures:
                h, a = f.get("team_h"), f.get("team_a")
                if h == team_id and a is not None:
                    opp = int(a)
                    break
                if a == team_id and h is not None:
                    opp = int(h)
                    break
            opponent_ids.append(opp)
        except Exception:
            opponent_ids.append(-1)

    return gw_numbers, opponent_ids


def get_team_fixture_weights(
    gw: int,
    n_gws: int,
    bootstrap: Optional[dict] = None,
    elo_ratings: Optional[dict] = None,
) -> dict[int, list[float]]:
    """
    Per-team fixture difficulty multipliers for the next n_gws gameweeks.

    Returns {team_id: [fw_gw0, fw_gw1, ..., fw_gw(n-1)]}.

    When elo_ratings is provided (a {str(team_id): float} dict from elo.py),
    weights use the Elo win-probability formula: 0.75 + win_prob * 0.45.
    When elo_ratings is None or empty, falls back to FDR-based multipliers.

    - No fixture in a GW → 0.0 (blank week).
    - DGW teams (two fixtures) → mean of their two opponent weights.
    - Any fetch failure → 1.0 (neutral) for that GW.
    """
    bs       = bootstrap or get_bootstrap()
    team_ids = {t["id"] for t in bs.get("teams", [])}
    weights  = {tid: [1.0] * n_gws for tid in team_ids}

    use_elo = bool(elo_ratings)

    if use_elo:
        # Lazy import avoids circular dependency (elo.py imports fpl_api)
        from elo import get_fixture_weight as _elo_weight

    for i, future_gw in enumerate(range(gw, gw + n_gws)):
        try:
            fixtures   = _get(f"{BASE}/fixtures/?event={future_gw}")
            gw_mults: dict[int, list[float]] = {}
            for f in fixtures:
                h, a = f.get("team_h"), f.get("team_a")
                if use_elo and h is not None and a is not None:
                    h_m = _elo_weight(int(h), int(a), elo_ratings)
                    a_m = _elo_weight(int(a), int(h), elo_ratings)
                else:
                    h_m = _FDR_MULTIPLIER.get(f.get("team_h_difficulty", 3), 1.0)
                    a_m = _FDR_MULTIPLIER.get(f.get("team_a_difficulty", 3), 1.0)
                if h is not None:
                    gw_mults.setdefault(h, []).append(h_m)
                if a is not None:
                    gw_mults.setdefault(a, []).append(a_m)
            for tid in team_ids:
                mults           = gw_mults.get(tid)
                weights[tid][i] = (sum(mults) / len(mults)) if mults else 0.0
        except Exception:
            pass  # leave 1.0 (neutral) on any fetch failure

    return weights


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

def get_manager_info(manager_id: int) -> dict:
    return _get(f"{BASE}/entry/{manager_id}/")


def get_manager_picks(manager_id: int, gameweek: int) -> list[int]:
    data = _get(f"{BASE}/entry/{manager_id}/event/{gameweek}/picks/")
    return [p["element"] for p in data["picks"]]


# ---------------------------------------------------------------------------
# Mini-League
# ---------------------------------------------------------------------------

def get_mini_league_standings(league_id: int, page: int = 1) -> dict:
    return _get(f"{BASE}/leagues-classic/{league_id}/standings/?page_standings={page}")


def get_top_n_managers(league_id: int, n: int = 5) -> list[dict]:
    standings = get_mini_league_standings(league_id)
    return standings["standings"]["results"][:n]


def get_rival_player_ids(
    rival_manager_ids: list[int], gameweek: int
) -> dict[int, list[int]]:
    return {mid: get_manager_picks(mid, gameweek) for mid in rival_manager_ids}


# ---------------------------------------------------------------------------
# Ownership Gap (Assassin Feed)
# ---------------------------------------------------------------------------

def compute_ownership_gap(
    target_id: int,
    league_id: int,
    gameweek: int,
    players_df: Optional[pd.DataFrame] = None,
    bootstrap: Optional[dict] = None,
    low_ownership_threshold: float = 15.0,
    xgi_threshold: float = 0.3,
    bgw_teams: Optional[set] = None,
    top5_standings: Optional[list] = None,
    target_picks: Optional[list] = None,
    rival_picks: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Returns Assassin differential candidates: players with low global
    ownership and high xGI (xG+xA) that rivals are not holding.

    Pass top5_standings, target_picks, and rival_picks from the caller to
    avoid redundant network requests — all three are already fetched by the
    main run block and can be forwarded here at zero cost.
    """
    bs = bootstrap or get_bootstrap()
    df = players_df if players_df is not None else get_players_df(bs)

    top5        = top5_standings or get_top_n_managers(league_id, n=5)
    rival_ids   = [m["entry"] for m in top5 if m["entry"] != target_id]
    target_owns = set(target_picks) if target_picks is not None else set(get_manager_picks(target_id, gameweek))
    rival_owns  = rival_picks if rival_picks is not None else get_rival_player_ids(rival_ids, gameweek)

    rival_counts: dict[int, int] = {}
    for picks in rival_owns.values():
        for pid in picks:
            rival_counts[pid] = rival_counts.get(pid, 0) + 1

    assassins = df[
        (df["ownership_pct"] < low_ownership_threshold)
        & (df["xGI"] >= xgi_threshold)
    ].copy()

    assassins["rival_ownership_count"] = (
        assassins["id"].map(rival_counts).fillna(0).astype(int)
    )
    assassins["target_owns"] = assassins["id"].isin(target_owns)
    assassins["has_bgw"]     = (
        assassins["team"].isin(bgw_teams) if bgw_teams else False
    )

    assassins = assassins.sort_values("xGI", ascending=False)

    cols = [
        "id", "web_name", "short_name", "element_type",
        "ownership_pct", "xG", "xA", "xGI",
        "rival_ownership_count", "target_owns", "ppg", "has_bgw", "team",
    ]
    return assassins[[c for c in cols if c in assassins.columns]].reset_index(drop=True)
