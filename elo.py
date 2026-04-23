"""
CLUTCH – Elo Ratings System
Builds team Elo ratings by replaying every completed fixture from GW1.
Persists to elo_ratings.json; subsequent runs only process new GWs (delta update).

Importable outside Streamlit — no st.cache_data, no Streamlit dependencies.
"""

import json
import logging
import os

log = logging.getLogger(__name__)

ELO_INITIAL = 1500.0
ELO_K       = 20           # standard K for a 38-game season
ELO_PATH    = "elo_ratings.json"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_ratings(path: str = ELO_PATH) -> dict:
    """
    Load ratings from disk.

    Returns dict with keys:
      "updated_gw": int  — last GW fully processed (0 if fresh)
      "season":     str  — e.g. "2024-25"
      "ratings":    dict — {str(team_id): float}

    Returns a fresh structure if the file is missing or corrupt.
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as exc:
            log.warning("Elo: could not read %s (%s) — starting fresh.", path, exc)
    return {"updated_gw": 0, "season": "", "ratings": {}}


def save_ratings(data: dict, path: str = ELO_PATH) -> None:
    """Atomically write ratings dict to JSON (tmp-file + os.replace)."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        log.error("Elo: failed to save ratings to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Core Elo update
# ---------------------------------------------------------------------------

def _apply_elo_update(
    ratings: dict,
    home_id: int,
    away_id: int,
    home_goals: int,
    away_goals: int,
) -> None:
    """
    Mutate ratings in-place for one completed fixture.
    Teams absent from ratings are initialised at 1500 before the update.
    K = 20 (intentional — simple and auditable for a 38-game season).
    """
    h_key = str(home_id)
    a_key = str(away_id)

    home_elo = ratings.get(h_key, ELO_INITIAL)
    away_elo = ratings.get(a_key, ELO_INITIAL)

    expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))
    expected_away = 1.0 - expected_home

    if home_goals > away_goals:
        outcome_home = 1.0
    elif home_goals == away_goals:
        outcome_home = 0.5
    else:
        outcome_home = 0.0

    ratings[h_key] = home_elo + ELO_K * (outcome_home - expected_home)
    ratings[a_key] = away_elo + ELO_K * ((1.0 - outcome_home) - expected_away)


# ---------------------------------------------------------------------------
# Season build / delta update
# ---------------------------------------------------------------------------

def build_season_ratings(
    bootstrap: dict,
    path: str = ELO_PATH,
    force_rebuild: bool = False,
) -> dict:
    """
    Main entry point. Builds or incrementally updates Elo ratings.

    Behaviour:
      1. Load existing ratings from disk (or start fresh on missing/force_rebuild).
      2. Detect season change → reset to GW0 if the season label differs.
      3. If already current (updated_gw == last finished GW) → return immediately,
         zero API calls made.
      4. For each unprocessed GW in order: fetch fixtures, apply updates.
         A single GW fetch failure logs a warning and skips that GW.
      5. Save the full ratings dict once after all GWs are processed.
      6. Return {str(team_id): float}.

    Parameters
    ----------
    bootstrap   : already-fetched FPL bootstrap dict (avoids extra network call)
    path        : path to the JSON persistence file
    force_rebuild: delete existing file and replay from GW1
    """
    # Lazy import keeps elo.py importable even without fpl_api on the path,
    # and avoids a module-level circular dependency.
    from fpl_api import _get as _fpl_get, BASE as _FPL_BASE

    events = bootstrap.get("events", [])
    finished_gws = sorted(e["id"] for e in events if e.get("finished", False))

    if not finished_gws:
        log.warning("Elo: no finished GWs found in bootstrap — returning empty.")
        return {}

    last_finished_gw = finished_gws[-1]

    # Derive season label from GW1 deadline (used to detect season rollover)
    season = ""
    try:
        gw1 = next((e for e in events if e["id"] == 1), None)
        if gw1:
            dl     = gw1.get("deadline_time", "")
            year   = int(dl[:4])
            season = f"{year}-{str(year + 1)[2:]}"
    except Exception:
        pass

    # Load or wipe
    if force_rebuild and os.path.exists(path):
        os.remove(path)

    data = load_ratings(path)

    # Season rollover detection
    if data.get("season") and season and data["season"] != season:
        log.info("Elo: new season (%s → %s) — resetting ratings.", data["season"], season)
        data = {"updated_gw": 0, "season": season, "ratings": {}}

    from_gw = data["updated_gw"] + 1

    if from_gw > last_finished_gw:
        # Already up to date — no network calls
        return data["ratings"]

    ratings = dict(data["ratings"])  # copy; do not mutate cached load result

    # Fresh start: seed all teams at 1500
    if not ratings:
        for team in bootstrap.get("teams", []):
            ratings[str(team["id"])] = ELO_INITIAL

    # Process each unprocessed GW in chronological order
    last_processed = data["updated_gw"]
    for gw in range(from_gw, last_finished_gw + 1):
        try:
            fixtures = _fpl_get(f"{_FPL_BASE}/fixtures/?event={gw}")
            for f in fixtures:
                if not f.get("finished", False):
                    continue
                h_score = f.get("team_h_score")
                a_score = f.get("team_a_score")
                if h_score is None or a_score is None:
                    continue                          # not yet played
                h_id = f.get("team_h")
                a_id = f.get("team_a")
                if h_id is None or a_id is None:
                    continue
                _apply_elo_update(
                    ratings, int(h_id), int(a_id), int(h_score), int(a_score)
                )
            last_processed = gw
        except Exception as exc:
            log.warning("Elo: GW%d fetch failed (%s) — skipping.", gw, exc)

    save_ratings(
        {"updated_gw": last_processed, "season": season, "ratings": ratings},
        path,
    )
    return ratings


# ---------------------------------------------------------------------------
# Fixture weight API
# ---------------------------------------------------------------------------

def get_fixture_weight(
    attacker_team_id: int,
    opponent_team_id: int,
    ratings: dict,
) -> float:
    """
    Convert Elo ratings into a fixture difficulty multiplier for the attacker.

    Formula:
      win_prob = 1 / (1 + 10 ** ((opp_elo - att_elo) / 400))
      weight   = 0.75 + win_prob * 0.45

    Output range: [0.75, 1.20]
      - 0.75 → attacker wins with probability ~0%  (hardest possible)
      - 1.00 → attacker wins with probability ~50% (neutral)
      - 1.20 → attacker wins with probability ~100% (easiest possible)

    Returns 1.0 (neutral) if either team is absent from ratings.
    """
    att_key = str(attacker_team_id)
    opp_key = str(opponent_team_id)

    if att_key not in ratings or opp_key not in ratings:
        return 1.0

    att_elo  = ratings[att_key]
    opp_elo  = ratings[opp_key]
    win_prob = 1.0 / (1.0 + 10.0 ** ((opp_elo - att_elo) / 400.0))
    return 0.75 + win_prob * 0.45


def get_team_fixture_weights_elo(
    team_id: int,
    upcoming_gws: list[int],
    opponent_ids: list[int],
    ratings: dict,
) -> list[float]:
    """
    Return fixture weights for a team across upcoming GWs.

    upcoming_gws and opponent_ids are parallel lists of the same length.
    opponent_id == -1 signals a blank gameweek for that team → weight 0.0.
    """
    weights = []
    for opp_id in opponent_ids:
        if opp_id == -1:
            weights.append(0.0)
        else:
            weights.append(get_fixture_weight(team_id, opp_id, ratings))
    return weights
