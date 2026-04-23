# ⚡ CLUTCH — Elite FPL Strategy Engine

A Monte Carlo simulation engine, differential-hunting assassin feed, press conference AI interpreter, and Egyptian Arabic content generator — built for Fantasy Premier League managers who treat FPL as a data problem.

---

## What it does

CLUTCH is split into two pages:

### FPL Decision Engine (`app.py`)
| Panel | What you get |
|---|---|
| **Win Probability** | 10,000-iteration Monte Carlo gives you a percentage chance of overtaking the mini-league leader by end of season, a per-GW expected points sparkline, and the statistical standard error so you know when a number is signal vs noise |
| **Assassin Feed** | Differential hunters: low-ownership players (< 15% globally) with high xGI (xG + xA) that your top-5 rivals aren't holding. Blanking players are flagged; BGW/DGW context is baked in |
| **Presser Intel** | Paste a manager's press conference quote, get back a structured JSON verdict: start likelihood 0–100, confidence level, key signals, risk flags, and a transfer action (Hold / Buy / Sell / Monitor) |
| **Transfer Lab** | What-if engine: swap any player out, any same-position player in, and see the full causal impact — win probability delta, raw xP gain per GW, cumulative clutch gain, and a SE-calibrated verdict (MAKE IT / WORTH IT / INCONCLUSIVE / HOLD / AVOID / LATERAL MOVE) |

### Content Creator (`pages/1_Content_Creator.py`)
Separate page for FPL content creators. Takes the simulation output and generates:
- **60-second YouTube script** in Egyptian Arabic (Ammiya) — energetic, structurally sound (hook → win prob → assassin move → transfer tip → CTA)
- **Social carousel captions** (5 slides) for Instagram, TikTok, or Twitter, with per-slide RTL/LTR detection

---

## Architecture

```
clutch/
├── app.py                   # Streamlit UI — FPL Decision Engine
├── pages/
│   └── 1_Content_Creator.py # Streamlit UI — Script & Caption Generator
├── fpl_api.py               # FPL REST API client
├── simulator.py             # Monte Carlo engine
├── nlp_utils.py             # LLM integration (OpenRouter)
├── .streamlit/
│   ├── config.toml          # Dark theme
│   └── secrets.toml.example # API key template
└── requirements.txt
```

### `fpl_api.py` — API layer

- Persistent `requests.Session` with `urllib3.Retry`: 3 attempts, exponential backoff (1s/2s/4s), retries on 429/500/502/503/504
- `get_bootstrap()` — full FPL season data (~3MB)
- `get_players_df(bootstrap)` — master player DataFrame with `xG`, `xA`, `xGI`, `ppg`, `ownership_pct`, `form`; missing stat columns fill with 0 and emit `RuntimeWarning` rather than silently substituting PPG
- `get_gw_info(gw, bootstrap)` — deadline, DGW/BGW detection, minutes-to-deadline urgency
- `compute_ownership_gap(...)` — Assassin feed; accepts pre-fetched `top5_standings`, `target_picks`, and `rival_picks` to avoid redundant API calls

### `simulator.py` — Monte Carlo engine

Player-level simulation, not squad-average. Every player draws independently.

**Per-player σ formula:**
```
σ_i = position_base + |form - ppg| × 0.3
σ_i = clamp(σ_i, floor=1.0, ceiling=max(ppg × 0.8, base × 1.5))
```

Position bases: GK 2.5 · DEF 3.5 · MID 4.5 · FWD 5.5

The ceiling prevents malformed API data (e.g. `form=25, ppg=3`) from producing distributions that regularly sample negative or 30+ point gameweeks.

**Starting XI selection — FPL-legal two-pass:**
1. Fill positional floors (1 GK, 3 DEF, 2 MID, 1 FWD) by highest PPG within each position
2. Fill remaining 4 flex slots by PPG from the rest of the squad regardless of position

This ensures two managers' simulations are comparable even if one has five premium midfielders.

**DGW/BGW handling:**
- DGW player: two independent draws summed (`draws += rng.normal(ppg, σ, ...)`)
- BGW player: contribution is exactly 0 (not noise-bumped)
- Captain: their draw is added a second time (2× multiplier)

**SE-calibrated verdicts in `simulate_transfer`:**
```
SE = sqrt(p(1-p) / n) × 100
noise_floor = 2 × SE
```
A delta is only labelled WORTH IT or better if it exceeds the noise floor — sub-noise deltas are INCONCLUSIVE.

**Captain auto-promotion:** when the transferred-out player was the captain, the highest-PPG player in the new squad is promoted. The return dict includes `captain_promoted: bool` and `promoted_captain: str` so the UI can surface this to the user.

### `nlp_utils.py` — LLM integration

- Provider: [OpenRouter](https://openrouter.ai) — OpenAI-compatible endpoint, **not** the Groq SDK
- Model: `meta-llama/llama-3.3-70b-instruct`
- Temperature: 0.8 for scripts (feel-based, not ablation-tested), 0.2 for presser analysis

**`interpret_presser`** returns structured JSON: `start_likelihood`, `confidence`, `key_signals`, `risk_flags`, `impact_summary`, `transfer_action`. Falls back to an Egyptian Arabic heuristic response (`_ammiya_fallback`) on API failure — internal error state is logged, not rendered to the user.

**`generate_video_script`** — hope filter: win probability below 5% is replaced with a motivational Arabic phrase rather than a demoralising number. `win_prob_raw` is cast to `float | None` at extraction time; `None` gets its own prompt branch so malformed sim data never reaches the LLM as a garbage string.

**`batch_interpret`** — genuinely parallel via `ThreadPoolExecutor`; results returned in input order.

---

## Setup

### Local

```bash
git clone https://github.com/your-username/clutch-fpl
cd clutch-fpl
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml and add your OpenRouter key
streamlit run app.py
```

### Streamlit Cloud

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. In **App Settings → Secrets**, paste:
   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   ```
4. Deploy — no other config needed

---

## Configuration

### API key resolution (three-layer fallback)
1. `st.secrets["OPENROUTER_API_KEY"]` — Streamlit Cloud secrets
2. `OPENROUTER_API_KEY` environment variable — local `.env` or shell export
3. Sidebar password input — manual entry at runtime

### Caching
Bootstrap data (~3MB) is cached for 5 minutes via `@st.cache_data(ttl=300)`. League standings and per-manager picks are cached with the same TTL. A single "RUN CLUTCH" click makes exactly:
- 1 bootstrap fetch
- 1 fixtures fetch (for DGW/BGW detection)
- 1 standings fetch
- N+1 picks fetches (target + each rival)
- 1 manager info fetch

No fetch is made more than once per 5-minute window.

### `.streamlit/config.toml`
```toml
[theme]
base = "dark"
primaryColor = "#00e5ff"
backgroundColor = "#0d0d0d"
secondaryBackgroundColor = "#111122"
textColor = "#e0e0e0"
```

---

## Usage

### 1. Configure the sidebar
| Field | What to enter |
|---|---|
| Your Manager ID | Found in your FPL profile URL: `fantasy.premierleague.com/entry/{ID}/` |
| Mini-League ID | Found in the mini-league URL: `fantasy.premierleague.com/leagues/{ID}/standings/` |
| Current GW | The gameweek you want to simulate |
| GWs Remaining | How many gameweeks left in your race |
| MC Iterations | 10,000 recommended; 1,000 for fast preview |

### 2. Click RUN CLUTCH
Fetches live FPL data and runs the simulation. The three tabs populate:
- **Win Probability** — gauge, sparkline, captain badge, gap summary
- **Assassin Feed** — differential table + top BUY targets
- **Presser Intel** — waiting for a quote (see step 4)

### 3. Use the Transfer Lab
Select a player out (from your squad) and a player in (same position, sorted by xGI). Click **Run What-If**. The verdict card shows:
- Verdict label with colour coding
- Win probability before/after with SE and noise floor
- Raw Points tab: per-GW xP gain bars + cumulative clutch gain line
- PPG shift / xGI shift / Gap Δ strip

> **Inertia threshold** (Advanced Settings): the minimum PPG gain required to avoid a "Lateral Move" verdict. Default 1.0 is a starting point — tune it to your own transfer philosophy. It is not calibrated against the simulation model.

### 4. Analyze a press conference quote
Paste any manager's quote and the player name in the sidebar. Works independently of the main simulation — useful mid-week before deadline.

### 5. Content Creator page
Open via the sidebar navigation. Requires the main simulation to have been run first (picks up from session state). Generate a YouTube script or social caption in one click.

---

## Key design decisions

**xGI not xGA** — the attacking metric is xG + xA = expected Goal Involvement. xGA is expected Goals Against, a defensive metric. Every column, table header, and prompt in this codebase uses xGI.

**Pass data down, never re-fetch** — `compute_ownership_gap` accepts pre-computed picks and standings. `get_players_df` accepts a bootstrap dict. `get_gw_info` accepts a bootstrap dict. This eliminates redundant network calls within a single run.

**FPL-legal starting XI** — the simulator enforces positional floors (1 GK, 3 DEF, 2 MID, 1 FWD) before filling flex slots. A squad with five premium midfielders doesn't get simulated with five midfielders in the XI.

**Sigma ceiling** — `_player_sigma` caps at `max(ppg × 0.8, base × 1.5)`. Without this, a player returning from injury with low season PPG and high recent form can produce sigma > 10, which regularly samples negative or 30+ point gameweeks from a Normal distribution — physically impossible in FPL.

**Bilingual error handling** — the Presser Intel fallback responds in Egyptian Arabic, consistent with the tool's target audience. Internal error state (reason codes, exception details) goes to `logging.warning`, not to the UI.

**Content Creator is a separate page** — the FPL decision engine (Win Probability, Assassin, Transfer Lab, Presser) and the content creation tool (scripts, captions) are different use cases with different users. Streamlit multipage keeps them separate while sharing session state.

---

## Requirements

```
requests>=2.31
pandas>=2.1
numpy>=1.26
streamlit>=1.35
plotly>=5.20
openai>=1.30
```

Python 3.11+ recommended (uses `list[int]`, `dict[int, list[int]]` type hints throughout).

---

## Limitations

- FPL API is unauthenticated and occasionally flaky — retry logic handles transient 5xx but sustained outages will fail
- xGI thresholds and inertia threshold are not statistically calibrated against historical FPL data
- Monte Carlo uses Normal distributions; actual FPL score distributions are right-skewed (hauls are fat-tailed). The model understates haul probability for premium attackers
- `simulate_transfer` computes expected points from season PPG, not fixture-adjusted xG — a player with a great upcoming run won't show a higher `in_ppg` than their season average reflects
- Content Creator scripts are generated in Egyptian Arabic (عامية مصرية) and are not suitable for English-language channels without translation
