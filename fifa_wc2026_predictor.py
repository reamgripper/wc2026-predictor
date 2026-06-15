"""
FIFA World Cup 2026 Match Prediction Engine
============================================
A probabilistic match outcome predictor using Poisson regression, Monte Carlo
simulation, and uncertainty-aware confidence intervals.

Data Sources (all free, no API key required)
---------------------------------------------
1. jfjelstul/worldcup (GitHub raw CSV)
   1,248 FIFA World Cup matches, 1930–2022.
   URL: https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/matches.csv

2. fixturedownload.com (JSON feed)
   Recent major international tournaments with complete scorelines:
   - FIFA World Cup 2018 & 2022
   - UEFA Euro 2020 & 2024
   - Copa América 2021 & 2024
   - AFCON 2023
   - AFC Asian Cup 2023

3. eloratings.net (TSV)
   Live World Football ELO ratings, updated after every match.
   Used as a team-quality prior to calibrate the uncertainty metric and
   adjust for opponent strength in the Poisson regression.

Mathematical Rationale for Margin of Error
-------------------------------------------
Goal differential (GD = goals_A − goals_B) is computed across 10,000 Monte Carlo
draws. Each draw independently samples from Poisson(λ_A) and Poisson(λ_B), where
λ values come from a Poisson regression model trained on the merged historical
dataset.

The 95% CI is derived empirically from the 2.5th and 97.5th percentiles of the
simulated GD distribution — NOT a parametric normal approximation. This is
intentional: the difference of two Poisson variates follows a Skellam distribution,
which is skewed for asymmetric λ pairs. The empirical percentile method captures
this skewness naturally.

The "Uncertainty Metric" widens the CI by inflating λ with additive Gaussian
noise (std = uncertainty × λ) before each Poisson draw. This models epistemic
uncertainty: teams with fewer matches against strong opposition have higher
variance in their true λ estimate, so the CI expands accordingly.
"""

import io
import json
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CACHE PATHS
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR     = _PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HIST_PARQUET        = CACHE_DIR / "historical_matches.parquet"
WC2026_PARQUET      = CACHE_DIR / "wc2026_results.parquet"
ELO_PARQUET         = CACHE_DIR / "elo_ratings.parquet"
PLAYER_STATS_PARQUET= CACHE_DIR / "player_stats_wc2026.parquet"
META_JSON           = CACHE_DIR / "metadata.json"

KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"
KAGGLE_CSV     = CACHE_DIR / "results.csv"                 # Kaggle manual drop path

# jfjelstul/worldcup dataset cache
JFJELSTUL_DIR         = CACHE_DIR / "jfjelstul"
WC_TEAM_STATS_PARQUET = JFJELSTUL_DIR / "wc_team_stats.parquet"
JFJELSTUL_DIR.mkdir(parents=True, exist_ok=True)

# ELO is small — refresh once per day; WC 2026 results refresh every run
ELO_TTL_HOURS  = 24


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 0: SQUAD & INJURY DATA  —  ESPN unofficial public API
# ─────────────────────────────────────────────────────────────────────────────

# Maps team display names used throughout this codebase to their ESPN WC team IDs.
ESPN_TEAM_IDS: Dict[str, int] = {
    "Algeria": 624, "Argentina": 202, "Australia": 628, "Austria": 474,
    "Belgium": 459, "Bosnia-Herzegovina": 452, "Brazil": 205, "Canada": 206,
    "Cape Verde": 2597, "Colombia": 208, "Congo DR": 2850, "Croatia": 477,
    "Curaçao": 11678, "Czechia": 450, "Ecuador": 209, "Egypt": 2620,
    "England": 448, "France": 478, "Germany": 481, "Ghana": 4469,
    "Haiti": 2654, "Iran": 469, "Iraq": 4375, "Ivory Coast": 4789,
    "Japan": 627, "Jordan": 2917, "Mexico": 203, "Morocco": 2869,
    "Netherlands": 449, "New Zealand": 2666, "Norway": 464, "Panama": 2659,
    "Paraguay": 210, "Portugal": 482, "Qatar": 4398, "Saudi Arabia": 655,
    "Scotland": 580, "Senegal": 654, "South Africa": 467, "South Korea": 451,
    "Spain": 164, "Sweden": 466, "Switzerland": 475, "Tunisia": 659,
    "Türkiye": 465, "Turkey": 465, "United States": 660, "Uruguay": 212,
    "Uzbekistan": 2570,
}

# Position group → attacking impact weight when player is injured/unavailable.
# Scale: 1.0 = full squad replacement assumed, higher = harder to replace.
POSITION_IMPACT: Dict[str, Dict[str, float]] = {
    "Forward":    {"attack": 0.12, "defence": 0.02},
    "Midfielder": {"attack": 0.07, "defence": 0.05},
    "Defender":   {"attack": 0.02, "defence": 0.10},
    "Goalkeeper": {"attack": 0.00, "defence": 0.06},
}

ESPN_ROSTER_URL   = ("https://site.api.espn.com/apis/site/v2/sports/soccer"
                     "/fifa.world/teams/{team_id}/roster")
ESPN_SCHEDULE_URL = ("https://site.api.espn.com/apis/site/v2/sports/soccer"
                     "/fifa.world/scoreboard?dates=20260611-20260712&limit=100")
ESPN_SUMMARY_URL  = ("https://site.api.espn.com/apis/site/v2/sports/soccer"
                     "/fifa.world/summary?event={event_id}")


def fetch_squad(team_name: str) -> List[Dict]:
    """
    Fetch the current WC 2026 squad for a team from the ESPN unofficial API.

    Returns a list of player dicts with keys:
        name, position, age, status, is_injured, jersey
    Returns [] if the team has no ESPN ID or the request fails.
    """
    team_id = ESPN_TEAM_IDS.get(team_name)
    if not team_id:
        return []

    try:
        r = _get(ESPN_ROSTER_URL.format(team_id=team_id))
        athletes = r.json().get("athletes", [])
    except Exception:
        return []

    players = []
    for a in athletes:
        pos_obj  = a.get("position") or {}
        pos_name = pos_obj.get("displayName", "Unknown") if isinstance(pos_obj, dict) else str(pos_obj)

        status_obj  = a.get("status") or {}
        status_type = status_obj.get("type", "active") if isinstance(status_obj, dict) else "active"
        status_name = status_obj.get("name", "Active") if isinstance(status_obj, dict) else "Active"

        injuries = a.get("injuries") or []
        injury_detail = ""
        if injuries:
            inj = injuries[0]
            injury_detail = inj.get("longComment") or inj.get("shortComment") or "Injured"

        is_injured = (
            status_type not in ("active",) or
            bool(injuries) or
            status_type in ("out", "doubtful", "day-to-day", "injured",
                            "questionable", "suspension")
        )

        players.append({
            "name":          a.get("fullName", "Unknown"),
            "jersey":        a.get("jersey", ""),
            "position":      pos_name,
            "age":           a.get("age"),
            "status":        status_name,
            "status_type":   status_type,
            "is_injured":    is_injured,
            "injury_detail": injury_detail,
            "espn_id":       a.get("id", ""),
        })

    return players


def squad_injury_adjustment(players: List[Dict]) -> Tuple[float, float, List[Dict]]:
    """
    Compute multiplicative λ adjustments based on injured/unavailable players.

    Logic
    -----
    Each injured player reduces the team's attacking or defensive λ by their
    position's impact weight (see POSITION_IMPACT). The cumulative adjustment
    is bounded at [-25%, 0%] — squad depth limits total degradation.

    Returns
    -------
    attack_adj  : float  multiplier for λ_attack  (e.g. 0.88 = 12% reduction)
    defence_adj : float  multiplier for λ_defence (e.g. 0.90 = 10% reduction)
    injured_list: list of dicts for the injured players
    """
    attack_pen  = 0.0
    defence_pen = 0.0
    injured     = [p for p in players if p["is_injured"]]

    for p in injured:
        pos = p["position"]
        # Map granular ESPN positions to our four buckets
        if any(k in pos for k in ("Forward", "Winger", "Striker", "Centre-Forward")):
            grp = "Forward"
        elif any(k in pos for k in ("Midfielder", "Midfield", "Attacking Mid",
                                     "Defensive Mid", "Central Mid")):
            grp = "Midfielder"
        elif any(k in pos for k in ("Defender", "Back", "Centre-Back", "Full-Back")):
            grp = "Defender"
        else:
            grp = "Goalkeeper"

        impact = POSITION_IMPACT.get(grp, {"attack": 0.05, "defence": 0.05})
        attack_pen  += impact["attack"]
        defence_pen += impact["defence"]

    # Cap at 25% degradation (squad depth floor)
    attack_pen  = min(attack_pen,  0.25)
    defence_pen = min(defence_pen, 0.25)

    return 1.0 - attack_pen, 1.0 - defence_pen, injured


def manual_injury_adjustment(
    squad: List[Dict],
    injured_names: List[str],
) -> Tuple[float, float, List[Dict]]:
    """
    Apply injury penalties for manually entered player names.

    Matches each name (case-insensitive substring) against the fetched squad
    to retrieve position. Unmatched names default to Midfielder impact.
    Returns the same (atk_adj, def_adj, injured_list) tuple as
    squad_injury_adjustment.
    """
    if not injured_names:
        return 1.0, 1.0, []

    attack_pen  = 0.0
    defence_pen = 0.0
    injured_list: List[Dict] = []

    squad_lower = {p["name"].lower(): p for p in squad}

    for raw_name in injured_names:
        name = raw_name.strip()
        if not name:
            continue

        # Exact match first, then substring
        player = squad_lower.get(name.lower())
        if player is None:
            for sq_name, sq_player in squad_lower.items():
                if name.lower() in sq_name or sq_name in name.lower():
                    player = sq_player
                    break

        pos = player["position"] if player else "Midfielder"

        if any(k in pos for k in ("Forward", "Winger", "Striker", "Centre-Forward")):
            grp = "Forward"
        elif any(k in pos for k in ("Midfielder", "Midfield", "Attacking Mid",
                                     "Defensive Mid", "Central Mid")):
            grp = "Midfielder"
        elif any(k in pos for k in ("Defender", "Back", "Centre-Back", "Full-Back")):
            grp = "Defender"
        else:
            grp = "Goalkeeper"

        impact = POSITION_IMPACT.get(grp, {"attack": 0.05, "defence": 0.05})
        attack_pen  += impact["attack"]
        defence_pen += impact["defence"]

        injured_list.append({
            "name":          player["name"] if player else name,
            "position":      pos,
            "status":        "Injured (manual)",
            "status_type":   "out",
            "is_injured":    True,
            "injury_detail": "Entered manually",
            "jersey":        player.get("jersey", "") if player else "",
            "age":           player.get("age") if player else None,
        })

    attack_pen  = min(attack_pen,  0.25)
    defence_pen = min(defence_pen, 0.25)
    return 1.0 - attack_pen, 1.0 - defence_pen, injured_list


ESPN_DATE_URL = ("https://site.api.espn.com/apis/site/v2/sports/soccer"
                 "/fifa.world/scoreboard?dates={date}")

# Status names ESPN uses when a match has finished
_FINAL_STATUSES = {
    "STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FT",
    "STATUS_FULL_TIME_OT", "STATUS_PENALTY",
}


def fetch_wc2026_results() -> pd.DataFrame:
    """
    Fetch every completed WC 2026 match result by scanning dates from the
    tournament start (11 Jun 2026) through today.

    Uses per-date ESPN scoreboard queries because the bulk schedule endpoint
    only marks a match completed=True after the page refreshes, whereas the
    date-specific endpoint reflects STATUS_FULL_TIME immediately.

    Returns a DataFrame ready to be appended to the training set.
    """
    from datetime import date, timedelta
    import time as _time

    start = date(2026, 6, 11)
    today = date.today()
    rows  = []

    d = start
    while d <= today:
        date_str = d.strftime("%Y%m%d")
        try:
            r = _get(ESPN_DATE_URL.format(date=date_str))
            events = r.json().get("events", [])
        except Exception:
            d += timedelta(days=1)
            continue

        for e in events:
            status_type = e.get("status", {}).get("type", {})
            status_name = status_type.get("name", "")
            completed   = (
                status_type.get("completed", False) or
                status_name in _FINAL_STATUSES
            )
            if not completed:
                d += timedelta(days=0)   # still scan the day
                continue

            comps       = e.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"),
                        competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"),
                        competitors[1])

            try:
                h_score = int(home.get("score", 0))
                a_score = int(away.get("score", 0))
            except (TypeError, ValueError):
                continue

            rows.append({
                "date":          d.isoformat(),
                "home_team":     home.get("team", {}).get("displayName", ""),
                "away_team":     away.get("team", {}).get("displayName", ""),
                "home_score":    h_score,
                "away_score":    a_score,
                "tournament":    "FIFA World Cup",
                "neutral_venue": True,
                "event_id":      e.get("id", ""),
                "status":        status_name,
            })

        d += timedelta(days=1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_wc2026_fixtures() -> pd.DataFrame:
    """
    Fetch all WC 2026 scheduled/completed fixtures from ESPN.
    Returns a DataFrame with columns: event_id, date, home_team, away_team,
    home_score, away_score, status, group.
    """
    try:
        r = _get(ESPN_SCHEDULE_URL)
        events = r.json().get("events", [])
    except Exception:
        return pd.DataFrame()

    rows = []
    for e in events:
        comps = e.get("competitions", [{}])[0]
        competitors = comps.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        status_obj   = e.get("status", {})
        status_type  = status_obj.get("type", {}).get("name", "STATUS_SCHEDULED")
        completed    = status_obj.get("type", {}).get("completed", False)

        h_score = int(home.get("score", 0)) if completed else None
        a_score = int(away.get("score", 0)) if completed else None

        rows.append({
            "event_id":   e["id"],
            "date":       e.get("date", "")[:10],
            "home_team":  home.get("team", {}).get("displayName", ""),
            "away_team":  away.get("team", {}).get("displayName", ""),
            "home_score": h_score,
            "away_score": a_score,
            "status":     status_type,
            "completed":  completed,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1A: DATA INGESTION  —  fetch from free public APIs
# ─────────────────────────────────────────────────────────────────────────────

# Tournament weight — more competitive = higher weight in the Poisson model
TOURNAMENT_WEIGHTS: Dict[str, float] = {
    "FIFA World Cup":               1.00,
    "UEFA Euro":                    0.90,
    "Copa América":                 0.90,
    "AFCON":                        0.80,
    "AFC Asian Cup":                0.80,
    "CONCACAF Gold Cup":            0.75,
    "FIFA World Cup qualification": 0.75,
    "Nations League":               0.70,
    "Friendly":                     0.40,
}

# WC 2026 host nations
WC2026_HOSTS = {"United States", "Canada", "Mexico"}

# Multiplicative λ boost for a host nation playing in its own country.
# Literature on home/host advantage in international tournaments puts the
# effect around +15–25% expected goals; 1.20 is the midpoint. Applied at
# prediction time in predict_lambdas (see note there on why the GLM's own
# fitted host coefficient is unusable).
HOST_BOOST = 1.20

# fixturedownload slugs → (label, tournament_weight, is_neutral)
FIXTURE_SOURCES = [
    ("fifa-world-cup-2022",  "FIFA World Cup",  1.00, False),
    ("fifa-world-cup-2018",  "FIFA World Cup",  1.00, False),
    ("uefa-euro-2024",       "UEFA Euro",       0.90, True),
    ("uefa-euro-2020",       "UEFA Euro",       0.90, True),
    ("copa-america-2024",    "Copa América",    0.90, True),
    ("copa-america-2021",    "Copa América",    0.90, True),
    ("afcon-2023",           "AFCON",           0.80, True),
    ("afc-asian-cup-2023",   "AFC Asian Cup",   0.80, True),
]

ELO_URL     = "https://www.eloratings.net/World.tsv"
WC_HIST_URL = ("https://raw.githubusercontent.com/jfjelstul/worldcup"
               "/master/data-csv/matches.csv")
FIXTURE_URL = "https://fixturedownload.com/feed/json/{slug}"


def _get(url: str, **kwargs) -> requests.Response:
    """HTTP GET with a polite retry on transient failures."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(1.5)


def fetch_elo_ratings() -> Dict[str, float]:
    """
    Download current World Football ELO ratings from eloratings.net.

    TSV columns (no header): rank, prev_rank, country_code, elo, ...
    Returns a dict mapping 3-letter country code → ELO rating.
    """
    print("  Fetching ELO ratings from eloratings.net ...")
    r = _get(ELO_URL)
    ratings: Dict[str, float] = {}
    for line in r.text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            code = parts[2].strip()
            try:
                elo = float(parts[3].replace("−", "-").replace(",", ""))
                ratings[code] = elo
            except ValueError:
                pass
    print(f"    → {len(ratings)} teams loaded.")
    return ratings


def _elo_code_to_name_map() -> Dict[str, str]:
    """
    Manually curated mapping from eloratings.net 2-/3-letter codes to the
    team names used in the fixturedownload and jfjelstul datasets.
    Only WC-relevant teams are included; extend as needed.
    """
    return {
        "ES": "Spain",        "AR": "Argentina",    "FR": "France",
        "EN": "England",      "BR": "Brazil",       "PT": "Portugal",
        "CO": "Colombia",     "NL": "Netherlands",  "EC": "Ecuador",
        "DE": "Germany",      "NO": "Norway",       "HR": "Croatia",
        "TR": "Turkey",       "JP": "Japan",        "BE": "Belgium",
        "UY": "Uruguay",      "CH": "Switzerland",  "MX": "Mexico",
        "DK": "Denmark",      "IT": "Italy",        "MA": "Morocco",
        "SN": "Senegal",      "NG": "Nigeria",      "CI": "Côte d'Ivoire",
        "GH": "Ghana",        "CM": "Cameroon",     "EG": "Egypt",
        "AU": "Australia",    "KR": "South Korea",  "IR": "Iran",
        "SA": "Saudi Arabia", "QA": "Qatar",        "US": "United States",
        "CA": "Canada",       "PA": "Panama",       "JM": "Jamaica",
        "GT": "Guatemala",    "SV": "El Salvador",  "HN": "Honduras",
        "CR": "Costa Rica",   "CL": "Chile",        "PE": "Peru",
        "BO": "Bolivia",      "VE": "Venezuela",    "PY": "Paraguay",
        "AT": "Austria",      "CZ": "Czech Republic","PL": "Poland",
        "HU": "Hungary",      "SE": "Sweden",       "RS": "Serbia",
        "SK": "Slovakia",     "RO": "Romania",      "GR": "Greece",
        "UA": "Ukraine",      "AL": "Albania",      "GE": "Georgia",
        "SI": "Slovenia",     "SC": "Scotland",     "IE": "Ireland",
        "AM": "Armenia",      "KZ": "Kazakhstan",   "AZ": "Azerbaijan",
        "OM": "Oman",         "KW": "Kuwait",       "JO": "Jordan",
        "IQ": "Iraq",         "BH": "Bahrain",      "LB": "Lebanon",
        # WC 2026 qualifiers previously missing — defaulting them to elo_min
        # silently mislabeled their strength (e.g. South Africa in the opener).
        "ZA": "South Africa", "DZ": "Algeria",      "TN": "Tunisia",
        "CV": "Cape Verde",   "CD": "Congo DR",     "CW": "Curaçao",
        "HT": "Haiti",        "NZ": "New Zealand",  "UZ": "Uzbekistan",
        "BA": "Bosnia-Herzegovina",
    }


# ESPN team names that differ from the canonical names used above / in the
# historical datasets. Both spellings get the same ELO entry.
_TEAM_NAME_ALIASES = {
    "Czechia":      "Czech Republic",
    "Türkiye":      "Turkey",
    "Ivory Coast":  "Côte d'Ivoire",
}


# ─────────────────────────────────────────────────────────────────────────────
# COUNTRY ALTERNATE SPELLINGS  — one registry for all external sources
# ─────────────────────────────────────────────────────────────────────────────
# Polymarket, DraftKings/ESPN and other feeds spell some nations differently
# (Cabo Verde vs Cape Verde, Türkiye vs Turkey, …). List every alternate
# spelling for a team here ONCE; both the betting-market match-up lookup and the
# Polymarket per-match lookup match against these, accent/case/punctuation-
# insensitively. To add a country: "App name": ["Alt 1", "Alt 2", ...].
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_ALIASES: Dict[str, List[str]] = {
    "USA":                     ["United States", "United States of America", "US"],
    "South Korea":             ["Korea Republic", "Korea", "Republic of Korea"],
    "North Korea":             ["Korea DPR", "DPR Korea"],
    "Ivory Coast":             ["Cote d'Ivoire", "Côte d'Ivoire"],
    "Netherlands":             ["Holland"],
    "Cape Verde":              ["Cabo Verde"],
    "Czechia":                 ["Czech Republic"],
    "Turkey":                  ["Turkiye", "Türkiye"],
    "Iran":                    ["IR Iran", "Islamic Republic of Iran"],
    "Bosnia and Herzegovina":  ["Bosnia", "Bosnia & Herzegovina"],
    "DR Congo":                ["Congo DR", "Democratic Republic of the Congo"],
    "Curacao":                 ["Curaçao"],
    "United Arab Emirates":    ["UAE"],
    "Republic of Ireland":     ["Ireland"],
}


def _norm_name(s: str) -> str:
    """Accent/case/punctuation-insensitive key for a team name."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def country_name_variants(team: str) -> set:
    """All normalized spellings for `team`: itself + every alias, in both
    directions (whether `team` is the canonical key or one of its aliases)."""
    raw = {team}
    raw.update(COUNTRY_ALIASES.get(team, []))
    for canon, alts in COUNTRY_ALIASES.items():
        if team == canon or team in alts:
            raw.add(canon)
            raw.update(alts)
    return {_norm_name(v) for v in raw if v}


def _names_match(a: str, b: str) -> bool:
    """True if two team names refer to the same country under known aliases."""
    return bool(country_name_variants(a) & country_name_variants(b))


# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read_meta() -> dict:
    if META_JSON.exists():
        try:
            return json.loads(META_JSON.read_text())
        except Exception:
            pass
    return {}


def _write_meta(meta: dict) -> None:
    META_JSON.write_text(json.dumps(meta, indent=2, default=str))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce date column and keep only the five core columns."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["home_score"] = df["home_score"].fillna(0).astype(int)
    df["away_score"] = df["away_score"].fillna(0).astype(int)
    if "neutral_venue" not in df.columns:
        df["neutral_venue"] = False
    if "tournament" not in df.columns:
        df["tournament"] = "Friendly"
    return df[["date", "home_team", "away_team",
               "home_score", "away_score", "tournament", "neutral_venue"]]


# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE / HISTORICAL BASE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _load_kaggle_csv(csv_path: Path) -> pd.DataFrame:
    """Parse the martj42 Kaggle CSV into the unified schema."""
    raw = pd.read_csv(csv_path)
    # Kaggle schema: date, home_team, away_team, home_score, away_score,
    #                tournament, city, country, neutral
    df = pd.DataFrame({
        "date":          raw["date"],
        "home_team":     raw["home_team"],
        "away_team":     raw["away_team"],
        "home_score":    raw["home_score"],
        "away_score":    raw["away_score"],
        "tournament":    raw["tournament"],
        "neutral_venue": raw["neutral"].astype(bool),
    })
    return _normalize(df)


def _download_kaggle_dataset() -> Optional[pd.DataFrame]:
    """
    Try to download the martj42 Kaggle dataset via the kaggle Python API.
    Returns the DataFrame on success, None if kaggle is not configured.
    """
    try:
        import kaggle  # noqa: F401 — triggers auth check
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        print("  Downloading Kaggle dataset (martj42/international-football-results)...")
        api.dataset_download_files(
            KAGGLE_DATASET, path=str(CACHE_DIR), unzip=True, quiet=False
        )
        # The zip extracts to CACHE_DIR/results.csv
        csv = CACHE_DIR / "results.csv"
        if csv.exists():
            df = _load_kaggle_csv(csv)
            print(f"    → {len(df)} matches downloaded from Kaggle.")
            return df
    except Exception as e:
        print(f"  Kaggle download skipped: {e}")
    return None


def load_historical_base() -> pd.DataFrame:
    """
    Load the full international football results base dataset.

    Priority order:
      1. Cached parquet (fastest — no network)
      2. Manual CSV drop at data/cache/results.csv
      3. Kaggle API download (requires ~/.kaggle/kaggle.json)
      4. Legacy fallback: jfjelstul WC-only + fixturedownload

    On first successful load from any source, the result is saved to
    data/cache/historical_matches.parquet for all future startups.
    """
    # ── 1. Parquet cache (warm start) ─────────────────────────────────────────
    if HIST_PARQUET.exists():
        df = pd.read_parquet(HIST_PARQUET)
        meta = _read_meta()
        print(f"  ✓ Loaded {len(df):,} historical matches from local cache "
              f"(last updated {meta.get('hist_updated_at', 'unknown')}).")
        return df

    # ── 2. Manual CSV drop ────────────────────────────────────────────────────
    if KAGGLE_CSV.exists():
        print(f"  Loading Kaggle CSV from {KAGGLE_CSV} ...")
        df = _load_kaggle_csv(KAGGLE_CSV)
        print(f"    → {len(df):,} matches loaded.")
        _save_historical(df)
        return df

    # ── 3. Kaggle API ─────────────────────────────────────────────────────────
    df = _download_kaggle_dataset()
    if df is not None:
        _save_historical(df)
        return df

    # ── 4. Legacy fallback ────────────────────────────────────────────────────
    print("  Kaggle dataset not available — using legacy sources (WC-only + fixturedownload).")
    return _fetch_legacy_historical()


def _save_historical(df: pd.DataFrame) -> None:
    df.to_parquet(HIST_PARQUET, index=False)
    meta = _read_meta()
    meta["hist_updated_at"] = _now_iso()
    meta["hist_row_count"]  = len(df)
    meta["hist_max_date"]   = str(df["date"].max().date())
    _write_meta(meta)
    print(f"  ✓ Saved {len(df):,} matches to {HIST_PARQUET.name}.")


def _fetch_legacy_historical() -> pd.DataFrame:
    """Original download path — used only when Kaggle data is unavailable."""
    frames: List[pd.DataFrame] = []

    # jfjelstul WC 1930-2022
    try:
        print("  Fetching WC historical (jfjelstul) ...")
        r = _get(WC_HIST_URL)
        raw = pd.read_csv(io.StringIO(r.text))
        df = pd.DataFrame({
            "date":          pd.to_datetime(raw["match_date"]),
            "home_team":     raw["home_team_name"],
            "away_team":     raw["away_team_name"],
            "home_score":    raw["home_team_score"],
            "away_score":    raw["away_team_score"],
            "tournament":    "FIFA World Cup",
            "neutral_venue": True,
        })
        frames.append(_normalize(df.dropna(subset=["home_score", "away_score"])))
        print(f"    → {len(frames[-1])} matches.")
    except Exception as e:
        print(f"  ✗ jfjelstul failed: {e}")

    # fixturedownload recent tournaments
    for slug, label, _w, neutral in FIXTURE_SOURCES:
        try:
            r = _get(FIXTURE_URL.format(slug=slug))
            rows = []
            for m in r.json():
                hs, as_ = m.get("HomeTeamScore"), m.get("AwayTeamScore")
                if hs is None or as_ is None:
                    continue
                rows.append({"date": pd.to_datetime(m["DateUtc"]),
                             "home_team": m["HomeTeam"], "away_team": m["AwayTeam"],
                             "home_score": int(hs), "away_score": int(as_),
                             "tournament": label, "neutral_venue": neutral})
            if rows:
                frames.append(_normalize(pd.DataFrame(rows)))
                print(f"    → {len(rows)} {label} matches.")
        except Exception as e:
            print(f"  ✗ {slug} failed: {e}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        _save_historical(combined)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# WC 2026 INCREMENTAL CACHE
# ─────────────────────────────────────────────────────────────────────────────

def load_wc2026_cached() -> pd.DataFrame:
    """
    Load WC 2026 results: merge persisted cache with any new ESPN results.
    Only matches newer than the cached max_date are fetched from ESPN.
    """
    cached = pd.DataFrame()
    if WC2026_PARQUET.exists():
        cached = pd.read_parquet(WC2026_PARQUET)

    print("  Checking ESPN for new WC 2026 results ...")
    try:
        live = fetch_wc2026_results()
    except Exception as exc:
        print(f"    ⚠ ESPN fetch failed ({exc}); using {len(cached)} cached result(s).")
        live = pd.DataFrame()

    if live.empty and cached.empty:
        return pd.DataFrame()

    if live.empty:
        print(f"    → No new results (using {len(cached)} cached).")
        return cached

    live = _normalize(live[["date","home_team","away_team",
                             "home_score","away_score","tournament","neutral_venue"]])

    if cached.empty:
        combined = live
    else:
        combined = pd.concat([cached, live], ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        )

    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_parquet(WC2026_PARQUET, index=False)

    new_rows = len(combined) - len(cached)
    if new_rows > 0:
        print(f"    → {new_rows} new WC 2026 result(s) appended (total {len(combined)}).")
    else:
        print(f"    → No new results ({len(combined)} total).")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# ELO CACHE  (refreshed at most once per day)
# ─────────────────────────────────────────────────────────────────────────────

def load_elo_cached() -> Dict[str, float]:
    """Load ELO ratings from disk if fresh (<24 h), otherwise re-fetch."""
    meta = _read_meta()
    if ELO_PARQUET.exists():
        fetched_at = meta.get("elo_fetched_at", "")
        if fetched_at:
            age_h = (datetime.now(timezone.utc) -
                     datetime.fromisoformat(fetched_at)).total_seconds() / 3600
            if age_h < ELO_TTL_HOURS:
                elo_df = pd.read_parquet(ELO_PARQUET)
                elo = dict(zip(elo_df["name"], elo_df["elo"]))
                print(f"  ✓ ELO ratings from cache ({age_h:.1f}h old, "
                      f"{len(elo)} teams).")
                return elo

    # Live (re)fetch — but never let a blocked/failed request crash the app.
    # Fall back to the cached parquet (even if stale) when the source is
    # unreachable, e.g. eloratings.net blocking a cloud datacenter IP.
    try:
        elo_by_code = fetch_elo_ratings()
        code_to_name = _elo_code_to_name_map()
        elo_by_name = {code_to_name[c]: v for c, v in elo_by_code.items()
                       if c in code_to_name}
        if not elo_by_name:
            raise RuntimeError("empty ELO response")

        elo_df = pd.DataFrame({"name": list(elo_by_name.keys()),
                                "elo":  list(elo_by_name.values())})
        elo_df.to_parquet(ELO_PARQUET, index=False)
        meta["elo_fetched_at"] = _now_iso()
        _write_meta(meta)
        return elo_by_name
    except Exception as exc:
        if ELO_PARQUET.exists():
            elo_df = pd.read_parquet(ELO_PARQUET)
            elo = dict(zip(elo_df["name"], elo_df["elo"]))
            print(f"  ⚠ ELO live fetch failed ({exc}); using cached "
                  f"parquet ({len(elo)} teams).")
            return elo
        print(f"  ⚠ ELO live fetch failed ({exc}) and no cache available.")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION  (now cache-aware)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1C: JFJELSTUL WORLDCUP DATASET  — github.com/jfjelstul/worldcup
# ─────────────────────────────────────────────────────────────────────────────
# Derives per-team WC heritage features from two tables:
#   team_appearances  — goals scored/conceded per match (1930-2022)
#   qualified_teams   — deepest stage reached per tournament
#
# Features per team (used as additional GLM covariates):
#   wc_attack_rate    — goals per WC match, adjusted for tournament era
#   wc_defence_rate   — goals conceded per WC match
#   wc_win_rate       — WC match win fraction
#   wc_stage_score    — avg best-stage score (0=group … 5=winner)
#   wc_appearances    — number of World Cup tournaments entered
# ─────────────────────────────────────────────────────────────────────────────

_JFJELSTUL_BASE = (
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
)

# Predictor name → name used in jfjelstul dataset
_JFJELSTUL_TEAM_ALIASES: Dict[str, str] = {
    "USA":               "United States",
    "South Korea":       "South Korea",
    "Ivory Coast":       "Ivory Coast",
    "Cote d'Ivoire":     "Ivory Coast",
    "Netherlands":       "Netherlands",
    "Czech Republic":    "Czechoslovakia",  # closest predecessor
    "North Macedonia":   "North Macedonia",
    "Bosnia-Herzegovina":"Bosnia and Herzegovina",
    "Trinidad & Tobago": "Trinidad and Tobago",
}

# Stage depth scoring: deeper = higher number
_STAGE_SCORES: Dict[str, float] = {
    "group stage":         0.0,
    "final round":         0.5,
    "second group stage":  1.0,
    "round of 16":         1.5,
    "quarter-final":       2.0,
    "quarter-finals":      2.0,
    "third-place match":   3.0,
    "semi-finals":         3.5,
    "final":               4.0,
    "winner":              5.0,
}


def _jf_team_name(team: str) -> str:
    return _JFJELSTUL_TEAM_ALIASES.get(team, team)


def _download_jf_csv(table: str) -> pd.DataFrame:
    """Download one jfjelstul/worldcup CSV table."""
    url = f"{_JFJELSTUL_BASE}/{table}.csv"
    try:
        r = _get(url)
        from io import StringIO
        return pd.read_csv(StringIO(r.text))
    except Exception as exc:
        print(f"  ⚠ Could not fetch jfjelstul/{table}.csv: {exc}")
        return pd.DataFrame()


def load_wc_team_stats() -> pd.DataFrame:
    """
    Return a DataFrame indexed by team_name with WC heritage features.

    Loads from parquet cache if available; otherwise downloads the two
    jfjelstul tables, aggregates, saves, and returns.

    Columns: team_name, wc_attack_rate, wc_defence_rate, wc_win_rate,
             wc_stage_score, wc_appearances
    """
    if WC_TEAM_STATS_PARQUET.exists():
        df = pd.read_parquet(WC_TEAM_STATS_PARQUET)
        print(f"  ✓ WC heritage stats  : {len(df)} teams (cache)")
        return df

    print("  ↓ Downloading jfjelstul/worldcup CSVs …")
    appearances_df  = _download_jf_csv("team_appearances")
    qualified_df    = _download_jf_csv("qualified_teams")

    if appearances_df.empty or qualified_df.empty:
        print("  ⚠ jfjelstul data unavailable — skipping WC heritage features")
        return pd.DataFrame(columns=["team_name","wc_attack_rate","wc_defence_rate",
                                     "wc_win_rate","wc_stage_score","wc_appearances"])

    # Exclude replayed matches to avoid double-counting
    if "replay" in appearances_df.columns:
        appearances_df = appearances_df[appearances_df["replay"] == 0]

    # Per-team match-level stats
    match_stats = (
        appearances_df
        .groupby("team_name")
        .agg(
            matches      = ("key_id",        "count"),
            goals_for    = ("goals_for",     "sum"),
            goals_against= ("goals_against", "sum"),
            wins         = ("win",           "sum"),
        )
        .reset_index()
    )
    match_stats["wc_attack_rate"]  = match_stats["goals_for"]     / match_stats["matches"]
    match_stats["wc_defence_rate"] = match_stats["goals_against"]  / match_stats["matches"]
    match_stats["wc_win_rate"]     = match_stats["wins"]           / match_stats["matches"]

    # Per-team tournament-level stage depth
    qualified_df["stage_score"] = qualified_df["performance"].map(_STAGE_SCORES).fillna(0.0)
    stage_stats = (
        qualified_df
        .groupby("team_name")
        .agg(
            wc_appearances = ("tournament_id", "count"),
            wc_stage_score = ("stage_score",   "mean"),
        )
        .reset_index()
    )

    result = match_stats.merge(stage_stats, on="team_name", how="outer")
    result["wc_appearances"] = result["wc_appearances"].fillna(0).astype(int)
    result["wc_stage_score"] = result["wc_stage_score"].fillna(0.0)
    result["wc_attack_rate"] = result["wc_attack_rate"].fillna(result["wc_attack_rate"].median())
    result["wc_defence_rate"]= result["wc_defence_rate"].fillna(result["wc_defence_rate"].median())
    result["wc_win_rate"]    = result["wc_win_rate"].fillna(result["wc_win_rate"].median())

    out = result[["team_name","wc_attack_rate","wc_defence_rate",
                  "wc_win_rate","wc_stage_score","wc_appearances"]]
    out.to_parquet(WC_TEAM_STATS_PARQUET, index=False)
    print(f"  ✓ WC heritage stats  : {len(out)} teams (downloaded)")
    return out


# Lookup helper used in engineer_features and build_team_profile
_WC_STATS_CACHE: Optional[pd.DataFrame] = None

def _wc_stats_for(team: str, wc_stats_df: pd.DataFrame) -> Dict[str, float]:
    """Return WC heritage stats dict for a team, with sensible defaults."""
    jf_name = _jf_team_name(team)
    row = wc_stats_df[wc_stats_df["team_name"] == jf_name]
    if row.empty:
        # Team has never appeared at a World Cup — use league-wide medians
        return {
            "wc_attack_rate":  float(wc_stats_df["wc_attack_rate"].median())  if len(wc_stats_df) else 1.2,
            "wc_defence_rate": float(wc_stats_df["wc_defence_rate"].median()) if len(wc_stats_df) else 1.2,
            "wc_win_rate":     float(wc_stats_df["wc_win_rate"].median())     if len(wc_stats_df) else 0.33,
            "wc_stage_score":  0.0,
            "wc_appearances":  0,
        }
    r = row.iloc[0]
    return {
        "wc_attack_rate":  float(r["wc_attack_rate"]),
        "wc_defence_rate": float(r["wc_defence_rate"]),
        "wc_win_rate":     float(r["wc_win_rate"]),
        "wc_stage_score":  float(r["wc_stage_score"]),
        "wc_appearances":  int(r["wc_appearances"]),
    }


def build_dataset() -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Assemble the training dataset using persistent local cache.

    Cold start  : downloads Kaggle base + ELO + WC 2026 ESPN + jfjelstul; saves to disk.
    Warm start  : loads parquet files; only calls ESPN for new WC 2026 rows.

    Returns (training_df, elo_by_name, wc2026_results_df, wc_team_stats_df).
    """
    print("\n[DATA INGESTION]")

    # ── ELO (cached ≤24 h) ────────────────────────────────────────────────────
    elo_by_name = load_elo_cached()
    for alias, canonical in _TEAM_NAME_ALIASES.items():
        if canonical in elo_by_name:
            elo_by_name[alias] = elo_by_name[canonical]

    _unmapped = sorted(set(ESPN_TEAM_IDS) - set(elo_by_name) - {"Turkey"})
    if _unmapped:
        print(f"  ⚠ No ELO for: {', '.join(_unmapped)} (will default to elo_min)")

    # ── jfjelstul WC heritage stats (permanent parquet) ───────────────────────
    wc_team_stats = load_wc_team_stats()

    # ── Historical base (permanent parquet) ───────────────────────────────────
    hist = load_historical_base()

    # ── WC 2026 live results (incremental ESPN append) ────────────────────────
    wc2026_results = load_wc2026_cached()

    # ── Combine ───────────────────────────────────────────────────────────────
    frames = [hist]
    if not wc2026_results.empty:
        frames.append(wc2026_results[["date","home_team","away_team",
                                      "home_score","away_score",
                                      "tournament","neutral_venue"]])

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True).dt.tz_localize(None)
    combined = (combined
                .sort_values("date")
                .drop_duplicates(subset=["date", "home_team", "away_team"])
                .reset_index(drop=True))

    print(f"\n  Total training matches : {len(combined):,}")
    print(f"  Date range             : {combined['date'].min().date()} → "
          f"{combined['date'].max().date()}")
    print(f"  Unique teams           : "
          f"{len(set(combined['home_team']) | set(combined['away_team']))}")
    print(f"  ELO ratings            : {len(elo_by_name)} teams")
    print(f"  WC heritage teams      : {len(wc_team_stats)} teams\n")

    return combined, elo_by_name, wc2026_results, wc_team_stats


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1B: WC 2026 PLAYER STATS  —  ESPN match-level, incrementally cached
# ─────────────────────────────────────────────────────────────────────────────

_STAT_KEYS = [
    "totalGoals", "goalAssists", "totalShots", "shotsOnTarget",
    "yellowCards", "redCards", "foulsCommitted", "saves", "goalsConceded",
    "ownGoals",
]


def _fetch_player_stats_for_event(event_id: str) -> List[Dict]:
    """
    Fetch roster-level player stats from one completed ESPN match summary.
    Returns a flat list of dicts — one row per player-appearance.
    """
    url = ESPN_SUMMARY_URL.format(event_id=event_id)
    try:
        r = _get(url)
        sd = r.json()
    except Exception:
        return []

    # Pull match date from header
    header = sd.get("header", {})
    competitions = header.get("competitions", [{}])
    match_date = competitions[0].get("date", "") if competitions else ""

    rosters = sd.get("rosters", [])
    rows = []
    for team_entry in rosters:
        team_name = team_entry.get("team", {}).get("displayName", "Unknown")
        for player in team_entry.get("roster", []):
            athlete = player.get("athlete", {})
            stat_map = {s["name"]: s["value"] for s in player.get("stats", [])
                        if "name" in s and "value" in s}
            rows.append({
                "event_id":    event_id,
                "match_date":  match_date,
                "team":        team_name,
                "player_name": athlete.get("displayName", "Unknown"),
                "espn_id":     athlete.get("id", ""),
                "starter":     player.get("starter", False),
                **{k: float(stat_map.get(k, 0.0)) for k in _STAT_KEYS},
            })
    return rows


def fetch_wc2026_player_stats_incremental(seen_event_ids: set) -> Tuple[List[Dict], set]:
    """
    Fetch player stats for all completed WC 2026 events not yet in the cache.
    Returns (new_rows, updated_seen_ids).
    """
    # Get all completed event IDs from the scoreboard (multi-date scan)
    completed_events: List[Tuple[str, str, str]] = []  # (id, home, away)

    # Scan each match date ESPN knows about
    try:
        r = _get(ESPN_SCHEDULE_URL)
        events = r.json().get("events", [])
        for ev in events:
            status = ev.get("status", {}).get("type", {}).get("name", "")
            if status in ("STATUS_FULL_TIME", "STATUS_FINAL", "STATUS_FT",
                          "STATUS_END_PERIOD", "STATUS_SHOOTOUT"):
                eid = str(ev["id"])
                if eid not in seen_event_ids:
                    comps = ev.get("competitions", [{}])[0]
                    teams = [c["team"]["displayName"]
                             for c in comps.get("competitors", [])]
                    home = teams[0] if len(teams) > 0 else "?"
                    away = teams[1] if len(teams) > 1 else "?"
                    completed_events.append((eid, home, away))
    except Exception as e:
        print(f"  ⚠ Could not fetch schedule for player stats: {e}")
        return [], seen_event_ids

    if not completed_events:
        return [], seen_event_ids

    new_rows: List[Dict] = []
    for eid, home, away in completed_events:
        rows = _fetch_player_stats_for_event(eid)
        if rows:
            new_rows.extend(rows)
            seen_event_ids.add(eid)
            print(f"    → {home} vs {away} ({len(rows)} player rows)")
        time.sleep(0.3)  # polite rate limit

    return new_rows, seen_event_ids


def load_player_stats_cached() -> pd.DataFrame:
    """
    Load WC 2026 player stats from disk, appending any new completed matches.
    Returns a DataFrame with one row per player-appearance.
    """
    cached = pd.DataFrame()
    seen_ids: set = set()

    if PLAYER_STATS_PARQUET.exists():
        cached = pd.read_parquet(PLAYER_STATS_PARQUET)
        seen_ids = set(cached["event_id"].astype(str).unique())

    print(f"  Checking ESPN for new player stats ({len(seen_ids)} events cached)...")
    new_rows, seen_ids = fetch_wc2026_player_stats_incremental(seen_ids)

    if not new_rows:
        if cached.empty:
            print("    → No player stats available yet.")
        else:
            print(f"    → No new events ({len(cached)} rows cached).")
        return cached

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([cached, new_df], ignore_index=True) if not cached.empty else new_df
    combined.to_parquet(PLAYER_STATS_PARQUET, index=False)
    print(f"    → {len(new_rows)} new rows added (total {len(combined)}).")
    return combined


def compute_team_form(team: str, stats_df: pd.DataFrame) -> Dict:
    """
    Aggregate per-player WC 2026 stats into team-level tournament form metrics.

    Returns a dict with:
      games_played, goals_pg, shots_pg, shots_on_target_pg,
      goals_conceded_pg, saves_pg, yellow_cards_pg, red_cards_pg,
      top_scorers (list), suspended (list of players with red cards)
    """
    empty = {
        "games_played": 0,
        "goals_pg": 0.0, "shots_pg": 0.0, "shots_on_target_pg": 0.0,
        "goals_conceded_pg": 0.0, "saves_pg": 0.0,
        "yellow_cards_pg": 0.0, "red_cards_pg": 0.0,
        "top_scorers": [], "suspended": [],
    }
    if stats_df.empty:
        return empty

    team_df = stats_df[stats_df["team"] == team]
    if team_df.empty:
        return empty

    events = team_df["event_id"].nunique()
    if events == 0:
        return empty

    totals = team_df[_STAT_KEYS].sum()

    # Top scorers: player name → total goals (only players with ≥1 goal)
    scorer_df = team_df.groupby("player_name")["totalGoals"].sum()
    top_scorers = (scorer_df[scorer_df > 0]
                   .sort_values(ascending=False)
                   .head(5)
                   .reset_index()
                   .rename(columns={"totalGoals": "goals"})
                   .to_dict("records"))

    # Suspended: players who have accumulated a red card
    red_card_df = team_df.groupby("player_name")["redCards"].sum()
    suspended = list(red_card_df[red_card_df >= 1].index)

    return {
        "games_played":         events,
        "goals_pg":             totals["totalGoals"] / events,
        "shots_pg":             totals["totalShots"] / events,
        "shots_on_target_pg":   totals["shotsOnTarget"] / events,
        "goals_conceded_pg":    totals["goalsConceded"] / events / 11,  # per-GK → team level
        "saves_pg":             totals["saves"] / events,
        "yellow_cards_pg":      totals["yellowCards"] / events,
        "red_cards_pg":         totals["redCards"] / events,
        "top_scorers":          top_scorers,
        "suspended":            suspended,
    }


def tournament_form_lambda_adjustment(
    form: Dict,
    tournament_avg_goals_pg: float = 1.3,
    max_games_for_full_weight: int = 3,
) -> Tuple[float, float]:
    """
    Compute (attack_multiplier, defence_multiplier) based on WC 2026 form.

    Attack : team's goals_pg vs tournament average.
    Defence: team's goals_conceded_pg vs tournament average.
    Weight ramps from 0 (no games) to 0.35 (≥3 games) so early results
    don't dominate the historical Poisson λ.

    Returns multipliers where 1.0 = no adjustment.
    """
    n = form.get("games_played", 0)
    if n == 0:
        return 1.0, 1.0

    weight = min(n / max_games_for_full_weight, 1.0) * 0.35

    goals_pg   = form.get("goals_pg", tournament_avg_goals_pg)
    conceded_pg = form.get("goals_conceded_pg", tournament_avg_goals_pg)

    # Ratio of actual vs expected, capped to avoid extreme swings
    atk_ratio = np.clip(goals_pg / max(tournament_avg_goals_pg, 0.01), 0.5, 2.5)
    def_ratio = np.clip(tournament_avg_goals_pg / max(conceded_pg, 0.01), 0.5, 2.5)

    atk_mult = 1.0 + weight * (atk_ratio - 1.0)
    def_mult = 1.0 + weight * (def_ratio - 1.0)

    return float(atk_mult), float(def_mult)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1C: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    elo_by_name: Dict[str, float],
    wc_team_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Transform the raw match DataFrame into a feature-rich training set.

    Features engineered
    -------------------
    • tournament_weight    : credibility weight per competition
    • host_advantage       : 1 if home_team is a WC 2026 host nation
    • neutral_venue        : 1 if match played at a neutral ground
    • home/away_attack     : time-weighted rolling goals scored, ELO-adjusted
    • home/away_defence    : time-weighted rolling goals conceded, ELO-adjusted
    • home/away_elo_norm   : normalised ELO rating (0–1 scale)
    • home/away_uncertainty: data-sparsity metric — drives CI width in MC step
    • home/away_wc_attack  : historical WC goals/game (jfjelstul/worldcup)
    • home/away_wc_defence : historical WC goals conceded/game
    • home/away_wc_stage   : avg best stage score across WC appearances (0–5)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["tournament_weight"] = (
        df["tournament"].map(TOURNAMENT_WEIGHTS).fillna(0.60)
    )
    df["host_advantage"] = df["home_team"].isin(WC2026_HOSTS).astype(int)
    df["neutral_venue"]  = df["neutral_venue"].astype(int)

    # ELO normalisation: global range defines [0, 1]
    all_elos  = np.array(list(elo_by_name.values()))
    elo_min, elo_max = all_elos.min(), all_elos.max()

    def elo_norm(team: str) -> float:
        raw = elo_by_name.get(team, elo_min)
        return (raw - elo_min) / (elo_max - elo_min + 1e-9)

    DECAY   = 0.85
    WINDOW  = 10

    def recent_form(team: str, before: pd.Timestamp) -> Dict:
        mask   = (
            ((df["home_team"] == team) | (df["away_team"] == team)) &
            (df["date"] < before)
        )
        recent = df[mask].tail(WINDOW).copy()
        n      = len(recent)

        if n == 0:
            return {"scored": 1.2, "conceded": 1.2, "n": 0,
                    "quality": 0.0, "opp_elo": 0.0}

        w = np.array([DECAY ** (n - 1 - i) for i in range(n)])
        w /= w.sum()

        scored   = np.where(recent["home_team"] == team,
                            recent["home_score"], recent["away_score"]).astype(float)
        conceded = np.where(recent["home_team"] == team,
                            recent["away_score"], recent["home_score"]).astype(float)
        t_wts    = recent["tournament_weight"].values.clip(min=0.4)

        # Divide by tournament weight so elite-competition goals weight more
        adj_s = np.average(scored   / t_wts, weights=w)
        adj_c = np.average(conceded / t_wts, weights=w)

        # Data-quality signals for the uncertainty metric:
        #   quality  — how much elite competition is in the window
        #              (10 WC matches → 1.0; 10 friendlies → 0.4; 3 matches → ≤0.3)
        #   opp_elo  — average strength of the opponents actually faced
        opponents = np.where(recent["home_team"] == team,
                             recent["away_team"], recent["home_team"])
        quality   = float(recent["tournament_weight"].sum() / WINDOW)
        opp_elo   = float(np.average([elo_norm(o) for o in opponents], weights=w))

        return {
            "scored":   max(adj_s, 0.2),
            "conceded": max(adj_c, 0.2),
            "n":        n,
            "quality":  quality,
            "opp_elo":  opp_elo,
        }

    home_att, home_def, home_n = [], [], []
    away_att, away_def, away_n = [], [], []
    home_unc, away_unc = [], []

    def _uncertainty(form: Dict) -> float:
        """
        Epistemic uncertainty about a team's true strength.

        Blends two signals, both in [0, 1]:
          quality  — volume × seriousness of recent matches
          opp_elo  — calibre of opposition actually faced

        A team with 10 recent WC/Euro games vs elite sides → ~0.05–0.15.
        A team playing mostly friendlies vs weak sides     → ~0.40–0.60.
        A near-debutant with a thin record                  → ~0.70+.

        (The previous formula collapsed to its 0.05 floor for any team
        with a full 10-match window, making the metric inert.)
        """
        u = 1.0 - 0.55 * form["quality"] - 0.45 * form["opp_elo"]
        return float(np.clip(u, 0.05, 0.95))

    for _, row in df.iterrows():
        hf = recent_form(row["home_team"], row["date"])
        af = recent_form(row["away_team"], row["date"])
        home_att.append(hf["scored"]);   home_def.append(hf["conceded"]); home_n.append(hf["n"])
        away_att.append(af["scored"]);   away_def.append(af["conceded"]); away_n.append(af["n"])
        home_unc.append(_uncertainty(hf))
        away_unc.append(_uncertainty(af))

    df["home_attack"]  = home_att;  df["home_defence"] = home_def;  df["home_n"] = home_n
    df["away_attack"]  = away_att;  df["away_defence"] = away_def;  df["away_n"] = away_n

    df["home_elo_norm"] = df["home_team"].map(elo_norm)
    df["away_elo_norm"] = df["away_team"].map(elo_norm)

    df["home_uncertainty"] = home_unc
    df["away_uncertainty"] = away_unc

    # ── jfjelstul WC heritage features ───────────────────────────────────────
    if wc_team_stats is not None and not wc_team_stats.empty:
        def _wc_att(team):   return _wc_stats_for(team, wc_team_stats)["wc_attack_rate"]
        def _wc_def(team):   return _wc_stats_for(team, wc_team_stats)["wc_defence_rate"]
        def _wc_stage(team): return _wc_stats_for(team, wc_team_stats)["wc_stage_score"]

        df["home_wc_attack"]  = df["home_team"].map(_wc_att)
        df["away_wc_attack"]  = df["away_team"].map(_wc_att)
        df["home_wc_defence"] = df["home_team"].map(_wc_def)
        df["away_wc_defence"] = df["away_team"].map(_wc_def)
        df["home_wc_stage"]   = df["home_team"].map(_wc_stage)
        df["away_wc_stage"]   = df["away_team"].map(_wc_stage)
    else:
        for col in ["home_wc_attack","away_wc_attack","home_wc_defence",
                    "away_wc_defence","home_wc_stage","away_wc_stage"]:
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: PREDICTION ENGINE — POISSON REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class PoissonMatchPredictor:
    """
    Two independent Poisson GLMs:

        Home goals ~ home_attack + away_defence + home_elo_norm
                   + away_elo_norm + host_advantage + neutral_venue
                   + tournament_weight

        Away goals ~ away_attack + home_defence + home_elo_norm
                   + away_elo_norm + neutral_venue + tournament_weight

    ELO normalised scores are added as covariates so the model has a
    quality-adjusted baseline even for teams with limited recent data.
    """

    HOME_FORMULA = (
        "home_score ~ home_attack + away_defence + home_elo_norm + "
        "away_elo_norm + host_advantage + neutral_venue + tournament_weight + "
        "home_wc_attack + away_wc_defence + home_wc_stage"
    )
    AWAY_FORMULA = (
        "away_score ~ away_attack + home_defence + home_elo_norm + "
        "away_elo_norm + neutral_venue + tournament_weight + "
        "away_wc_attack + home_wc_defence + away_wc_stage"
    )

    def __init__(self):
        self.home_model = None
        self.away_model = None
        self._fitted    = False

    def fit(self, df: pd.DataFrame) -> "PoissonMatchPredictor":
        print("[MODEL FITTING]")
        print("  Fitting home-goals Poisson GLM ...")
        self.home_model = smf.glm(
            self.HOME_FORMULA, data=df, family=sm.families.Poisson()
        ).fit(disp=False)

        print("  Fitting away-goals Poisson GLM ...")
        self.away_model = smf.glm(
            self.AWAY_FORMULA, data=df, family=sm.families.Poisson()
        ).fit(disp=False)

        self._fitted = True
        print(f"  Home model AIC: {self.home_model.aic:.1f} "
              f"| Away model AIC: {self.away_model.aic:.1f}\n")
        return self

    def predict_lambdas(
        self,
        home_feats: Dict,
        away_feats: Dict,
    ) -> Tuple[float, float]:
        """
        Predict λ_A and λ_B for a match.

        Neutral-venue symmetry fix
        --------------------------
        The two GLMs have different intercepts (away teams historically
        score less than home teams). On a non-neutral pitch this is correct.
        On a neutral ground — including all WC 2026 group-stage matches —
        the result should be order-invariant: swapping Team A and Team B
        must only swap the two λ values, not change their magnitudes.

        We achieve this by running the prediction TWICE (A-as-home,
        B-as-away) and (B-as-home, A-as-away) then averaging:
            λ_A = (home_model(A,B) + away_model(A,B-swapped)) / 2
            λ_B = (away_model(B,A) + home_model(B,A-swapped)) / 2

        For non-neutral matches the standard single-pass is used,
        preserving the home-advantage signal in the GLM intercept.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

        is_neutral = bool(home_feats.get("neutral_venue", 0))
        t_wt       = home_feats.get("tournament_weight", 0.75)

        def _lh(hf, af, host_flag):
            """Home-goals model: team hf attacking, team af defending."""
            row = pd.DataFrame([{
                "home_attack":       hf["attack"],
                "away_defence":      af["defence"],
                "home_elo_norm":     hf.get("elo_norm", 0.5),
                "away_elo_norm":     af.get("elo_norm", 0.5),
                "host_advantage":    host_flag,
                "neutral_venue":     int(is_neutral),
                "tournament_weight": t_wt,
                "home_wc_attack":    hf.get("wc_attack", 1.2),
                "away_wc_defence":   af.get("wc_defence", 1.2),
                "home_wc_stage":     hf.get("wc_stage", 0.0),
            }])
            return max(float(self.home_model.predict(row).iloc[0]), 0.05)

        def _la(af, hf):
            """Away-goals model: team af attacking, team hf defending."""
            row = pd.DataFrame([{
                "away_attack":       af["attack"],
                "home_defence":      hf["defence"],
                "home_elo_norm":     hf.get("elo_norm", 0.5),
                "away_elo_norm":     af.get("elo_norm", 0.5),
                "neutral_venue":     int(is_neutral),
                "tournament_weight": t_wt,
                "away_wc_attack":    af.get("wc_attack", 1.2),
                "home_wc_defence":   hf.get("wc_defence", 1.2),
                "away_wc_stage":     af.get("wc_stage", 0.0),
            }])
            return max(float(self.away_model.predict(row).iloc[0]), 0.05)

        # Host-advantage multiplier.
        # NOTE: we deliberately do NOT use the GLM's fitted host_advantage
        # coefficient here. The training flag marks every historical match
        # where USA/Canada/Mexico was the home side (90 years of data) — a
        # team-identity indicator, not a "hosting this tournament" signal —
        # and its fitted value is mildly NEGATIVE. The flag stays in the
        # training formula so it absorbs that identity confound away from
        # the other covariates, but for prediction we apply a fixed boost
        # in line with the literature on host advantage in international
        # tournaments (~ +15–25% goals for the host playing at home).
        host_mult = HOST_BOOST

        if is_neutral:
            # Average both orderings so result is symmetric on neutral ground.
            # host_advantage is excluded from the GLM passes (flag = 0); the
            # host boost is applied once, in full, after averaging — putting
            # it inside only one pass would let the geometric mean dilute it.
            # Pass A  (A as home, B as away)
            lh_ab = _lh(home_feats, away_feats, 0)
            la_ab = _la(away_feats, home_feats)
            # Pass B  (B as home, A as away) — roles fully swapped
            lh_ba = _lh(away_feats, home_feats, 0)   # B's attack vs A's defence
            la_ba = _la(home_feats, away_feats)      # A's attack vs B's defence

            # λ_A = geometric mean of A-as-home and A-as-away estimates
            lh = float(np.sqrt(lh_ab * la_ba))
            la = float(np.sqrt(la_ab * lh_ba))

            # Full host boost for whichever side(s) are hosts
            if home_feats.get("host_advantage", 0):
                lh *= host_mult
            if away_feats.get("host_advantage", 0):
                la *= host_mult
        else:
            lh = _lh(home_feats, away_feats, 0)
            la = _la(away_feats, home_feats)
            if home_feats.get("host_advantage", 0):
                lh *= host_mult
            if away_feats.get("host_advantage", 0):
                la *= host_mult

        return lh, la


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2.5: BETTING MARKET ODDS  —  ESPN / DraftKings
# ─────────────────────────────────────────────────────────────────────────────
#
# Rationale
# ---------
# Betting markets aggregate sharp money and real-time information the statistical
# model cannot see (confirmed line-ups, weather, late injury news, momentum). The
# market's implied probabilities are, on average, better calibrated than a pure
# Poisson model. We therefore:
#   1. Pull the 3-way moneyline (home / draw / away) and the over/under total.
#   2. Convert American odds → implied probabilities, then remove the bookmaker
#      "vig" (overround) to get fair probabilities.
#   3. Invert those fair probabilities into a MARKET-IMPLIED λ pair, so the whole
#      downstream Monte-Carlo machinery (scorelines, CI, margin of error) stays
#      coherent.
#   4. Blend market λ with model λ via a configurable weight.
# ─────────────────────────────────────────────────────────────────────────────


def american_to_prob(moneyline: float) -> float:
    """Convert American moneyline odds to implied probability (incl. vig)."""
    if moneyline is None:
        return 0.0
    ml = float(moneyline)
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    return (-ml) / ((-ml) + 100.0)


def devig_three_way(p_home: float, p_draw: float, p_away: float
                    ) -> Tuple[float, float, float]:
    """
    Remove bookmaker overround from raw implied probabilities using the
    proportional (normalisation) method. Returns fair probabilities summing to 1.
    """
    total = p_home + p_draw + p_away
    if total <= 0:
        return (0.0, 0.0, 0.0)
    return (p_home / total, p_draw / total, p_away / total)


# Dixon-Coles low-score correction parameter.
# Independent Poissons systematically under-predict draws in low-scoring
# football: real matches show positive dependence at 0-0/1-1 (teams shut up
# shop) and the market prices this in. ρ = −0.10 is in the range estimated
# by Dixon & Coles (1997) for international/league football.
DC_RHO = -0.10


def _dc_tau_grid(lh: float, la: float, max_goals: int, rho: float = DC_RHO
                 ) -> np.ndarray:
    """
    Dixon-Coles adjustment factors τ(i, j) on a joint score grid.
    Only the four low-score cells deviate from 1:
        τ(0,0) = 1 − λμρ     τ(0,1) = 1 + λρ
        τ(1,0) = 1 + μρ      τ(1,1) = 1 − ρ
    With ρ < 0 this boosts 0-0 and 1-1 (draws) and trims 1-0 / 0-1.
    """
    tau = np.ones((max_goals + 1, max_goals + 1))
    tau[0, 0] = max(1.0 - lh * la * rho, 0.0)
    tau[0, 1] = max(1.0 + lh * rho, 0.0)
    tau[1, 0] = max(1.0 + la * rho, 0.0)
    tau[1, 1] = max(1.0 - rho, 0.0)
    return tau


def _skellam_wdl(lh: float, la: float, max_goals: int = 12
                ) -> Tuple[float, float, float]:
    """
    Exact W/D/L probabilities for two Poisson scorers with the Dixon-Coles
    low-score correction. Using the same correction here and in the Monte
    Carlo simulator keeps the market-implied λ inversion consistent with
    the simulated outcomes.
    """
    from scipy.stats import poisson
    gh = poisson.pmf(np.arange(max_goals + 1), lh)
    ga = poisson.pmf(np.arange(max_goals + 1), la)
    joint = np.outer(gh, ga) * _dc_tau_grid(lh, la, max_goals)
    joint /= joint.sum()                          # renormalise after τ
    p_home = np.tril(joint, -1).sum()             # i > j
    p_draw = np.trace(joint)                      # i == j
    p_away = np.triu(joint, 1).sum()              # i < j
    return float(p_home), float(p_draw), float(p_away)


def total_from_over_under(line: float, p_over: float,
                          search=(0.3, 4.5, 0.01)) -> float:
    """
    Invert the over/under market into an expected total-goals μ.
    Finds μ such that P(total goals > line) under Poisson(μ) ≈ p_over.
    """
    from scipy.stats import poisson
    lo, hi, step = search
    threshold = int(np.floor(line)) + 1           # e.g. line 2.5 → P(X >= 3)
    best_mu, best_err = (lo + hi) / 2, 1e9
    mu = lo
    while mu <= hi:
        p = 1.0 - poisson.cdf(threshold - 1, mu)
        err = abs(p - p_over)
        if err < best_err:
            best_err, best_mu = err, mu
        mu += step
    return best_mu


def market_implied_lambdas(
    p_home: float, p_draw: float, p_away: float,
    total_mu: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Solve for the (λ_home, λ_away) pair whose Skellam W/D/L probabilities best
    match the de-vigged market probabilities.

    If total_mu (from the over/under market) is supplied it is used both to
    centre the search and as a soft constraint on λ_home + λ_away — the
    over/under tells us the expected total, the moneyline tells us the split.
    """
    # Search grid for the goal difference (split) and total
    mu_centre = total_mu if total_mu else 2.6
    mu_grid   = ([mu_centre] if total_mu
                 else np.arange(1.4, 3.8, 0.1))
    delta_grid = np.arange(-2.6, 2.6, 0.05)       # λ_home − λ_away

    best, best_err = (mu_centre / 2, mu_centre / 2), 1e9
    for mu in np.atleast_1d(mu_grid):
        for delta in delta_grid:
            lh = (mu + delta) / 2.0
            la = (mu - delta) / 2.0
            if lh < 0.05 or la < 0.05:
                continue
            ph, pd_, pa = _skellam_wdl(lh, la)
            err = (ph - p_home) ** 2 + (pd_ - p_draw) ** 2 + (pa - p_away) ** 2
            if err < best_err:
                best_err, best = err, (lh, la)
    return float(best[0]), float(best[1])


def fetch_match_odds(home_team: str, away_team: str) -> Optional[Dict]:
    """
    Fetch the live betting market for a specific matchup from ESPN/DraftKings.

    Looks up the event_id from the WC 2026 schedule (matching by team names in
    either order), then pulls the odds block from the match summary endpoint.

    Returns a dict with raw odds, de-vigged probabilities, and the market-implied
    λ pair — or None if no market is available (e.g. an unscheduled hypothetical).
    """
    fixtures = fetch_wc2026_fixtures()
    if fixtures.empty:
        return None

    # Match the schedule row by team names, tolerant of alternate spellings
    # (either orientation), via the central COUNTRY_ALIASES registry.
    def _is_this_fixture(r) -> bool:
        a, b = r["home_team"], r["away_team"]
        return ((_names_match(home_team, a) and _names_match(away_team, b)) or
                (_names_match(home_team, b) and _names_match(away_team, a)))

    match = fixtures[fixtures.apply(_is_this_fixture, axis=1)]
    if match.empty:
        return None

    event_id    = match.iloc[0]["event_id"]
    # The schedule orientation may differ from the user's A/B choice.
    sched_home  = match.iloc[0]["home_team"]

    try:
        r = _get(ESPN_SUMMARY_URL.format(event_id=event_id))
        data = r.json()
    except Exception:
        return None

    odds_list = data.get("odds") or data.get("pickcenter") or []
    if not odds_list:
        return None
    o = odds_list[0]

    ml_home = (o.get("homeTeamOdds") or {}).get("moneyLine")
    ml_away = (o.get("awayTeamOdds") or {}).get("moneyLine")
    ml_draw = (o.get("drawOdds") or {}).get("moneyLine")
    if ml_home is None or ml_away is None or ml_draw is None:
        return None

    # Raw implied probs (schedule orientation: home = sched_home)
    raw_h = american_to_prob(ml_home)
    raw_d = american_to_prob(ml_draw)
    raw_a = american_to_prob(ml_away)
    fair_h, fair_d, fair_a = devig_three_way(raw_h, raw_d, raw_a)

    # Over/under → expected total goals
    line   = o.get("overUnder")
    over_o = o.get("overOdds")
    under_o = o.get("underOdds")
    total_mu = None
    if line is not None and over_o is not None and under_o is not None:
        p_over_raw  = american_to_prob(over_o)
        p_under_raw = american_to_prob(under_o)
        tot = p_over_raw + p_under_raw
        if tot > 0:
            p_over = p_over_raw / tot
            total_mu = total_from_over_under(float(line), p_over)

    # Market-implied λ in SCHEDULE orientation
    lh_sched, la_sched = market_implied_lambdas(fair_h, fair_d, fair_a, total_mu)

    # Re-orient to the user's A/B choice
    if sched_home == home_team:
        lam_home, lam_away = lh_sched, la_sched
        prob_home, prob_draw, prob_away = fair_h, fair_d, fair_a
        ml_a, ml_b = ml_home, ml_away
    else:
        lam_home, lam_away = la_sched, lh_sched
        prob_home, prob_draw, prob_away = fair_a, fair_d, fair_h
        ml_a, ml_b = ml_away, ml_home

    return {
        "provider":      (o.get("provider") or {}).get("name", "Market"),
        "moneyline_home": ml_a,
        "moneyline_draw": ml_draw,
        "moneyline_away": ml_b,
        "over_under":     line,
        "market_prob_home": round(prob_home, 4),
        "market_prob_draw": round(prob_draw, 4),
        "market_prob_away": round(prob_away, 4),
        "market_total_goals": round(total_mu, 3) if total_mu else None,
        "market_lambda_home": round(lam_home, 3),
        "market_lambda_away": round(lam_away, 3),
        "details":        o.get("details", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2.6: POLYMARKET TOURNAMENT WINNER ODDS
# ─────────────────────────────────────────────────────────────────────────────
# Polymarket runs binary "Will X win the 2026 FIFA World Cup?" markets.
# The Yes price in USDC ≈ probability (e.g. 0.1675 → 16.75% chance to win WC).
# We fetch these as crowd-intelligence context alongside per-match odds.
# ─────────────────────────────────────────────────────────────────────────────

_POLYMARKET_API = "https://gamma-api.polymarket.com/markets?tag=soccer&limit=200"
_POLY_WC_CACHE: Dict[str, Any] = {}          # {team_name: {prob, slug, updated}}
_POLY_WC_LOADED_AT: float = 0.0
_POLY_WC_TTL: float = 300.0                  # refresh every 5 min

# Map predictor team names → fragments that appear in Polymarket question text.
_POLY_TEAM_ALIASES: Dict[str, str] = {
    "USA":          "Will USA win",
    "United States":"Will USA win",
    "South Korea":  "Will South Korea win",
    "Ivory Coast":  "Will Ivory Coast win",
    "Cote d'Ivoire":"Will Ivory Coast win",
    "Netherlands":  "Will Netherlands win",
    "Holland":      "Will Netherlands win",
}


def _poly_team_key(team: str) -> str:
    """Build the Polymarket question fragment to search for a given team."""
    return _POLY_TEAM_ALIASES.get(team, f"Will {team} win")


def fetch_polymarket_wc_odds_all() -> Dict[str, float]:
    """
    Fetch all 2026 FIFA World Cup winner markets from Polymarket.

    Returns {team_name: yes_prob} for every team found.
    Caches results for _POLY_WC_TTL seconds.
    """
    global _POLY_WC_CACHE, _POLY_WC_LOADED_AT
    import time as _time
    now = _time.time()
    if _POLY_WC_CACHE and (now - _POLY_WC_LOADED_AT) < _POLY_WC_TTL:
        return _POLY_WC_CACHE

    result: Dict[str, float] = {}
    try:
        r = _get(_POLYMARKET_API)
        markets = r.json()
        for m in markets:
            q: str = m.get("question", "")
            if "2026 FIFA World Cup" not in q:
                continue
            prices_raw = m.get("outcomePrices", "[]")
            outcomes_raw = m.get("outcomes", "[]")
            try:
                prices   = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            except Exception:
                continue
            if len(outcomes) != 2 or len(prices) != 2:
                continue
            try:
                yes_idx = outcomes.index("Yes")
                yes_prob = float(prices[yes_idx])
            except (ValueError, IndexError):
                continue
            # Extract team name from "Will X win the 2026 FIFA World Cup?"
            team = q.replace("Will ", "").replace(" win the 2026 FIFA World Cup?", "").strip()
            result[team] = yes_prob

    except Exception:
        pass

    _POLY_WC_CACHE = result
    _POLY_WC_LOADED_AT = now
    return result


def fetch_polymarket_team_odds(team: str) -> Optional[float]:
    """
    Return the Polymarket implied probability that `team` wins WC 2026.
    Returns None if the team has no active Polymarket market.
    """
    all_odds = fetch_polymarket_wc_odds_all()
    if not all_odds:
        return None
    # Direct match first
    if team in all_odds:
        return all_odds[team]
    # Alias match
    key_frag = _poly_team_key(team)
    for poly_team, prob in all_odds.items():
        if key_frag.lower() in f"will {poly_team.lower()} win":
            return prob
    # Fuzzy: team name substring
    tl = team.lower()
    for poly_team, prob in all_odds.items():
        if tl in poly_team.lower() or poly_team.lower() in tl:
            return prob
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2.7: POLYMARKET PER-MATCH ODDS (optional, discovered dynamically)
# ─────────────────────────────────────────────────────────────────────────────
# Polymarket lists individual soccer matches as an *event* containing three
# binary markets:
#       "Will {A} beat {B}?"            → Yes price ≈ P(A wins)
#       "Will {B} beat {A}?"            → Yes price ≈ P(B wins)
#       "Will {A} vs. {B} end in a draw?" → Yes price ≈ P(draw)
# We discover the event by team name (no URLs to pre-load), de-vig the three Yes
# prices, and invert them to a λ pair through the same Skellam machinery used for
# DraftKings. Match markets are created selectively, so this returns None for any
# fixture Polymarket has not listed (the common case for minor group games).
# ─────────────────────────────────────────────────────────────────────────────

_POLY_SEARCH_URL = "https://gamma-api.polymarket.com/public-search?q={q}&limit_per_type=20"
_POLY_EVENT_URL  = "https://gamma-api.polymarket.com/events?slug={slug}"
_POLY_MATCH_CACHE: Dict[frozenset, Any] = {}
_POLY_MATCH_TTL: float = 300.0               # 5-minute cache per fixture

def _poly_name_in(team: str, text: str) -> bool:
    """True if `team` (or any of its alternate spellings) appears in `text`.
    Uses the central COUNTRY_ALIASES registry, accent/case/punctuation-blind."""
    txt = _norm_name(text)
    return any(v and v in txt for v in country_name_variants(team))


def _poly_yes_price(market: Dict) -> Optional[float]:
    """Pull the 'Yes' outcome price out of a binary Polymarket market."""
    try:
        outs = market.get("outcomes", "[]")
        prs  = market.get("outcomePrices", "[]")
        outs = json.loads(outs) if isinstance(outs, str) else outs
        prices = json.loads(prs) if isinstance(prs, str) else prs
        return float(prices[outs.index("Yes")])
    except Exception:
        return None


def _poly_find_match_event(home: str, away: str):
    """Search Polymarket for an open soccer event between `home` and `away`.

    Returns (event_dict, markets_list) or (None, None).
    """
    queries = [f"{home} vs {away}", f"{away} vs {home}", f"{home} {away}"]
    for q in queries:
        try:
            r = _get(_POLY_SEARCH_URL.format(q=requests.utils.quote(q)))
            events = (r.json() or {}).get("events", [])
        except Exception:
            continue
        for e in events:
            title = e.get("title", "")
            if "vs" not in title.lower():
                continue
            if not (_poly_name_in(home, title) and _poly_name_in(away, title)):
                continue
            if e.get("closed"):                 # skip resolved matches
                continue
            markets = e.get("markets")
            if not markets and e.get("slug"):
                try:
                    fev = _get(_POLY_EVENT_URL.format(slug=e["slug"])).json()
                    markets = fev[0].get("markets") if fev else None
                except Exception:
                    markets = None
            if markets:
                return e, markets
    return None, None


def _poly_odds_from_markets(markets, home: str, away: str,
                            slug: Optional[str] = None) -> Optional[Dict]:
    """
    Turn a list of Polymarket binary markets (the three W/L/D questions for one
    match) into the de-vigged probabilities + market-implied λ pair, oriented to
    `home`/`away`. Returns None if the three questions can't be identified.
    """
    if not markets:
        return None
    p_home = p_away = p_draw = None
    for m in markets:
        ql = (m.get("question") or "").lower()
        yes = _poly_yes_price(m)
        if yes is None:
            continue
        if "draw" in ql:                                 # "... end in a draw?"
            p_draw = yes
            continue
        if "beat" not in ql and "win" not in ql:
            continue
        # Two question shapes seen in the wild:
        #   "Will {A} beat {B}?"        → winner is named before "beat"
        #   "Will {A} win on {date}?"   → only the winner is named
        side = ql.split("beat")[0] if "beat" in ql else ql
        if _poly_name_in(home, side):
            p_home = yes
        elif _poly_name_in(away, side):
            p_away = yes
    if None in (p_home, p_draw, p_away):
        return None
    s = p_home + p_draw + p_away
    if s <= 0 or max(p_home, p_draw, p_away) >= 0.999:    # resolved/degenerate
        return None
    fair_h, fair_d, fair_a = p_home / s, p_draw / s, p_away / s
    lam_h, lam_a = market_implied_lambdas(fair_h, fair_d, fair_a, None)
    return {
        "provider":           "Polymarket",
        "event_slug":         slug,
        "market_prob_home":   round(fair_h, 4),
        "market_prob_draw":   round(fair_d, 4),
        "market_prob_away":   round(fair_a, 4),
        "market_lambda_home": round(lam_h, 3),
        "market_lambda_away": round(lam_a, 3),
        "overround":          round(s, 4),                # >1 = bookmaker margin
    }


def fetch_polymarket_match_odds(home: str, away: str) -> Optional[Dict]:
    """
    Discover and return Polymarket's per-match odds for `home` vs `away`.

    Returns a dict shaped like fetch_match_odds() (de-vigged W/D/L probabilities
    plus a market-implied λ pair, oriented to the caller's home/away), or None
    if Polymarket has no open market for this fixture.
    """
    key = frozenset((home, away))
    now = time.time()
    cached = _POLY_MATCH_CACHE.get(key)
    if cached and (now - cached[0]) < _POLY_MATCH_TTL:
        return cached[1]

    ev, markets = _poly_find_match_event(home, away)
    result = _poly_odds_from_markets(markets, home, away,
                                     ev.get("slug") if ev else None)
    _POLY_MATCH_CACHE[key] = (now, result)
    return result


def _poly_parse_url(url: str):
    """Extract (event_slug, event_id) from a Polymarket / polym.trade URL."""
    from urllib.parse import urlparse, parse_qs
    try:
        u = urlparse(url.strip())
    except Exception:
        return None, None
    qs = parse_qs(u.query)
    slug = (qs.get("eventSlug") or qs.get("slug") or [None])[0]
    eid  = (qs.get("eventId") or qs.get("id") or [None])[0]
    if not slug and u.path:                              # /event/<slug>[/...]
        parts = [p for p in u.path.split("/") if p]
        if "event" in parts and parts.index("event") + 1 < len(parts):
            slug = parts[parts.index("event") + 1]
        elif parts:
            slug = parts[-1]
    return slug, eid


def fetch_polymarket_match_odds_from_url(url: str, home: str, away: str) -> Optional[Dict]:
    """
    Resolve per-match odds from a user-supplied Polymarket (or polym.trade) URL.

    Returns the same dict as fetch_polymarket_match_odds(), or None if the URL
    can't be resolved to a match market with the three W/L/D questions.
    """
    slug, eid = _poly_parse_url(url)
    if not slug and not eid:
        return None

    markets = None
    resolved_slug = slug
    try:
        if slug:
            fev = _get(_POLY_EVENT_URL.format(slug=slug)).json()
            if fev:
                markets = fev[0].get("markets")
                resolved_slug = fev[0].get("slug", slug)
        if not markets and eid:
            fev = _get(f"https://gamma-api.polymarket.com/events/{eid}").json()
            ev = fev if isinstance(fev, dict) else (fev[0] if fev else {})
            markets = ev.get("markets")
            resolved_slug = ev.get("slug", slug)
    except Exception:
        return None

    return _poly_odds_from_markets(markets, home, away, resolved_slug)


def blend_lambdas(
    model_lh: float, model_la: float,
    market_lh: float, market_la: float,
    market_weight: float = 0.5,
) -> Tuple[float, float]:
    """
    Blend model and market expected goals using a geometric mean weighted by
    `market_weight` ∈ [0, 1]. Geometric (not arithmetic) because λ is a
    multiplicative rate.

        market_weight = 0.0  → pure statistical model
        market_weight = 1.0  → pure market
        market_weight = 0.5  → equal blend
    """
    w = max(0.0, min(1.0, market_weight))
    lh = (model_lh ** (1 - w)) * (market_lh ** w)
    la = (model_la ** (1 - w)) * (market_la ** w)
    return float(lh), float(la)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: MONTE CARLO SIMULATION & MARGIN OF ERROR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_match(
    lambda_home: float,
    lambda_away: float,
    home_uncertainty: float = 0.10,
    away_uncertainty: float = 0.10,
    home_attack_adj: float = 1.0,
    home_defence_adj: float = 1.0,
    away_attack_adj: float = 1.0,
    away_defence_adj: float = 1.0,
    home_form_atk_adj: float = 1.0,
    home_form_def_adj: float = 1.0,
    away_form_atk_adj: float = 1.0,
    away_form_def_adj: float = 1.0,
    n_simulations: int = 10_000,
    ci_level: float = 0.95,
    seed: Optional[int] = 42,
) -> Dict:
    """
    Monte Carlo match simulation with optional squad/injury adjustments.

    Squad adjustment logic
    ----------------------
    Before the MC loop, λ values are scaled by injury multipliers:
        λ_home_adj = λ_home × home_attack_adj   (reduced if home attackers out)
        λ_away_adj = λ_away × away_attack_adj
    The opponent's defensive λ is also adjusted (missing defenders inflate
    the opponent's effective λ):
        λ_home_eff = λ_home_adj / away_defence_adj  (away missing defenders → easier to score)
        λ_away_eff = λ_away_adj / home_defence_adj

    For each of n_simulations draws:
        1. Perturb λ_eff with N(0, uncertainty × λ) — epistemic noise.
        2. Sample goals_home ~ Poisson(λ̃_home), goals_away ~ Poisson(λ̃_away).
        3. goal_diff = goals_home − goals_away.

    Margin of error = (p_{97.5} − p_{2.5}) / 2  of the GD distribution.
    This is an empirical, non-parametric interval that captures Skellam skewness.
    """
    # Apply injury + tournament form adjustments:
    # injury: home scores more easily if away defence depleted
    # form:   scale by tournament goals-per-game ratio vs average
    lambda_home = max(
        lambda_home
        * home_attack_adj * home_form_atk_adj
        / max(away_defence_adj * away_form_def_adj, 0.75),
        0.05,
    )
    lambda_away = max(
        lambda_away
        * away_attack_adj * away_form_atk_adj
        / max(home_defence_adj * home_form_def_adj, 0.75),
        0.05,
    )
    rng = np.random.default_rng(seed)
    n   = n_simulations

    noise_h = rng.normal(0, home_uncertainty * lambda_home, n)
    noise_a = rng.normal(0, away_uncertainty * lambda_away, n)
    lam_h   = np.maximum(lambda_home + noise_h, 0.05)
    lam_a   = np.maximum(lambda_away + noise_a, 0.05)

    goals_h = rng.poisson(lam_h)
    goals_a = rng.poisson(lam_a)

    # Dixon-Coles low-score correction via importance resampling.
    # Independent Poissons under-predict draws; reweight the four low-score
    # cells by τ (boosting 0-0 / 1-1, trimming 1-0 / 0-1) and resample so the
    # final 10k draws follow the corrected joint distribution. Matches the τ
    # used in _skellam_wdl, keeping the market inversion consistent.
    tau = np.ones(n)
    m00 = (goals_h == 0) & (goals_a == 0)
    m01 = (goals_h == 0) & (goals_a == 1)
    m10 = (goals_h == 1) & (goals_a == 0)
    m11 = (goals_h == 1) & (goals_a == 1)
    tau[m00] = np.maximum(1.0 - lam_h[m00] * lam_a[m00] * DC_RHO, 0.0)
    tau[m01] = np.maximum(1.0 + lam_h[m01] * DC_RHO, 0.0)
    tau[m10] = np.maximum(1.0 + lam_a[m10] * DC_RHO, 0.0)
    tau[m11] = np.maximum(1.0 - DC_RHO, 0.0)
    idx = rng.choice(n, size=n, p=tau / tau.sum())
    goals_h, goals_a = goals_h[idx], goals_a[idx]

    gd = goals_h.astype(int) - goals_a.astype(int)

    p_hw  = float((gd > 0).mean())
    p_d   = float((gd == 0).mean())
    p_aw  = float((gd < 0).mean())

    scorelines = pd.Series(list(zip(goals_h, goals_a))).value_counts()
    top_score  = scorelines.index[0]

    alpha      = 1 - ci_level
    mean_gd    = float(gd.mean())
    ci_lo      = float(np.percentile(gd, 100 * alpha / 2))
    ci_hi      = float(np.percentile(gd, 100 * (1 - alpha / 2)))
    half_width = (ci_hi - ci_lo) / 2.0

    return {
        "lambda_home":       round(lambda_home,  3),
        "lambda_away":       round(lambda_away,  3),
        "prob_home_win":     round(p_hw,          4),
        "prob_draw":         round(p_d,           4),
        "prob_away_win":     round(p_aw,          4),
        "most_likely_score": top_score,
        "mean_goal_diff":    round(mean_gd,       3),
        "ci_low":            round(ci_lo,         3),
        "ci_high":           round(ci_hi,         3),
        "margin_of_error":   round(half_width,    3),
        "ci_level":          ci_level,
        "n_simulations":     n_simulations,
        "home_uncertainty":  round(home_uncertainty, 3),
        "away_uncertainty":  round(away_uncertainty, 3),
    }


def print_match_report(
    home_team: str,
    away_team: str,
    result: Dict,
    home_elo: Optional[float] = None,
    away_elo: Optional[float] = None,
    stage: str = "Group Stage",
) -> None:
    ci_pct = int(result["ci_level"] * 100)
    hs, as_ = result["most_likely_score"]
    gd = result["mean_goal_diff"]

    if gd > 0.05:
        leader, trailer, abs_gd = home_team, away_team, gd
    elif gd < -0.05:
        leader, trailer, abs_gd = away_team, home_team, abs(gd)
    else:
        leader = trailer = None
        abs_gd = 0.0

    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  FIFA WORLD CUP 2026  •  {stage}")
    print(f"  {home_team:^27} vs  {away_team}")
    print(sep)

    if home_elo and away_elo:
        print(f"\n  World Football ELO")
        print(f"    {home_team:<24} : {home_elo:.0f}")
        print(f"    {away_team:<24} : {away_elo:.0f}")

    print(f"\n  Expected Goals (λ)")
    print(f"    {home_team:<24} : {result['lambda_home']:.2f}")
    print(f"    {away_team:<24} : {result['lambda_away']:.2f}")

    print(f"\n  Match Outcome Probabilities")
    bar_w = 30
    for label, prob in [
        (f"{home_team} Win", result["prob_home_win"]),
        ("Draw",             result["prob_draw"]),
        (f"{away_team} Win", result["prob_away_win"]),
    ]:
        bar = "█" * int(prob * bar_w)
        print(f"    {label:<28}  {prob*100:5.1f}%  {bar}")

    print(f"\n  Most Likely Scoreline")
    print(f"    {home_team} {hs} – {as_} {away_team}")

    print(f"\n  Goal Differential  [{ci_pct}% CI]")
    if leader:
        print(f"    {leader} expected to win by "
              f"{abs_gd:.2f} goals  ± {result['margin_of_error']:.2f}")
    else:
        print(f"    Essentially even  (mean GD = {gd:.2f})")
    print(f"    95% CI : [{result['ci_low']:.1f},  {result['ci_high']:.1f}]")

    print(f"\n  Uncertainty (CI inflation factor)")
    for team, u in [(home_team, result["home_uncertainty"]),
                    (away_team, result["away_uncertainty"])]:
        tier = "LOW" if u < 0.25 else ("MED" if u < 0.50 else "HIGH — CI significantly widened")
        print(f"    {team:<24} : {u:.2f}  [{tier}]")

    print(f"\n  Monte Carlo draws : {result['n_simulations']:,}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def build_team_profile(
    df: pd.DataFrame,
    elo_by_name: Dict[str, float],
    team: str,
    is_host: bool = False,
    neutral: bool = True,
    t_weight: float = 1.00,
    elo_min: float = 1300.0,
    elo_max: float = 2200.0,
    include_squad: bool = True,
    wc_team_stats: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Derive a team's feature profile from the most recent rows in the
    engineered dataset, supplemented by live ELO data and WC heritage stats.
    """
    mask   = (df["home_team"] == team) | (df["away_team"] == team)
    recent = df[mask].tail(1)

    raw_elo  = elo_by_name.get(team, 1600.0)
    elo_norm = (raw_elo - elo_min) / (elo_max - elo_min + 1e-9)

    # WC heritage features (safe defaults when dataset unavailable)
    wc_s = _wc_stats_for(team, wc_team_stats) if (
        wc_team_stats is not None and not wc_team_stats.empty
    ) else {"wc_attack_rate": 1.2, "wc_defence_rate": 1.2,
            "wc_stage_score": 0.0, "wc_win_rate": 0.33, "wc_appearances": 0}

    if recent.empty:
        base = {
            "attack":            1.2,
            "defence":           1.2,
            "elo_norm":          elo_norm,
            "uncertainty":       0.65,
            "host_advantage":    int(is_host),
            "neutral_venue":     int(neutral),
            "tournament_weight": t_weight,
        }
    else:
        row = recent.iloc[-1]
        if row["home_team"] == team:
            att, def_, unc = row["home_attack"], row["home_defence"], row["home_uncertainty"]
        else:
            att, def_, unc = row["away_attack"], row["away_defence"], row["away_uncertainty"]
        base = {
            "attack":            att,
            "defence":           def_,
            "elo_norm":          elo_norm,
            "uncertainty":       float(unc),
            "host_advantage":    int(is_host),
            "neutral_venue":     int(neutral),
            "tournament_weight": t_weight,
        }

    # Inject WC heritage into the feature dict for predict_lambdas
    base["wc_attack"]  = wc_s["wc_attack_rate"]
    base["wc_defence"] = wc_s["wc_defence_rate"]
    base["wc_stage"]   = wc_s["wc_stage_score"]
    base["wc_win_rate"]       = wc_s["wc_win_rate"]
    base["wc_appearances"]    = wc_s["wc_appearances"]

    # ── Squad & injury data from ESPN ─────────────────────────────────────────
    if include_squad:
        squad  = fetch_squad(team)
        atk_adj, def_adj, injured = squad_injury_adjustment(squad)
    else:
        squad, atk_adj, def_adj, injured = [], 1.0, 1.0, []

    base["squad"]           = squad
    base["attack_adj"]      = atk_adj
    base["defence_adj"]     = def_adj
    base["injured_players"] = injured
    base["n_injured"]       = len(injured)

    return base


def run_pipeline():
    print("=" * 62)
    print("  FIFA WORLD CUP 2026  —  Prediction Engine")
    print("  Powered by real public data (no API key required)")
    print("=" * 62)

    # ── 1. Ingest ──────────────────────────────────────────────────────────────
    df_raw, elo_by_name, _wc2026, wc_team_stats = build_dataset()

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("[FEATURE ENGINEERING]")
    df = engineer_features(df_raw, elo_by_name, wc_team_stats)
    print(f"  Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # ── 3. Fit Poisson models ─────────────────────────────────────────────────
    predictor = PoissonMatchPredictor()
    predictor.fit(df)

    # ELO range for normalisation (recomputed here to stay consistent)
    all_elos = np.array(list(elo_by_name.values()))
    elo_min, elo_max = float(all_elos.min()), float(all_elos.max())

    def profile(team, is_host=False, neutral=True):
        return build_team_profile(df, elo_by_name, team,
                                   is_host=is_host, neutral=neutral,
                                   elo_min=elo_min, elo_max=elo_max)

    # ── 4. Simulate group-stage fixtures ──────────────────────────────────────
    print("[SIMULATIONS]")

    fixtures = [
        # (home_team, away_team, home_is_host, stage_label)
        ("France",         "Morocco",       False, "Group Stage  •  Neutral Venue"),
        ("United States",  "Germany",       True,  "Group Stage  •  Host Nation Advantage"),
        ("Argentina",      "Japan",         False, "Group Stage  •  Neutral Venue"),
        ("England",        "Panama",        False, "Group Stage  •  Neutral Venue"),
    ]

    for home, away, home_host, stage in fixtures:
        hf = profile(home, is_host=home_host, neutral=not home_host)
        af = profile(away, is_host=False,     neutral=not home_host)

        lh, la = predictor.predict_lambdas(hf, af)
        result = simulate_match(
            lh, la,
            home_uncertainty   = hf["uncertainty"],
            away_uncertainty   = af["uncertainty"],
            home_attack_adj    = hf.get("attack_adj",  1.0),
            home_defence_adj   = hf.get("defence_adj", 1.0),
            away_attack_adj    = af.get("attack_adj",  1.0),
            away_defence_adj   = af.get("defence_adj", 1.0),
        )
        print_match_report(
            home, away, result,
            home_elo=elo_by_name.get(home),
            away_elo=elo_by_name.get(away),
            stage=stage,
        )
        # Print squad summary
        for team_name, feats in [(home, hf), (away, af)]:
            squad = feats.get("squad", [])
            injured = feats.get("injured_players", [])
            if squad:
                print(f"  {team_name} squad ({len(squad)} players):")
                for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
                    group = [p for p in squad if pos in p["position"]]
                    names = ", ".join(
                        f"{'⚠ ' if p['is_injured'] else ''}{p['name']}"
                        for p in group
                    )
                    if names:
                        print(f"    {pos:<12}: {names}")
                if injured:
                    print(f"\n  ⚠  {team_name} injury concerns:")
                    for p in injured:
                        detail = f" — {p['injury_detail']}" if p["injury_detail"] else ""
                        print(f"    • {p['name']} ({p['position']}){detail}")
                else:
                    print(f"  ✓  {team_name}: No injury concerns reported.")
                print()

    # ── Interpretation note ────────────────────────────────────────────────────
    print("─" * 62)
    print("  MARGIN OF ERROR — MATHEMATICAL NOTE")
    print("─" * 62)
    print("""
  half_width = (p_97.5 − p_2.5) / 2

  where p_2.5 and p_97.5 are the 2.5th/97.5th percentiles of the
  10,000 simulated goal-differential (GD) values.

  Why empirical percentiles (not ±1.96σ)?
  The difference of two Poisson variates follows a Skellam
  distribution, which is asymmetric when λ_home ≠ λ_away.  A
  normal-approximation CI (mean ± 1.96σ) under-covers one tail.
  The percentile method is automatically correct for any shape.

  Uncertainty inflation: before each draw, λ is perturbed by
  N(0, uncertainty × λ).  Teams with sparse or low-quality
  match history get higher uncertainty scores → wider CI.
  France (many elite matches) ≈ 0.08; Panama ≈ 0.55+.
""")


if __name__ == "__main__":
    run_pipeline()
