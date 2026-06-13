"""
Pydantic data models for pre-match scouting reports.
All fields are optional to gracefully handle varying media outlet coverage.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class TechnicalDetails(BaseModel):
    venue: Optional[str] = None
    referee: Optional[str] = None
    kickoff_time: Optional[str] = None


class RosterDynamics(BaseModel):
    injured_players: List[str] = Field(default_factory=list)
    suspended_players: List[str] = Field(default_factory=list)
    doubtful_players: List[str] = Field(default_factory=list)
    # key = team name, value = list of player names in order
    probable_lineups: Dict[str, List[str]] = Field(default_factory=dict)


class QuantitativeAnchors(BaseModel):
    # e.g. {"home": 0.52, "draw": 0.26, "away": 0.22}
    win_probabilities: Dict[str, float] = Field(default_factory=dict)
    key_stats: List[str] = Field(default_factory=list)
    # e.g. {"home": "WWDLL", "away": "DWWLD"}
    form_guide: Dict[str, str] = Field(default_factory=dict)
    head_to_head_summary: Optional[str] = None


class PreMatchReport(BaseModel):
    # ── Metadata ────────────────────────────────────────────────────────────
    source_url: str
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    match_date: Optional[str] = None        # ISO date string  e.g. "2026-06-14"
    tournament: Optional[str] = None
    report_title: Optional[str] = None

    # ── Teams ────────────────────────────────────────────────────────────────
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    match_slug: str = ""                    # generated if blank

    # ── Sub-models ───────────────────────────────────────────────────────────
    technical_details: TechnicalDetails = Field(default_factory=TechnicalDetails)
    roster_dynamics: RosterDynamics = Field(default_factory=RosterDynamics)
    quantitative_anchors: QuantitativeAnchors = Field(default_factory=QuantitativeAnchors)

    # ── Textual content ──────────────────────────────────────────────────────
    tactical_narrative: str = ""            # LLM-distilled tactical summary
    raw_markdown: str = ""                  # full scraped markdown for RAG

    @model_validator(mode="after")
    def _generate_slug(self) -> "PreMatchReport":
        if not self.match_slug:
            home = _slugify(self.home_team or "unknown")
            away = _slugify(self.away_team or "unknown")
            # Slugify the date too — handles "June 8, 2026", "2026-06-08", bare timestamps, etc.
            date = _slugify(self.match_date or self.extraction_timestamp.strftime("%Y%m%d"))
            self.match_slug = f"{home}_vs_{away}_{date}"
        return self


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
