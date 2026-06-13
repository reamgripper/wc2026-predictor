"""
Local storage utilities.

Paths are resolved relative to the project root (parent of this package),
so ``streamlit run wc2026_ui.py`` and direct script invocations both write
to the same ``<project_root>/data/`` tree.

  data/
    processed/        ← structured JSON (one file per report)
    raw_markdown/     ← raw markdown (one file per report, used for RAG)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from .models import PreMatchReport

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = _PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_ROOT / "processed"
RAW_MD_DIR = DATA_ROOT / "raw_markdown"


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_MD_DIR.mkdir(parents=True, exist_ok=True)


# ── Write ─────────────────────────────────────────────────────────────────────

def save_report(report: PreMatchReport) -> tuple[Path, Path]:
    """
    Persist *report* to disk.

    Returns ``(json_path, markdown_path)``.
    """
    _ensure_dirs()
    slug = report.match_slug

    # ── Structured JSON ───────────────────────────────────────────────────────
    json_path = PROCESSED_DIR / f"{slug}.json"
    data = report.model_dump(mode="json")
    # Ensure datetime fields are plain strings
    for key in ("extraction_timestamp",):
        if isinstance(data.get(key), datetime):
            data[key] = data[key].isoformat()
        elif hasattr(data.get(key), "isoformat"):
            data[key] = data[key].isoformat()
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Raw Markdown (for RAG embedding) ─────────────────────────────────────
    md_path = RAW_MD_DIR / f"{slug}.md"
    header = (
        f"# {report.report_title or slug}\n\n"
        f"**Source:** {report.source_url}\n"
        f"**Date:** {report.match_date or 'Unknown'}\n"
        f"**Teams:** {report.home_team or '?'} vs {report.away_team or '?'}\n"
        f"**Tournament:** {report.tournament or 'Unknown'}\n\n"
        "---\n\n"
    )
    md_path.write_text(header + report.raw_markdown, encoding="utf-8")

    return json_path, md_path


# ── Read ──────────────────────────────────────────────────────────────────────

def list_reports() -> List[dict]:
    """Return lightweight summary dicts, sorted newest-first."""
    _ensure_dirs()
    reports = []
    for p in sorted(PROCESSED_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            reports.append({
                "slug":                 p.stem,
                "home_team":            data.get("home_team") or "?",
                "away_team":            data.get("away_team") or "?",
                "tournament":           data.get("tournament") or "?",
                "match_date":           data.get("match_date") or "?",
                "source_url":           data.get("source_url", ""),
                "extraction_timestamp": data.get("extraction_timestamp", ""),
                "has_structure":        bool(data.get("home_team")),
            })
        except Exception:
            pass
    return reports


def load_report(slug: str) -> PreMatchReport:
    path = PROCESSED_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"No report found for slug: {slug}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return PreMatchReport(**data)


def load_markdown(slug: str) -> str:
    path = RAW_MD_DIR / f"{slug}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def all_markdowns() -> List[tuple[str, str]]:
    """Return list of ``(slug, markdown_text)`` for every saved report."""
    _ensure_dirs()
    result = []
    for p in sorted(RAW_MD_DIR.glob("*.md")):
        try:
            result.append((p.stem, p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return result


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_report(slug: str) -> None:
    for directory, ext in [(PROCESSED_DIR, ".json"), (RAW_MD_DIR, ".md")]:
        path = directory / f"{slug}{ext}"
        if path.exists():
            path.unlink()
