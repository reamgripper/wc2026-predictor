"""
RAG Orchestration Engine
========================
LangChain + Ollama (local) + FAISS vector store.
All processing is fully local — no external API calls.

Embedding model : the same Ollama model used for chat (avoids pulling extra weights)
Vector store    : FAISS in-memory (re-built per session; fast for small corpora)
Splitter        : RecursiveCharacterTextSplitter (semantic boundaries first)
Chain           : RetrievalQA with a domain-specific Pro-Analyst prompt

Usage
-----
    from expert_analysis.rag_engine import RAGEngine

    engine = RAGEngine(base_url="http://localhost:11434", model="deepseek-r1:1.5b")
    engine.build_index([("slug1", markdown1), ("slug2", markdown2)])
    answer = engine.query("What injury absences most affect the defensive shape?")

    # Single-doc direct analysis (no index required)
    answer = engine.direct_analysis(markdown, "What is the pressing strategy?")

    # Structured extraction
    extracted = engine.extract_structure(markdown)
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

# ── System prompts ─────────────────────────────────────────────────────────────

_ANALYST_PROMPT_TEMPLATE = """\
You are an elite Pro Football Performance Analyst. Your analysis must be grounded \
exclusively in the tactical and statistical evidence provided — do not invent facts.

Your area of focus:
• **Micro-situational KPIs**: pressures per 90, PPDA, xG split (open play vs. set pieces),
  progressive carries, line-breaking passes
• **Phase transitions**: how each side reorganises defensively after losing the ball, and
  attacks from structured vs. transition situations
• **Spatial mechanics**: high-block vs. mid-block vs. low-block tendencies; width \
exploitation; half-space occupation
• **Injury-induced vulnerabilities**: pinpoint how specific absences reshape the back line,
  press intensity, or set-piece threats — be explicit and specific
• **Matchup asymmetries**: identify the one or two individual duels that will decide the game

Context from scouting reports:
-------------------------------
{context}
-------------------------------

Analyst question: {question}

Write a sharp, pundit-quality analytical response. Lead with your headline verdict, \
support it with numbers and player names where available, and close with a single \
"match-deciding factor" sentence. Avoid vague generalities.

Analysis:"""

_EXTRACTION_PROMPT = """\
You are a football data extraction specialist. Read the match preview text below \
and return ONLY a valid JSON object — no markdown fences, no preamble.

Required schema (use null for any missing field, [] for empty lists):
{{
  "home_team": "string or null",
  "away_team": "string or null",
  "match_date": "YYYY-MM-DD or null",
  "tournament": "string or null",
  "venue": "string or null",
  "referee": "string or null",
  "kickoff_time": "string or null",
  "injured_players_home": [],
  "injured_players_away": [],
  "suspended_players_home": [],
  "suspended_players_away": [],
  "doubtful_players_home": [],
  "doubtful_players_away": [],
  "probable_lineup_home": [],
  "probable_lineup_away": [],
  "win_probability_home": null,
  "win_probability_draw": null,
  "win_probability_away": null,
  "form_home": "WWDLL or null",
  "form_away": "WWDLL or null",
  "key_stats": [],
  "head_to_head_summary": "string or null",
  "tactical_narrative": "2-3 sentence expert tactical summary"
}}

Match preview text (first {max_chars} characters):
---
{text}
---

JSON:"""


# ── Engine ─────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Wraps LangChain, Ollama, and FAISS into a single re-usable object.

    Parameters
    ----------
    base_url : str
        Ollama base URL, e.g. ``"http://localhost:11434"``.
        If the caller passes the OpenAI-compat URL (``…/v1``), the ``/v1``
        suffix is stripped because LangChain's Ollama integration uses the
        native Ollama endpoint.
    model : str
        Ollama model tag, e.g. ``"deepseek-r1:1.5b"`` or ``"llama3"``.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        # Strip OpenAI-compat suffix if present
        self.base_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
        self.model = model
        self._vectorstore = None

    # ── Dependency checks ──────────────────────────────────────────────────────

    @staticmethod
    def check_deps() -> List[str]:
        """Return list of missing package names (empty = all good)."""
        missing = []
        for pkg in ("langchain", "langchain_community", "faiss"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg.replace("_", "-"))
        return missing

    # ── LLM / embedding factories ──────────────────────────────────────────────

    def _llm(self):
        from langchain_community.llms import Ollama  # type: ignore
        return Ollama(model=self.model, base_url=self.base_url, temperature=0.15)

    def _embeddings(self):
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
        return OllamaEmbeddings(model=self.model, base_url=self.base_url)

    # ── Index building ─────────────────────────────────────────────────────────

    def build_index(self, docs: List[Tuple[str, str]]) -> int:
        """
        Build a FAISS index from a list of ``(slug, markdown_text)`` pairs.

        Returns the number of chunks indexed.
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
        from langchain.schema import Document  # type: ignore
        from langchain_community.vectorstores import FAISS  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        all_docs = []
        for slug, text in docs:
            if not text.strip():
                continue
            for chunk in splitter.split_text(text):
                all_docs.append(Document(page_content=chunk, metadata={"slug": slug}))

        if not all_docs:
            raise ValueError("No content found in the provided documents.")

        self._vectorstore = FAISS.from_documents(all_docs, self._embeddings())
        return len(all_docs)

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, question: str, k: int = 5) -> str:
        """
        Run a RAG query against the indexed documents.
        Raises ``RuntimeError`` if ``build_index`` has not been called.
        """
        if self._vectorstore is None:
            raise RuntimeError("No index built. Call build_index() first.")

        from langchain.chains import RetrievalQA  # type: ignore
        from langchain.prompts import PromptTemplate  # type: ignore

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=_ANALYST_PROMPT_TEMPLATE,
        )
        chain = RetrievalQA.from_chain_type(
            llm=self._llm(),
            chain_type="stuff",
            retriever=self._vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False,
        )
        result = chain.invoke({"query": question})
        return result.get("result", str(result))

    # ── Direct (no index) ──────────────────────────────────────────────────────

    def direct_analysis(self, markdown: str, question: str, max_chars: int = 10_000) -> str:
        """
        Feed the markdown directly into the LLM context — no vector retrieval.
        Fast for single-document queries and avoids the embedding step.
        """
        context = markdown[:max_chars]
        prompt = _ANALYST_PROMPT_TEMPLATE.format(context=context, question=question)
        return self._llm().invoke(prompt)

    # ── Structured extraction ──────────────────────────────────────────────────

    def extract_structure(self, markdown: str, max_chars: int = 6_000) -> Dict:
        """
        Ask the LLM to extract structured fields from a raw markdown preview.
        Returns a dict that maps onto ``PreMatchReport`` fields.
        Returns ``{}`` on any parse failure (caller should treat as partial).
        """
        prompt = _EXTRACTION_PROMPT.format(
            text=markdown[:max_chars],
            max_chars=max_chars,
        )
        raw = self._llm().invoke(prompt)

        # Extract the first JSON block from the response
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    # ── Convenience: extraction → PreMatchReport ───────────────────────────────

    def extraction_to_report(
        self, extracted: Dict, source_url: str, raw_markdown: str, title: str
    ):
        """Convert the raw extraction dict into a ``PreMatchReport``."""
        from .models import (  # local import to avoid circular at module level
            PreMatchReport, RosterDynamics, QuantitativeAnchors, TechnicalDetails,
        )

        def _merge_players(home_key: str, away_key: str) -> List[str]:
            home = extracted.get(home_key) or []
            away = extracted.get(away_key) or []
            tagged_home = [f"{p} (home)" for p in home]
            tagged_away = [f"{p} (away)" for p in away]
            return tagged_home + tagged_away

        probs: Dict[str, float] = {}
        for key, field in [("home", "win_probability_home"),
                           ("draw", "win_probability_draw"),
                           ("away", "win_probability_away")]:
            val = extracted.get(field)
            if val is not None:
                try:
                    probs[key] = float(val)
                except (TypeError, ValueError):
                    pass

        form: Dict[str, str] = {}
        if extracted.get("form_home"):
            form[extracted.get("home_team") or "home"] = extracted["form_home"]
        if extracted.get("form_away"):
            form[extracted.get("away_team") or "away"] = extracted["form_away"]

        lineups: Dict[str, List[str]] = {}
        if extracted.get("probable_lineup_home"):
            lineups[extracted.get("home_team") or "home"] = extracted["probable_lineup_home"]
        if extracted.get("probable_lineup_away"):
            lineups[extracted.get("away_team") or "away"] = extracted["probable_lineup_away"]

        return PreMatchReport(
            source_url=source_url,
            report_title=title,
            match_date=extracted.get("match_date"),
            tournament=extracted.get("tournament"),
            home_team=extracted.get("home_team"),
            away_team=extracted.get("away_team"),
            technical_details=TechnicalDetails(
                venue=extracted.get("venue"),
                referee=extracted.get("referee"),
                kickoff_time=extracted.get("kickoff_time"),
            ),
            roster_dynamics=RosterDynamics(
                injured_players=_merge_players("injured_players_home", "injured_players_away"),
                suspended_players=_merge_players("suspended_players_home", "suspended_players_away"),
                doubtful_players=_merge_players("doubtful_players_home", "doubtful_players_away"),
                probable_lineups=lineups,
            ),
            quantitative_anchors=QuantitativeAnchors(
                win_probabilities=probs,
                key_stats=extracted.get("key_stats") or [],
                form_guide=form,
                head_to_head_summary=extracted.get("head_to_head_summary"),
            ),
            tactical_narrative=extracted.get("tactical_narrative", ""),
            raw_markdown=raw_markdown,
        )
