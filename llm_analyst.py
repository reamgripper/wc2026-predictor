"""
LLM Match Analyst — bring-your-own-key, provider-neutral
=========================================================
Sends STRUCTURED prediction data (numbers only, no free text) to any
OpenAI-compatible chat endpoint and streams back a written analysis.

Why OpenAI-compatible?
----------------------
Qwen — and almost every open model — is served through the OpenAI chat API
shape, so one tiny module works against all of them. The caller just supplies
a base URL + model name + key:

    Ollama (local, free) :  http://localhost:11434/v1   ·  qwen2.5         ·  no key
    OpenRouter           :  https://openrouter.ai/api/v1 ·  qwen/qwen-2.5-72b-instruct
    Alibaba DashScope    :  https://dashscope-intl.aliyuncs.com/compatible-mode/v1 · qwen-plus
    Together AI          :  https://api.together.xyz/v1 ·  Qwen/Qwen2.5-72B-Instruct-Turbo
    DeepInfra            :  https://api.deepinfra.com/v1/openai · Qwen/Qwen2.5-72B-Instruct

Key handling
------------
The key is passed in per call from the caller's session memory. This module
NEVER reads from or writes to disk, env vars, or any shared store — so a key a
user pastes into the browser lives only for that browser session and is gone
the moment they close the tab.
"""

import json
import math
from typing import Dict, Generator, List, Optional


class _SafeEncoder(json.JSONEncoder):
    """Convert numpy/pandas scalar types that standard json can't handle."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return super().default(obj)

SYSTEM_PROMPT = """\
You are a senior football analytics expert reviewing the output of a \
statistical match-prediction model for the FIFA World Cup 2026.

The model combines: Poisson regression on 1,600+ historical international \
matches, live World Football ELO ratings, squad/injury adjustments from live \
rosters, a Dixon-Coles draw correction, optional blending with bookmaker \
odds, and a 10,000-draw Monte Carlo simulation producing win/draw/loss \
probabilities and a 95% confidence interval on goal differential.

You will receive one JSON object containing the prediction for a single \
match. Write a concise analysis (250-400 words) covering:

1. **The headline** — who is favoured and how strongly, in plain English.
2. **What drives it** — interpret the expected-goals (lambda) values, ELO \
gap, and any injury adjustments. Which factor matters most here?
3. **Trust level** — interpret the margin of error and uncertainty scores. \
Is this a confident call or close to a coin flip? If model and betting \
market disagree notably, say which you'd lean toward and why.
4. **The upset case** — in one short paragraph, what realistically has to \
happen for the underdog to win, grounded in the numbers.

Style: sharp, broadcast-pundit-meets-quant. No bullet-point dumps — flowing \
short paragraphs with the four bold mini-headers above. Refer to teams by \
name. Quote the key numbers (percentages, lambdas) naturally in the prose. \
Never invent statistics that are not in the JSON; if a field is null or \
missing, simply don't mention it."""


def build_match_payload(
    home: str,
    away: str,
    result: Dict,
    model_result: Dict,
    odds: Optional[Dict],
    h_elo: Optional[float],
    a_elo: Optional[float],
    h_injured: list,
    a_injured: list,
    hf: Dict,
    af: Dict,
    stage: str,
    market_weight: float,
    blend_applied: bool,
) -> Dict:
    """
    Assemble the structured JSON the LLM analyses.
    Every field is numeric/enumerated app state — never user-typed text,
    which keeps the prompt-injection surface at zero.
    """
    payload = {
        "match": {"home_team": home, "away_team": away, "stage": stage},
        "final_prediction": {
            "prob_home_win":   result["prob_home_win"],
            "prob_draw":       result["prob_draw"],
            "prob_away_win":   result["prob_away_win"],
            "lambda_home":     result["lambda_home"],
            "lambda_away":     result["lambda_away"],
            "most_likely_score": list(result["most_likely_score"]),
            "mean_goal_diff":  result["mean_goal_diff"],
            "margin_of_error": result["margin_of_error"],
            "ci_95":           [result["ci_low"], result["ci_high"]],
        },
        "pure_model_prediction": {
            "prob_home_win": model_result["prob_home_win"],
            "prob_draw":     model_result["prob_draw"],
            "prob_away_win": model_result["prob_away_win"],
        },
        "elo": {"home": h_elo, "away": a_elo},
        "uncertainty": {
            "home": hf.get("uncertainty"),
            "away": af.get("uncertainty"),
        },
        "injuries": {
            "home_players_out": [
                {"name": p["name"], "position": p["position"]} for p in h_injured
            ],
            "away_players_out": [
                {"name": p["name"], "position": p["position"]} for p in a_injured
            ],
        },
        "betting_market": None,
    }
    if blend_applied and odds:
        payload["betting_market"] = {
            "provider":            odds.get("provider"),
            "prob_home_win":       odds.get("market_prob_home"),
            "prob_draw":           odds.get("market_prob_draw"),
            "prob_away_win":       odds.get("market_prob_away"),
            "over_under_line":     odds.get("over_under"),
            "implied_total_goals": odds.get("market_total_goals"),
            "blend_weight_used":   market_weight,
        }
    return payload


def stream_analysis(
    payload: Dict,
    api_key: str,
    base_url: str,
    model: str,
) -> Generator[str, None, None]:
    """
    Stream an analysis of `payload` from any OpenAI-compatible endpoint,
    yielding text chunks for st.write_stream().

    api_key / base_url / model come from the caller's per-session state.
    Raises RuntimeError with a friendly message on any failure.
    """
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("The `openai` package is not installed. Run: pip install openai")

    # Empty key is valid for local Ollama; the SDK just needs a non-None string.
    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url or None)

    messages: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, cls=_SafeEncoder)},
    ]

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=2048,
            temperature=0.7,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text
    except openai.AuthenticationError:
        raise RuntimeError("The API key was rejected. Check the key and that it matches the selected provider.")
    except openai.NotFoundError:
        raise RuntimeError(f"Model '{model}' was not found at this provider. Check the exact model name.")
    except openai.PermissionDeniedError:
        raise RuntimeError("This key doesn't have access to that model. Check your provider plan.")
    except openai.RateLimitError:
        raise RuntimeError("Rate limited or out of quota at the provider. Wait a minute or check your balance.")
    except openai.APIConnectionError:
        raise RuntimeError(
            "Couldn't reach the LLM endpoint. If you're using Ollama locally, make sure it's "
            "running (`ollama serve`) and the model is pulled (`ollama pull qwen2.5`). "
            "Note: local Ollama only works when running the app on your own machine, not on a hosted server."
        )
    except openai.APIStatusError as e:
        raise RuntimeError(f"LLM service error ({e.status_code}). Try again later.")
