"""
llm_agent.py
─────────────
LLM reasoning layer for the Data Crawler pipeline.
Supports two backends, selected via the LLM_PROVIDER environment variable:

  LLM_PROVIDER=anthropic  (default)
    Uses the Anthropic SDK (claude-opus-4-6).
    Requires: ANTHROPIC_API_KEY

  LLM_PROVIDER=openai
    Uses the OpenAI-compatible API — works with Ollama, llama.cpp, LM Studio,
    vLLM, OpenAI, and any other server that implements /v1/chat/completions.
    Requires: OPENAI_BASE_URL  (e.g. http://localhost:11434/v1 for Ollama)
              OPENAI_API_KEY   (default: "ollama" — any non-empty string works
                                for local servers that don't enforce auth)
              LLM_MODEL        (e.g. llama3.2, mistral, qwen2.5 — required for
                                local models; ignored when using Anthropic)

Quick-start examples
─────────────────────
  # Anthropic (cloud)
  export LLM_PROVIDER=anthropic
  export ANTHROPIC_API_KEY=sk-ant-...

  # Ollama (local)
  ollama pull llama3.2
  export LLM_PROVIDER=openai
  export OPENAI_BASE_URL=http://localhost:11434/v1
  export LLM_MODEL=llama3.2

  # llama.cpp server (local)
  export LLM_PROVIDER=openai
  export OPENAI_BASE_URL=http://localhost:8080/v1
  export OPENAI_API_KEY=sk-no-key-required
  export LLM_MODEL=local-model
"""

import json
import os
import re

# ─────────────────────────────────────────────────────────────────────────────
# Provider configuration  (read once at import time)
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower().strip()

# Anthropic defaults
_ANTHROPIC_MODEL = os.environ.get("LLM_MODEL", "claude-opus-4-6")

# OpenAI-compatible defaults
_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
_OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "ollama")
_OPENAI_MODEL    = os.environ.get("LLM_MODEL", "llama3.2")

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    if _PROVIDER == "openai":
        from openai import OpenAI
        _client = OpenAI(base_url=_OPENAI_BASE_URL, api_key=_OPENAI_API_KEY)
        print(f"[llm_agent] Using OpenAI-compatible backend: {_OPENAI_BASE_URL} | model: {_OPENAI_MODEL}")
    else:
        import anthropic
        _client = anthropic.Anthropic()
        print(f"[llm_agent] Using Anthropic backend | model: {_ANTHROPIC_MODEL}")

    return _client


_SYSTEM = (
    "You are an expert video dataset curator helping build high-quality "
    "training datasets by reasoning about what visual content to search for "
    "and whether collected video clips match the target concept."
)


# ─────────────────────────────────────────────────────────────────────────────
# Unified chat helper
# ─────────────────────────────────────────────────────────────────────────────

def _chat(prompt: str, max_tokens: int = 500, use_thinking: bool = False) -> str:
    """
    Send a single-turn prompt and return the text response.

    `use_thinking=True` enables adaptive thinking on Anthropic; it is silently
    ignored for OpenAI-compatible backends (local models don't support it).
    """
    client = _get_client()

    if _PROVIDER == "openai":
        response = client.chat.completions.create(
            model=_OPENAI_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    # Anthropic path
    import anthropic as _anthropic
    kwargs = dict(
        model=_ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}
        # adaptive thinking requires streaming on Opus
        with client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
    else:
        response = client.messages.create(**kwargs)

    return next((b.text for b in response.content if b.type == "text"), "")


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str, default):
    """
    Robustly extract the first JSON object or array from an LLM response.
    Local models are chattier than cloud models, so we need to be tolerant
    of surrounding prose, markdown fences, etc.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract the first JSON array or object
    for pattern in (r'\[[\s\S]*?\]', r'\{[\s\S]*?\}'):
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    return default


def _provider_info() -> dict:
    """Return current provider configuration (useful for the UI/logs)."""
    if _PROVIDER == "openai":
        return {"provider": "openai", "base_url": _OPENAI_BASE_URL, "model": _OPENAI_MODEL}
    return {"provider": "anthropic", "model": _ANTHROPIC_MODEL}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Search Query Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_search_queries(
    captions: list,
    detected_concepts: list,
    n: int = 3,
) -> list:
    """
    Generate n diverse YouTube search queries from image analysis.

    Uses thinking (Anthropic) or plain completion (OpenAI-compatible) to
    reason about what kinds of videos would contain this visual content.
    """
    prompt = (
        f"You are helping build a video dataset by crawling YouTube.\n\n"
        f"BLIP captions from query images: {captions}\n"
        f"CLIP-detected visual concepts: {detected_concepts}\n\n"
        f"Generate {n} diverse YouTube search queries to find videos containing this type of content.\n\n"
        f"Rules:\n"
        f"- Each query approaches the content from a different angle "
        f"(raw footage, highlights, tutorials, compilations, slow-motion, etc.)\n"
        f"- Be specific enough to find relevant content, not so niche it returns nothing\n"
        f"- Think about how YouTube creators title their videos\n"
        f"- Vary terminology and phrasing across queries\n\n"
        f"Return ONLY a valid JSON array of {n} query strings. No other text, no markdown.\n"
        f'Example: ["cricket batting highlights 2024", '
        f'"cricket batting technique slow motion", '
        f'"best cricket shots compilation"]'
    )

    text = _chat(prompt, max_tokens=600, use_thinking=True)
    result = _extract_json(text, [])
    return result if isinstance(result, list) else []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pre-Download Video Filtering  (scales to 100s of candidates)
# ─────────────────────────────────────────────────────────────────────────────

_BATCH_SIZE      = 20   # videos per LLM scoring call
_SCORE_THRESHOLD = 6    # minimum relevance score (0-10) to keep


def _score_batch(query_description: str, batch: list) -> list:
    """
    Score a batch of ≤ _BATCH_SIZE videos 0-10 for relevance.
    Returns a list of ints, same length as batch.
    """
    def _fmt(i, v):
        dur = f"{int(v['duration'])}s" if v.get("duration") else "?"
        return f"{i + 1}. \"{v['title']}\" | {v.get('channel', '?')} | {dur}"

    numbered = "\n".join(_fmt(i, v) for i, v in enumerate(batch))

    prompt = (
        f'Rate each video\'s relevance to: "{query_description}"\n\n'
        f"Score 0-10:\n"
        f"  10 = perfect match\n"
        f"   7 = likely relevant\n"
        f"   4 = possibly relevant\n"
        f"   0 = definitely irrelevant (news clip, reaction, off-topic)\n\n"
        f"{numbered}\n\n"
        f"Return ONLY a JSON array of {len(batch)} integers, one per video, same order.\n"
        f"No markdown, no explanation. Example for 3 videos: [8, 2, 7]"
    )

    text = _chat(prompt, max_tokens=150)
    scores = _extract_json(text, [])

    if not isinstance(scores, list) or len(scores) != len(batch):
        return [5] * len(batch)   # neutral fallback
    return [max(0, min(10, int(s))) for s in scores]


def filter_videos_by_relevance(
    query_description: str,
    candidates: list,
    keep_n: int = 5,
) -> tuple:
    """
    Filter a candidate list before downloading.

    Small lists (≤ _BATCH_SIZE): single call, pick by index.
    Large lists (> _BATCH_SIZE): batch-score every video 0-10, keep all
    scoring ≥ _SCORE_THRESHOLD sorted by score, trim to keep_n.
    """
    if not candidates:
        return [], "No candidates to filter."

    # ── Small list ────────────────────────────────────────────────────────────
    if len(candidates) <= _BATCH_SIZE:
        lines = []
        for i, v in enumerate(candidates):
            dur = f"{int(v['duration'])}s" if v.get("duration") else "?"
            lines.append(f"{i + 1}. \"{v['title']}\" | {v.get('channel', '?')} | {dur}")
        numbered = "\n".join(lines)

        prompt = (
            f'Building a dataset for: "{query_description}"\n\n'
            f"Candidates:\n{numbered}\n\n"
            f"Pick up to {keep_n} most likely to contain the target content.\n"
            f"Avoid: news clips, reactions, off-topic content with similar keywords.\n"
            f"Return ONLY JSON, no markdown: "
            f'{{"keep": [1, 3], "reasoning": "brief explanation"}}'
        )

        text = _chat(prompt, max_tokens=300)
        result = _extract_json(text, {})
        keep_idx = [i - 1 for i in result.get("keep", []) if 1 <= i <= len(candidates)]
        seen, filtered = set(), []
        for i in keep_idx:
            if i not in seen:
                seen.add(i)
                filtered.append(candidates[i])
        return filtered[:keep_n], result.get("reasoning", "")

    # ── Large list: batch scoring ─────────────────────────────────────────────
    print(f"  [llm_agent] Batch-scoring {len(candidates)} candidates ({_BATCH_SIZE}/batch)...")
    scored = []

    for start in range(0, len(candidates), _BATCH_SIZE):
        batch = candidates[start : start + _BATCH_SIZE]
        scores = _score_batch(query_description, batch)
        scored.extend(zip(batch, scores))
        print(f"  [llm_agent] Scored {min(start + _BATCH_SIZE, len(candidates))}/{len(candidates)}")

    passing = [(v, s) for v, s in scored if s >= _SCORE_THRESHOLD]
    passing.sort(key=lambda x: x[1], reverse=True)

    filtered = [v for v, _ in passing[:keep_n]]
    avg = sum(s for _, s in passing[:keep_n]) / max(len(filtered), 1)

    reasoning = (
        f"Scored {len(candidates)} candidates via {_PROVIDER}. "
        f"{len(passing)} scored ≥{_SCORE_THRESHOLD}/10. "
        f"Keeping top {len(filtered)} (avg {avg:.1f}/10)."
    )
    print(f"  [llm_agent] {reasoning}")
    return filtered, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# 3. Result Evaluation & Next-Action Decision
# ─────────────────────────────────────────────────────────────────────────────

def analyze_crawl_results(
    query_description: str,
    results: list,
    threshold: float,
    attempt: int,
    max_attempts: int,
) -> dict:
    """
    Evaluate crawl results and decide the next action.

    Returns a dict with:
      assessment      — "good" | "partial" | "poor"
      should_retry    — bool
      suggested_query — str or None
      threshold_delta — float  (e.g. -0.03 to lower threshold)
      diagnosis       — str
      stop_reason     — str or None
    """
    total_clips = sum(r.get("segments", 0) for r in results)
    passed = sum(1 for r in results if r.get("status") == "passed")
    summary = "\n".join(
        f"- \"{r['title'][:60]}\": {r['segments']} segments ({r['status']})"
        for r in results
    ) or "No results."

    prompt = (
        f"Evaluating video dataset crawl. Attempt {attempt}/{max_attempts}.\n\n"
        f'Target: "{query_description}"\n'
        f"Threshold: {threshold:.3f} | Clips found: {total_clips} "
        f"across {len(results)} videos ({passed} had matches)\n\n"
        f"Per-video:\n{summary}\n\n"
        f"Decide next action. Rules:\n"
        f"- ≥5 clips from ≥2 videos = acceptable yield\n"
        f"- 0 segments everywhere = threshold too high OR wrong query\n"
        f"- Mixed results = query likely correct, lower threshold slightly\n"
        f"- All videos irrelevant = change query\n\n"
        f"Return ONLY valid JSON, no markdown:\n"
        f'{{"assessment":"good|partial|poor","should_retry":true,"suggested_query":null,'
        f'"threshold_delta":0.0,"diagnosis":"explanation","stop_reason":null}}\n\n'
        f"threshold_delta: negative to lower (e.g. -0.03), positive to raise, 0.0 to keep."
    )

    text = _chat(prompt, max_tokens=400, use_thinking=True)
    result = _extract_json(text, {})

    return {
        "assessment":      result.get("assessment", "poor"),
        "should_retry":    bool(result.get("should_retry", False)),
        "suggested_query": result.get("suggested_query"),
        "threshold_delta": float(result.get("threshold_delta", 0.0)),
        "diagnosis":       result.get("diagnosis", ""),
        "stop_reason":     result.get("stop_reason"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visual Content Description
# ─────────────────────────────────────────────────────────────────────────────

def describe_visual_content(captions: list, concepts: list) -> str:
    """One sentence description of the target video content."""
    if not captions and not concepts:
        return "visual content matching query images"

    prompt = (
        f"Image captions: {captions}\n"
        f"Visual concepts: {concepts}\n\n"
        f"In exactly one concise sentence, describe what type of video content should be searched for. "
        f"Focus on the visual action or scene. No preamble."
    )

    text = _chat(prompt, max_tokens=80)
    return text.strip() or "visual content matching query images"
