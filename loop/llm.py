"""W&B Inference chat wrapper used by the failure-seeking loop."""

from __future__ import annotations

import json
import os
import re

import openai

try:
    from . import tracing
except ImportError:  # `import llm` with loop/ on sys.path
    import tracing

tracing.init()

INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
# Strong generally-available open model. Candidates are listed in loop/README.md.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro-0813"


def _api_key() -> str:
    return os.environ.get("WANDB_API_KEY", "").strip()


def _inference_project() -> str | None:
    """OpenAI-Project header value: team/project, per W&B Inference docs."""
    explicit = os.environ.get("WANDB_INFERENCE_PROJECT", "").strip()
    if explicit:
        return explicit
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "sporelabs-hackathon").strip() or "sporelabs-hackathon"
    if entity:
        return f"{entity}/{project}"
    return None


def _client() -> openai.OpenAI:
    kwargs: dict = {
        "base_url": INFERENCE_BASE_URL,
        "api_key": _api_key(),
    }
    project = _inference_project()
    if project:
        kwargs["project"] = project  # sent as the OpenAI-Project header
    return openai.OpenAI(**kwargs)


@tracing.traced
def chat(messages, model=None, json_mode=False) -> str:
    """Chat completion against W&B Inference. Returns "" when no API key is set."""
    if not _api_key():
        return ""
    kwargs = {
        "model": model or os.environ.get("SPORE_LLM_MODEL", "").strip() or DEFAULT_MODEL,
        "messages": messages,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = _client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _label(condition) -> str:
    if isinstance(condition, dict):
        keys = [k for k in ("weather", "time", "intensity", "kind", "name") if k in condition]
        if keys:
            return ", ".join(f"{k}={condition[k]}" for k in keys)
        return json.dumps(condition, sort_keys=True)
    return str(condition)


def _key(condition) -> str:
    if isinstance(condition, dict):
        return json.dumps(condition, sort_keys=True)
    return str(condition)


def _pick_condition(failure_map: dict, conditions: list):
    """Highest observed failure rate, else the first condition with no arm yet."""
    if not conditions:
        return None
    arms = failure_map.get("arms") if isinstance(failure_map, dict) else None
    by_key = {}
    for arm in arms or []:
        if isinstance(arm, dict) and "condition" in arm:
            by_key[_key(arm["condition"])] = arm

    best = None
    best_rate = -1.0
    for cond in conditions:
        arm = by_key.get(_key(cond))
        if arm is None:
            continue
        try:
            rate = float(arm.get("failure_rate") or 0)
        except (TypeError, ValueError):
            rate = 0.0
        if rate > best_rate:
            best_rate = rate
            best = cond
    if best is not None:
        return best
    for cond in conditions:
        if _key(cond) not in by_key:
            return cond
    return conditions[0]


def _fallback(failure_map: dict, conditions: list, why: str) -> dict:
    condition = _pick_condition(failure_map, conditions)
    if condition is None:
        rationale = f"{why} No conditions were provided."
    else:
        rationale = (
            f"{why} Displaying {_label(condition)}: "
            "the observed arm with the highest failure rate, "
            "or the first condition that has not been tested yet."
        )
    return {"condition": condition, "rationale": rationale}


def _parse_proposal(text: str, conditions: list) -> dict | None:
    raw = (text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return None
    condition = data.get("condition")
    if conditions:
        match = None
        for cond in conditions:
            if cond == condition or _key(cond) == _key(condition):
                match = cond
                break
            if isinstance(condition, str) and condition in (_label(cond), _key(cond)):
                match = cond
                break
        if match is None:
            return None
        condition = match
    return {"condition": condition, "rationale": rationale.strip()}


@tracing.traced
def propose_next_condition(failure_map: dict, conditions: list) -> dict:
    """Return {"condition", "rationale"} for display.

    Thompson sampling picks the numeric arm elsewhere. With no WANDB_API_KEY
    (or if the call fails) this returns a deterministic rationale and does not
    raise.
    """
    if not _api_key():
        return _fallback(failure_map, conditions, "No WANDB_API_KEY.")

    messages = [
        {
            "role": "system",
            "content": (
                "You justify the next video condition to test in a failure-seeking loop. "
                "Thompson sampling makes the numeric choice elsewhere; you only recommend "
                "one condition from the provided list for a demo display and explain why, "
                "using the failure map (higher failure_rate and low n are more informative). "
                'Reply with JSON only: {"condition": <one condition from the list>, '
                '"rationale": "<one or two sentences>"}.'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"failure_map": failure_map, "conditions": conditions},
                default=str,
            ),
        },
    ]
    try:
        text = chat(messages, json_mode=True)
        parsed = _parse_proposal(text, conditions)
    except Exception:
        return _fallback(failure_map, conditions, "W&B Inference call failed.")
    if parsed is None:
        return _fallback(failure_map, conditions, "Model did not return usable JSON.")
    return parsed
