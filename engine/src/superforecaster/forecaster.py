"""The baseline forecasting pass: one Claude call per question, no web search.

This is candidate A of the architecture portfolio, in its simplest form: hand
the question to Claude with a "superforecaster checklist" prompt and read back a
probability, a confidence level, and the reasoning. There is no retrieval yet —
the model works purely from the question text and its own background knowledge,
so the prompt leans hard on the *outside view* (base rates from comparable past
cases) rather than pretending to know today's news.

Key objects:

- :class:`ForecastOutput` — the structured answer we ask Claude to return.
- :class:`ForecastRecord` — that answer plus full bookkeeping (tokens, cost,
  model, timestamps) so every forecast is auditable and cheap to replay.
- :data:`CHECKLIST_PROMPT` — the system prompt. It is written in plain language
  and doubles as documentation of *how* we want the model to think.
- :func:`run_forecast_pass` — glue: build the prompt, call the API, record it.

Some forecasting terms, glossed once:

- **Reference class / base rate**: instead of guessing from scratch, you find a
  group of similar past situations ("how often do infrastructure projects miss
  their deadline?") and start from how often the thing happened in that group.
- **Outside view**: reasoning from that base rate rather than from the vivid
  specifics of this one case, which our intuition tends to over-weight.
- **Calibration**: if you say 70% on many questions, about 70% of them should
  come true. Round numbers (50/50, 90%) are a tell that you stopped thinking, so
  the prompt asks for 1% granularity.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .config import TaskModelConfig, load_task_config, price_for
from .llm import LlmClient, LlmError, LlmResult, build_llm_client
from .models import QuestionSnapshot
from .research import ResearchBundle

Confidence = Literal["low", "medium", "high"]

# Metaculus rejects a forecast of exactly 0 or 1 (you can never be certain), so
# we squeeze every probability into this closed interval before storing it.
_MIN_PROB = 0.01
_MAX_PROB = 0.99


class ForecastOutput(BaseModel):
    """The structured answer Claude returns for one question.

    Attributes:
        probability: Chance the question resolves YES, as a number in [0, 1].
        confidence: How much real signal the model felt it had (not the same as
            the probability): ``"low"`` when it is mostly guessing from a thin
            base rate, ``"high"`` when the outcome is close to settled.
        rationale: The legible argument — reference class, adjustments, and the
            strongest case against the chosen number.
        reference_class: The base-rate anchor the model used, named explicitly so
            we can later check whether it picked a sensible comparison group.
    """

    probability: float = Field(ge=0.0, le=1.0)
    confidence: Confidence
    rationale: str
    reference_class: str


class ForecastRecord(BaseModel):
    """One stored forecast: the answer plus everything needed to audit it.

    ``probability`` is the value we would actually submit — clamped to
    [0.01, 0.99]. ``raw_probability`` preserves exactly what the model said, so a
    clamp is never silently hidden. The token/cost/model fields satisfy the
    cost-visibility rule: no LLM call goes unrecorded.
    """

    created_at: datetime
    question_id: int
    post_id: int
    title: str
    probability: float
    raw_probability: float
    confidence: Confidence
    rationale: str
    reference_class: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    dry_run: bool = True
    submitted_at: datetime | None = None
    # Retrieval bookkeeping. ``research`` is the frozen bundle the forecast was
    # made from (``None`` for the blind baseline). ``research_search_credits``
    # surfaces the metered search draw at the record's top level so the cost
    # story is legible without opening the nested bundle — it is deliberately
    # kept *separate* from ``cost_usd`` (which stays LLM-only): search credits
    # are a different currency from token dollars and must not be conflated.
    research: ResearchBundle | None = None
    research_search_credits: int = 0


# ``{today}`` is filled in per call so the model knows the current date without
# us pretending it can look anything up. Read this prompt as the spec for how a
# careful forecaster should reason; the code just enforces the plumbing.
CHECKLIST_PROMPT = """\
You are a careful superforecaster. Today's date is {today}. You have NO web \
access and cannot look anything up — you reason only from the question text and \
your own background knowledge, which may be out of date. Because of that, lean \
on the outside view: base rates from comparable past cases, not the vivid \
specifics of this one.

Work through this checklist before answering. Your written rationale should show \
each step so a reader can follow and challenge your reasoning.

1. Restate the question. In your own words, say exactly what is being asked and \
what it would take to resolve YES versus NO. Pay attention to the resolution \
criteria and any fine print — many questions are lost by answering a slightly \
different question than the one written.

2. Find a reference class and state a base rate. Name a group of comparable past \
situations and say, roughly, how often the thing in question happened in that \
group. This base rate is your starting anchor. If you are unsure of the exact \
number, give an honest range and pick a point within it.

3. Adjust for the specifics. Move off the base rate for concrete, defensible \
reasons tied to this question — timing, who is involved, how close the deadline \
is, anything in the text that clearly pushes the odds up or down. Make small, \
justified adjustments rather than large intuitive leaps.

4. Argue against yourself. State the single strongest reason your current \
estimate is wrong, and let it move your number if it deserves to. A forecast \
that cannot survive its own best counterargument is not finished.

5. Give a final probability with 1% granularity. Do NOT snap to round numbers \
like 50%, 75%, or 90% out of habit — if the honest answer is 63% or 8%, say so. \
Then state your confidence as low, medium, or high, based on how much real \
signal you actually have: low when you are mostly leaning on a thin base rate, \
high when the outcome is close to already settled. Confidence is about the \
strength of your evidence, not about how extreme the probability is.

Also name the reference class you anchored on, as a short phrase, so it can be \
reviewed on its own.

Remember: with no retrieval available, a humble, well-anchored estimate beats a \
confident guess. When in genuine doubt, stay closer to the base rate."""


# Appended to the system prompt so the model returns something we can parse.
# We have no server-side schema enforcement via the CLI, so structured output is
# achieved by asking for JSON and validating it (with one retry) — see the ADR.
OUTPUT_FORMAT = """\
OUTPUT FORMAT

Write your full reasoning first, then END your response with ONLY a single JSON \
object — nothing after it. You may wrap it in a ```json code fence. It must match \
this schema exactly:

{
  "probability": a number between 0 and 1 — the chance the question resolves YES,
  "confidence": one of "low", "medium", or "high" — how much real signal you \
have, not how extreme the probability is,
  "rationale": a string — your legible argument: reference class, adjustments, \
and the strongest case against your number,
  "reference_class": a string — the short name of the base-rate anchor you used
}

Output nothing after the closing brace of that JSON object."""


# Appended to the system prompt ONLY when a research bundle is supplied. It
# supersedes the checklist's "no web access" framing: the model now has sources
# and must weigh them, cite them, and reason critically about their quality.
RESEARCH_INSTRUCTION = """\
RESEARCH PROVIDED

A SOURCES section accompanies the question below. This SUPERSEDES the "no web \
access" note above: you have been given real research gathered for this \
question. Use it as follows:

- Read every source. Weigh each by quality and recency: prefer authoritative, \
on-topic, recent sources (official bodies, primary reporting) and discount \
low-quality, off-topic, or stale ones.
- Sources can be wrong, biased, or irrelevant. Reason critically — do not simply \
trust a source because it is listed.
- When a source moves your estimate, CITE it by its number (e.g. [S3]) in your \
rationale so a reviewer can trace the evidence.
- Absence of evidence is still information: if the sources do not settle the \
question, say so and stay appropriately humble."""


def _render_sources(research: ResearchBundle) -> str:
    """Render a numbered SOURCES block from a research bundle for the prompt.

    Each result becomes one ``[S<n>]`` entry with its title, source domain, and
    publish date (``date unknown`` when absent). The body shown is the extracted
    document text when we successfully fetched that URL, otherwise the search
    snippet — so the model always sees the fullest content we have per source.
    """
    docs_by_url = {doc.url: doc for doc in research.docs if doc.ok and doc.text}
    lines = ["SOURCES"]
    for index, result in enumerate(research.results, start=1):
        title = result.title.strip() or "(untitled)"
        domain = result.source_domain or "unknown source"
        when = result.published_at.date().isoformat() if result.published_at else "date unknown"
        doc = docs_by_url.get(result.url)
        body = doc.text.strip() if doc is not None else result.snippet.strip()
        if not body:
            body = "(no content extracted)"
        lines.append(f"[S{index}] {title}\n    source: {domain} | published: {when}\n    {body}")
    return "\n\n".join(lines)


def _extract_json_object(text: str) -> str:
    """Pull the JSON object out of the model's reply.

    Preference order, both robust to prose around them:

    1. The **last** ```json ... ``` fenced block, if any.
    2. Otherwise the **last** balanced top-level ``{...}`` object in the text.

    Raises:
        json.JSONDecodeError: If no JSON object can be located at all.
    """
    fenced = _last_fenced_json(text)
    if fenced is not None:
        return fenced
    balanced = _last_balanced_object(text)
    if balanced is not None:
        return balanced
    raise json.JSONDecodeError("no JSON object found in model output", text, 0)


def _last_fenced_json(text: str) -> str | None:
    """Return the contents of the last ```json fenced block, or ``None``."""
    marker = "```json"
    lower = text.lower()
    start = lower.rfind(marker)
    if start == -1:
        return None
    body_start = start + len(marker)
    end = text.find("```", body_start)
    body = text[body_start:end] if end != -1 else text[body_start:]
    return body.strip() or None


def _last_balanced_object(text: str) -> str | None:
    """Return the last balanced top-level ``{...}`` substring, or ``None``.

    Scans once, tracking string literals and escapes so braces inside quoted
    values do not throw off the depth count.
    """
    spans: list[str] = []
    depth = 0
    start: int | None = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
                start = None
    return spans[-1] if spans else None


def _parse_output(text: str) -> ForecastOutput:
    """Extract, JSON-decode, and validate the model's answer.

    Raises:
        json.JSONDecodeError: If no JSON object is present or it is malformed.
        pydantic.ValidationError: If the JSON is present but the wrong shape.
    """
    raw = _extract_json_object(text)
    data = json.loads(raw)
    return ForecastOutput.model_validate(data)


def _build_user_prompt(
    snapshot: QuestionSnapshot,
    research: ResearchBundle | None = None,
) -> str:
    """Assemble the per-question user message from the snapshot's fields.

    Only the fields that help the model reason are included, each under a clear
    header. Empty fields are labelled ``(none provided)`` rather than omitted, so
    the model can tell "no fine print" from "fine print I forgot to include".
    When a research bundle is supplied its SOURCES block is inserted just before
    the final instruction, so the sources read as evidence for the answer.
    """

    def _section(label: str, value: str) -> str:
        text = value.strip() if value else ""
        return f"{label}:\n{text if text else '(none provided)'}"

    def _when(label: str, value: datetime | None) -> str:
        return f"{label}: {value.isoformat() if value else '(unknown)'}"

    parts = [
        _section("QUESTION", snapshot.title),
        _section("DESCRIPTION", snapshot.description),
        _section("RESOLUTION CRITERIA", snapshot.resolution_criteria),
        _section("FINE PRINT", snapshot.fine_print),
        _when("SCHEDULED CLOSE TIME", snapshot.scheduled_close_time),
        _when("SCHEDULED RESOLVE TIME", snapshot.scheduled_resolve_time),
    ]
    if research is not None:
        parts.append(_render_sources(research))
    parts.append("Give your probability that this question resolves YES.")
    return "\n\n".join(parts)


def _clamp(probability: float) -> float:
    """Squeeze a probability into [0.01, 0.99] so Metaculus will accept it."""
    return max(_MIN_PROB, min(_MAX_PROB, probability))


def _retry_note(error: Exception) -> str:
    """The user-message addendum that demands a clean JSON object after a failure."""
    return (
        "\n\nYour previous reply could not be parsed as the required JSON object. "
        f"The error was: {error}\n"
        "Reply again with your full answer and end with ONLY the single JSON "
        "object described in the OUTPUT FORMAT section — no text after the "
        "closing brace, and every field present and correctly typed."
    )


def _call_cost(result: LlmResult, model: str) -> float | None:
    """The cost of one call: the CLI's figure, else the priced-token fallback."""
    if result.cost_usd is not None:
        return result.cost_usd
    return price_for(model, result.input_tokens, result.output_tokens)


def run_forecast_pass(
    snapshot: QuestionSnapshot,
    *,
    client: LlmClient | None = None,
    config: TaskModelConfig | None = None,
    research: ResearchBundle | None = None,
    now: datetime | None = None,
) -> ForecastRecord:
    """Forecast one question with a single Claude call and return a full record.

    Structured output comes from prompting: the model ends its reply with a JSON
    object which we extract and validate. On a parse or validation failure we
    retry exactly once, appending the error and demanding a clean object; a
    second failure raises :class:`LlmError` rather than silently defaulting.

    Args:
        snapshot: The question to forecast (title, criteria, dates, etc.).
        client: Injection seam — any object with a matching ``.run(...)`` method.
            Defaults to the client for the task's configured provider. Tests pass
            a fake.
        config: Model settings; defaults to the ``forecast_pass`` task config.
        research: Optional frozen research bundle. When given, its sources are
            rendered into the prompt and the bundle is stored on the record; when
            ``None`` the model runs blind (the baseline control arm).
        now: Current time (UTC), overridable for deterministic tests.

    Returns:
        A :class:`ForecastRecord` — a dry-run forecast with tokens and cost. When
        a retry happened, tokens are summed and cost is the sum across both calls.

    Raises:
        LlmError: If the model's output cannot be parsed into a valid
            :class:`ForecastOutput` even after one retry.
    """
    task_config = config if config is not None else load_task_config("forecast_pass")
    moment = now if now is not None else datetime.now(UTC)
    llm = client if client is not None else build_llm_client(task_config)

    system = CHECKLIST_PROMPT.replace("{today}", moment.date().isoformat())
    if research is not None:
        system = f"{system}\n\n{RESEARCH_INSTRUCTION}"
    system = f"{system}\n\n{OUTPUT_FORMAT}"
    user_prompt = _build_user_prompt(snapshot, research)

    # Advisory schema for providers that support server-side structured output
    # (Codex CLI constrains its final message to it). Claude CLI ignores it and
    # relies on the prompt-JSON extractor below, which stays the universal net.
    schema = ForecastOutput.model_json_schema()

    results: list[LlmResult] = []
    first = llm.run(
        system,
        user_prompt,
        task_config.model,
        task_config.max_tokens,
        json_schema=schema,
        effort=task_config.effort,
    )
    results.append(first)
    try:
        output = _parse_output(first.text)
    except (json.JSONDecodeError, ValidationError) as first_error:
        retry_prompt = user_prompt + _retry_note(first_error)
        second = llm.run(
            system,
            retry_prompt,
            task_config.model,
            task_config.max_tokens,
            json_schema=schema,
            effort=task_config.effort,
        )
        results.append(second)
        try:
            output = _parse_output(second.text)
        except (json.JSONDecodeError, ValidationError) as second_error:
            raise LlmError(
                f"Forecast model did not return a valid JSON object after one retry: {second_error}"
            ) from second_error

    input_tokens = sum(r.input_tokens for r in results)
    output_tokens = sum(r.output_tokens for r in results)
    per_call = [_call_cost(r, task_config.model) for r in results]
    cost_usd = (
        sum(c for c in per_call if c is not None) if any(c is not None for c in per_call) else None
    )

    return ForecastRecord(
        created_at=moment,
        question_id=snapshot.question_id,
        post_id=snapshot.post_id,
        title=snapshot.title,
        probability=_clamp(output.probability),
        raw_probability=output.probability,
        confidence=output.confidence,
        rationale=output.rationale,
        reference_class=output.reference_class,
        model=task_config.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        dry_run=True,
        submitted_at=None,
        research=research,
        research_search_credits=research.search_credits_used if research is not None else 0,
    )
