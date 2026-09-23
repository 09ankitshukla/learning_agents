"""LLM-as-judge: using a model to score what code cannot.

Lesson 7's scorers are deterministic, and that ordering was deliberate. Free,
instant, reproducible, unbiased checks should be exhausted before a model is asked
to grade a model. What they cannot reach is anything subjective:

    "is this explanation clear?"
    "did it cite the source it actually used?"
    "is this refusal appropriately worded, or does it merely contain the word
     'cannot' while still asserting a made-up number?"

Lesson 7 left that gap open on purpose. `declined()` says so in its own docstring:
it detects refusal *language* by keyword, and an agent could refuse in words it
misses or say "cannot" while fabricating anyway.

Three commitments this file tries to keep, because a careless judge is worse than
no judge -- it produces numbers that look like measurement and are not.

**The judge returns structured output, validated.** Lesson 1's pattern:
describe with a schema, extract, validate, repair. A judge that returns prose you
then regex for "PASS" is a second unreliable component stacked on the first.

**The judge must be calibrated before it is trusted.** Run it on cases whose
answer is already known deterministically and measure agreement. That is what
`observe.py --calibrate` does, and it is the first thing to build, not the last.

**The judge must never see the expected answer** when grading quality. Give it the
answer key and it will agree with the key rather than assess the work -- a
subtle way to measure nothing while appearing rigorous.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from llmkit import LLMClient, system, user

CACHE_DIR = Path(__file__).parent / ".cache"

#: Judges need room to reason before answering (lesson 0). Too tight a budget
#: produces an empty verdict, which the caller then has to treat as... what?
#: A failure? An error? Avoid the question by budgeting properly.
JUDGE_MAX_TOKENS = 1200


class Verdict(BaseModel):
    """A judge's decision, as structured data rather than prose.

    Field order matters more than it looks. `reasoning` comes first so the model
    produces its justification *before* committing to a verdict -- a decision
    written first tends to be rationalised afterwards rather than derived.
    """

    reasoning: str = Field(
        description=(
            "One or two sentences explaining your assessment, referring to specific "
            "features of the answer. Write this before deciding."
        )
    )
    passed: bool = Field(description="True if the answer satisfies every criterion.")
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "How sure you are. Use 'low' when the criteria are ambiguous or the "
            "answer is borderline."
        )
    )

    #: Populated by the caller, not the model.
    failure_kind: str | None = None


@dataclass
class JudgeResult:
    verdict: Verdict | None
    raw: str | None
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    from_cache: bool = False
    error: str | None = None

    @property
    def passed(self) -> bool:
        """A judge that failed to produce a verdict does not get to pass anything.

        Defaulting an unparseable verdict to True would silently inflate every
        score; defaulting to False at least fails loudly and visibly.
        """
        return bool(self.verdict and self.verdict.passed)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# The rubric
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
    "You are a careful evaluator of an AI agent's answers. You are given the "
    "question the agent was asked, the answer it produced, and explicit criteria.\n\n"
    "Judge only against the criteria. Do not reward length, confidence, formatting "
    "or fluency. A short correct answer is better than a long one.\n"
    "An answer that states something it could not know is a failure, even if it "
    "sounds reasonable.\n"
    "An answer that correctly says it cannot do something is a PASS when the "
    "criteria ask for that.\n\n"
    "Return a single JSON object with exactly these keys: reasoning (string), "
    "passed (boolean), confidence (one of \"high\", \"medium\", \"low\").\n"
    "Output JSON only. No prose, no markdown fences."
)


def build_prompt(question: str, answer: str | None, criteria: list[str]) -> str:
    """Assemble the judging request.

    Note what is absent: the expected answer. The judge assesses the work against
    criteria, not against a key. Showing it the answer turns the judge into an
    agreement machine.
    """
    criteria_text = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))
    shown = answer if answer else "(the agent produced no answer)"
    return (
        f"QUESTION THE AGENT WAS ASKED:\n{question}\n\n"
        f"THE AGENT'S ANSWER:\n{shown}\n\n"
        f"CRITERIA -- every one must be satisfied:\n{criteria_text}\n\n"
        f"Assess the answer against these criteria and return your JSON verdict."
    )


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_key(model: str, question: str, answer: str | None, criteria: list[str]) -> str:
    """Judging is a model call, so it is cached on the same reasoning as lesson 7.

    Note the difference from lesson 7's rule. There, the *score* must never be
    cached because scoring is cheap and its logic changes. Here the judgement IS a
    model call -- expensive, and the thing the cache exists for. The key includes
    the criteria, so editing a rubric correctly invalidates it.
    """
    digest = hashlib.sha256()
    for part in (model, question, answer or "", "||".join(criteria)):
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()[:20]


def _load_cached(key: str) -> JudgeResult | None:
    path = CACHE_DIR / f"judge_{key}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        verdict = Verdict(**raw["verdict"]) if raw.get("verdict") else None
        return JudgeResult(
            verdict=verdict,
            raw=raw.get("raw"),
            attempts=raw.get("attempts", 1),
            prompt_tokens=raw.get("prompt_tokens", 0),
            completion_tokens=raw.get("completion_tokens", 0),
            from_cache=True,
        )
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError):
        return None


def _store_cached(key: str, result: JudgeResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": result.verdict.model_dump() if result.verdict else None,
        "raw": result.raw,
        "attempts": result.attempts,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }
    (CACHE_DIR / f"judge_{key}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------
#: A judge prompt WITHOUT the anti-bias clauses, for the bias experiment.
#:
#: The difference between this and JUDGE_SYSTEM_PROMPT is three sentences telling
#: the judge to ignore length, confidence and fluency. Running the same probe
#: against both is how you find out whether those sentences do anything -- and they
#: turn out to be the difference between a biased judge and a usable one.
NAIVE_JUDGE_SYSTEM_PROMPT = (
    "You are evaluating an AI agent's answer. You are given the question, the "
    "answer, and some criteria.\n"
    "Decide whether the answer is good.\n"
    "Return a single JSON object with keys: reasoning (string), passed (boolean), "
    'confidence (one of "high", "medium", "low"). Output JSON only.'
)

NAIVE_PAIRWISE_SYSTEM_PROMPT = (
    "You compare two answers to the same question and decide which is better.\n"
    "Return a single JSON object with keys: reasoning (string), winner (one of "
    '"A", "B", "tie"). Output JSON only.'
)


def judge_answer(
    client: LLMClient,
    question: str,
    answer: str | None,
    criteria: list[str],
    *,
    max_attempts: int = 3,
    use_cache: bool = True,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> JudgeResult:
    """Ask a model to grade an answer against explicit criteria.

    The describe/extract/validate/repair loop is lesson 1's, reused because a judge
    is exactly the case it was built for: a component with a failure rate whose
    output must be typed before your program can act on it.

    temperature=0.0 because a judge that varies its verdict run to run cannot be
    used to compare two agents -- you would be measuring the judge's noise.
    """
    prompt = system_prompt or JUDGE_SYSTEM_PROMPT
    # The rubric is part of the cache key: two different judge prompts are two
    # different instruments and must not share cached verdicts.
    key = _cache_key(client.config.model, question, answer, criteria + [prompt])
    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            return cached

    # Imported here rather than at module scope: lesson 1's directory is added to
    # sys.path by conftest/CLI, so a top-level import would break plain imports.
    from structured import extract_json

    messages = [
        system(prompt),
        user(build_prompt(question, answer, criteria)),
    ]

    prompt_tokens = completion_tokens = 0
    last_raw: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            reply = client.chat(
                messages, temperature=temperature, max_tokens=JUDGE_MAX_TOKENS
            )
        except Exception as exc:  # noqa: BLE001
            return JudgeResult(
                verdict=None,
                raw=None,
                attempts=attempt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )

        prompt_tokens += reply.usage.prompt_tokens
        completion_tokens += reply.usage.completion_tokens
        last_raw = reply.text

        candidate = extract_json(reply.text or "")
        if candidate is None:
            messages += [
                {"role": "assistant", "content": reply.text or ""},
                user("I could not find a JSON object. Reply with only the JSON verdict."),
            ]
            continue

        try:
            verdict = Verdict.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError) as exc:
            messages += [
                {"role": "assistant", "content": reply.text or ""},
                user(f"That JSON was invalid: {str(exc)[:200]}. Return corrected JSON only."),
            ]
            continue

        result = JudgeResult(
            verdict=verdict,
            raw=last_raw,
            attempts=attempt,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if use_cache:
            _store_cached(key, result)
        return result

    return JudgeResult(
        verdict=None,
        raw=last_raw,
        attempts=max_attempts,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error="judge never produced a valid verdict",
    )


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------
class Preference(BaseModel):
    reasoning: str = Field(description="One sentence on why, before choosing.")
    winner: Literal["A", "B", "tie"] = Field(description="Which answer is better.")


PAIRWISE_SYSTEM_PROMPT = (
    "You compare two answers to the same question and decide which is better.\n"
    "Judge only accuracy and whether the answer addresses the question. Ignore "
    "length, formatting, confidence and fluency: a shorter answer that is correct "
    "beats a longer one that is padded.\n"
    "Return a single JSON object with keys: reasoning (string), winner (one of "
    '"A", "B", "tie"). Output JSON only.'
)


def compare_answers(
    client: LLMClient,
    question: str,
    answer_a: str,
    answer_b: str,
    *,
    use_cache: bool = True,
    system_prompt: str | None = None,
) -> tuple[Preference | None, int]:
    """Ask which of two answers is better. Returns (preference, tokens_used).

    Used by the bias experiments. Pairwise comparison is popular because it is
    easier for a model than absolute scoring -- and it introduces **position
    bias**, where the answer presented first wins regardless of content.

    The only honest way to use it is to run both orders and check the verdicts are
    mirror images. `observe.py --bias` does exactly that.
    """
    from structured import extract_json

    prompt = system_prompt or PAIRWISE_SYSTEM_PROMPT
    key = _cache_key(
        client.config.model, question, f"A={answer_a}||B={answer_b}", ["pairwise", prompt]
    )
    if use_cache:
        path = CACHE_DIR / f"pair_{key}.json"
        if path.exists():
            try:
                return Preference(**json.loads(path.read_text(encoding="utf-8"))), 0
            except (OSError, json.JSONDecodeError, ValidationError):
                pass

    reply = client.chat(
        [
            system(prompt),
            user(
                f"QUESTION:\n{question}\n\n"
                f"ANSWER A:\n{answer_a}\n\n"
                f"ANSWER B:\n{answer_b}\n\n"
                f"Which is better? Return your JSON verdict."
            ),
        ],
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    candidate = extract_json(reply.text or "")
    if candidate is None:
        return None, reply.usage.total_tokens
    try:
        preference = Preference.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError):
        return None, reply.usage.total_tokens

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"pair_{key}.json").write_text(
            json.dumps(preference.model_dump(), indent=2), encoding="utf-8"
        )
    return preference, reply.usage.total_tokens


def clear_cache() -> int:
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.json"))
    for path in files:
        path.unlink()
    return len(files)
