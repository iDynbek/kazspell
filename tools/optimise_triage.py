"""Let GEPA write the triage prompt, and score it on answers we already know.

The prompt in `tools/triage.py` is hand-written, and the first version of it was
measurably bad: it told the model to be strict, and the model duly refused
`немере`, `бедер` and `жусан`. Hand-tuning that is guessing with a slow feedback
loop, and there is a better one available — the control set. Every string in it
has a known answer, so a prompt can be *scored*, and anything scorable can be
optimised.

GEPA does that by mutation and reflection: it runs a candidate prompt, reads the
cases it got wrong, asks a model why, and writes a new prompt from the answer.
What makes it work here is that the feedback can be concrete — not "72%" but
"you called `бедер` a fragment; it is an ordinary noun" — so `make_reflective_dataset`
below spells out every mistake by name.

The honesty rules are the same as everywhere else in this project:

    train, validate, test    GEPA sees the train and validation halves. The
                             score reported at the end is on a third slice it
                             never saw, because a prompt tuned on 200 strings
                             will fit 200 strings.
    the controls are real    positives are entries apertium hand-filed with a
                             part of speech; negatives are misspellings
                             attested in none of the 3,860 editions
    it optimises a filter    a better prompt still does not decide anything.
                             tools/regress.py does.

    python tools/optimise_triage.py --budget 80
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import gepa  # noqa: E402
from gepa.core.adapter import EvaluationBatch  # noqa: E402

from triage import (SYSTEM, Batch, Limiter, ask_all, build_agent,  # noqa: E402
                    controls, load_env)


@dataclass
class Item:
    """One request: a handful of strings and what each of them really is."""
    words: list[str]
    truth: dict[str, str]


def batched(truth: dict[str, str], size: int, seed: int) -> list[Item]:
    words = list(truth)
    random.Random(seed).shuffle(words)
    return [Item(words[i:i + size], truth)
            for i in range(0, len(words) - size + 1, size)]


def score_one(item: Item, got: dict) -> tuple[float, list[str]]:
    """Balanced accuracy, and the mistakes in words.

    Balanced, because the two errors are not the same error and the control set
    is only half non-words: a prompt that calls everything a word would score
    50% and be useless, and one that refuses everything would score 50% too.
    """
    kinds = {"word": [0, 0], "other": [0, 0]}
    notes = []
    for word in item.words:
        expected = item.truth[word]
        said = got.get(word)
        verdict = said.verdict if said else "no answer"
        right = (verdict == "word") if expected == "word" else (verdict != "word")
        bucket = kinds["word" if expected == "word" else "other"]
        bucket[1] += 1
        bucket[0] += int(right)
        if not right:
            notes.append(
                f"{word!r}: you said {verdict}; it is "
                + ("an ordinary Kazakh word" if expected == "word"
                   else "not a word at all — a misspelling of a real one"))
    halves = [n / d for n, d in kinds.values() if d]
    return (sum(halves) / len(halves) if halves else 0.0), notes


class Adapter:
    """Runs a candidate prompt over a batch and reports what it got wrong."""

    # GEPA tests this for None rather than for existence, and falls back to its
    # own proposer when it is None. Leaving it undefined makes every reflection
    # raise AttributeError, which is caught and reported as "did not propose a
    # new candidate" — the optimiser then runs to completion having done
    # nothing at all, and returns the prompt it started with.
    propose_new_texts = None

    def __init__(self, model_id: str, limiter: Limiter):
        self.model_id = model_id
        self.limiter = limiter
        self.agent = build_agent(model_id)

    def evaluate(self, batch: list[Item], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch:
        instructions = candidate["instructions"]

        async def go():
            out = []
            with self.agent.override(instructions=instructions):
                for item in batch:
                    try:
                        got = await ask_all(self.agent, self.limiter, item.words)
                    except Exception as exc:
                        print(f"    batch failed: {type(exc).__name__} "
                              f"{str(exc)[:90]}", file=sys.stderr)
                        got = {}
                    out.append(got)
            return out

        answers = asyncio.run(go())
        scores, outputs, traces = [], [], []
        for item, got in zip(batch, answers):
            score, notes = score_one(item, got)
            scores.append(score)
            outputs.append({w: (got[w].verdict if w in got else "no answer")
                            for w in item.words})
            traces.append({"words": item.words, "mistakes": notes,
                           "score": score})
        return EvaluationBatch(outputs=outputs, scores=scores,
                               trajectories=traces if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components):
        rows = []
        for trace, output in zip(eval_batch.trajectories, eval_batch.outputs):
            feedback = ("Every one of these was judged correctly."
                        if not trace["mistakes"] else
                        "You got these wrong:\n" + "\n".join(
                            f"  - {m}" for m in trace["mistakes"]))
            rows.append({
                "Inputs": {"candidates": trace["words"]},
                "Generated Outputs": output,
                "Feedback": feedback + f"\n\nScore: {trace['score']:.0%} "
                            "(half the strings given are real Kazakh words and "
                            "half are misspellings that appear in none of the "
                            "3,860 books; both halves count equally)",
            })
        return {name: rows for name in components}


def reflector(model_id: str, limiter: Limiter):
    """A plain text-in, text-out model, which is all GEPA asks for."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    import os
    agent = Agent(OpenAIChatModel(model_id, provider=OpenRouterProvider(
        api_key=os.environ["OPENROUTER_API_KEY"])), output_type=str)

    def call(prompt: str) -> str:
        async def go():
            await limiter.take()
            return (await agent.run(prompt)).output
        for attempt in range(4):
            try:
                return asyncio.run(go())
            except Exception as exc:
                limiter.refused()
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""
    return call


def report(adapter: Adapter, items: list[Item], prompt: str, label: str) -> float:
    result = adapter.evaluate(items, {"instructions": prompt})
    mean = sum(result.scores) / len(result.scores)
    print(f"  {label:<26}{mean:>6.1%}")
    return mean


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b:free")
    ap.add_argument("--reflection-model",
                    default="nvidia/nemotron-3-ultra-550b-a55b:free")
    ap.add_argument("--controls", type=int, default=300,
                    help="labelled strings, split three ways")
    ap.add_argument("--per-request", type=int, default=10)
    ap.add_argument("--budget", type=int, default=80,
                    help="how many candidate evaluations GEPA may spend")
    ap.add_argument("--rpm", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "data/triage_prompt.txt")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    truth = controls(args.controls, args.seed)
    items = batched(truth, args.per_request, args.seed)
    if len(items) < 6:
        sys.exit("not enough controls to split three ways")
    third = len(items) // 3
    train, val, test = items[:third], items[third:2 * third], items[2 * third:]
    print(f"{len(truth):,} controls → {len(train)} train, {len(val)} validation, "
          f"{len(test)} held-out requests of {args.per_request}", file=sys.stderr)

    limiter = Limiter(args.rpm, cap=10_000)
    adapter = Adapter(args.model, limiter)

    print("\nbefore, on the held-out third:")
    before = report(adapter, test, SYSTEM, "hand-written prompt")

    result = gepa.optimize(
        seed_candidate={"instructions": SYSTEM},
        trainset=train, valset=val, adapter=adapter,
        reflection_lm=reflector(args.reflection_model, limiter),
        max_metric_calls=args.budget, display_progress_bar=False,
        seed=args.seed, raise_on_exception=False,
    )
    best = result.best_candidate["instructions"]

    print("\nafter, on the same held-out third:")
    after = report(adapter, test, best, "GEPA's prompt")
    strings = sum(len(i.words) for i in test)
    print(f"\n{after - before:+.1%} on {strings} strings it never saw, "
          f"in {limiter.used} requests")

    # The model is not deterministic, so two runs of the same prompt differ.
    # On a hundred strings that is worth a couple of points, and a difference
    # smaller than that is the temperature talking rather than the prompt.
    margin = max(0.03, 1.0 / max(strings, 1) ** 0.5)
    if best == SYSTEM:
        print("GEPA returned the prompt it was given; nothing to keep")
    elif after - before < margin:
        print(f"inside the noise for {strings} strings (±{margin:.0%}); "
              "the hand-written prompt stands")
    else:
        args.output.write_text(best, encoding="utf-8")
        print(f"kept → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
