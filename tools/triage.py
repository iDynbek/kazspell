"""Ask a language model which candidates are words, and check whether it knows.

`tools/discover.py` leaves 2,440 candidates that the corpus argues for and
nobody has read. The corpus cannot finish the argument: it can say that a string
inflects, which is why the candidates are there, but not whether the thing
inflecting is a Kazakh word or scanning noise that happens to take suffixes. That
is a question about the language, and a model that has read Kazakh can answer it
faster than a person can.

It cannot answer it *reliably*. On the first eight candidates tried, a 550B model
called `мамыражай` a proper name — it is an ordinary adjective — and `аал` a
word, which is `ауыл` misscanned. So nothing here decides anything on its own:

    the model proposes          a verdict and a lemma, per candidate
    the controls measure it     known entries and known non-words are mixed
                                into every batch, so each run reports its own
                                accuracy instead of asking to be trusted
    two models must agree       a candidate is admitted only when both say
                                the same thing, which is what a single model's
                                confidence score cannot give
    tools/regress.py decides    the batch goes in only if the misspellings it
                                lets through stay within budget

Free models are rate-limited upstream and go away without warning, so the run is
resumable: every answer is appended to a checkpoint as it arrives, and starting
again skips what is already there.

    python tools/triage.py --calibrate 120     # how good is it, on known answers
    python tools/triage.py                     # triage the candidates
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

MODELS = ["nvidia/nemotron-3-ultra-550b-a55b:free", "google/gemma-4-31b-it:free"]

VERDICTS = ("word", "name", "fragment", "foreign")

SYSTEM = """\
You are a Kazakh lexicographer deciding what belongs in a spelling dictionary.

The test is whether a Kazakh writer could use the string and be spelling
correctly. Give exactly one verdict for each:
  word      anything that belongs in the dictionary, whatever its origin.
            A borrowing that Kazakh writes and inflects is a word: блокнот,
            хаос, карбюратор, норматив and компартия are all `word`.
  name      a proper name: a person, place, organisation or title
  fragment  not a spelling of anything: a truncation, a misscan, a typo, or a
            piece of a word. `кішмү` and `мархұ` are fragments.
  foreign   a Russian word appearing as Russian, which a Kazakh speller should
            not accept. Use this only when the string is not used in Kazakh.

Then give the dictionary form (lemma) and the part of speech, if it is a word.
Part of speech is one of: n, adj, adv, v, num, pron, post, interj, abbr.

Many candidates are fragments that happen to take suffixes, and many are
ordinary words you should recognise. Neither answer is the safe one — say what
you actually judge. If you are unsure whether a string is Kazakh or noise, weigh
whether it has a plausible Kazakh shape and meaning.

Answer for every candidate you are given, once each, and for no others."""


class Judgement(BaseModel):
    word: str
    verdict: str = Field(description="word, name, fragment or foreign")
    lemma: str = ""
    pos: str = ""


class Batch(BaseModel):
    items: list[Judgement]


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.strip().partition("=")
                os.environ.setdefault(key, value)


def build_agent(model_id: str):
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    model = OpenAIChatModel(model_id, provider=OpenRouterProvider(
        api_key=os.environ["OPENROUTER_API_KEY"]))
    return Agent(model, output_type=Batch, system_prompt=SYSTEM, retries=2)


class Limiter:
    """One request every `interval` seconds, and never after the daily cap.

    A free model is rate-limited by the provider rather than by the account, so
    the useful response to a refusal is to wait longer rather than to fail: the
    delay doubles on every 429 and decays back once requests are landing again.
    """

    def __init__(self, rpm: float, cap: int, concurrency: int = 1):
        self.interval = 60.0 / max(rpm, 0.1)
        self.cap = cap
        self.used = 0
        self.last = 0.0
        self.penalty = 0.0
        self.lock = asyncio.Lock()
        # A free model answers a batch in roughly a minute, so requests sent
        # one after another are limited by latency long before they are limited
        # by anyone's rate limit: a run that used no more than 14 requests a
        # minute of its allowance was managing four. The pacing above still
        # applies; this is how many may be in the air while it does.
        self.slots = asyncio.Semaphore(max(1, concurrency))

    async def take(self) -> None:
        if self.used >= self.cap:
            raise RuntimeError(f"daily cap of {self.cap} requests reached")
        async with self.lock:
            wait = self.last + self.interval + self.penalty - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = time.monotonic()
            self.used += 1
            self.penalty = max(0.0, self.penalty * 0.5 - 1.0)

    def refused(self) -> None:
        self.penalty = min(120.0, max(5.0, self.penalty * 2))


async def ask(agent, limiter: Limiter, words: list[str],
              attempts: int = 4) -> dict[str, Judgement]:
    prompt = "Candidates:\n" + "\n".join(f"- {w}" for w in words)
    for attempt in range(attempts):
        await limiter.take()
        try:
            async with limiter.slots:
                result = await agent.run(prompt)
        except Exception as exc:
            if "429" in str(exc) or "rate" in str(exc).lower():
                limiter.refused()
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(2 ** attempt)
            continue
        wanted = set(words)
        return {j.word: j for j in result.output.items if j.word in wanted}
    return {}


async def ask_all(agent, limiter: Limiter, words: list[str]) -> dict[str, Judgement]:
    """Ask, then ask again for whatever came back unanswered.

    A model given twenty candidates routinely returns eighteen, and the two it
    dropped were being recorded as refusals — a quarter of the apparent errors
    in the first calibration were words it had never been asked about twice.
    """
    got = await ask(agent, limiter, words)
    missing = [w for w in words if w not in got]
    if missing and len(missing) < len(words):
        got |= await ask(agent, limiter, missing)
    return got


def controls(n: int, seed: int) -> dict[str, str]:
    """Strings whose answer is already known, to score the model against.

    Positives are entries apertium hand-filed with a part of speech, so they are
    words on someone else's authority rather than ours. Negatives are
    misspellings of attested words that are themselves attested nowhere, so they
    are not words by construction. Nothing here is a candidate.
    """
    from measure import neighbours, read_attested, typos
    attested = read_attested(ROOT / "data/attested.tsv.gz")
    rng = random.Random(seed)

    good = []
    for line in (ROOT / "data/lexicon.tsv").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        form, pos, paradigms = (line.split("\t") + ["", ""])[:3]
        if paradigms and pos not in ("?", "") and "np" not in pos \
                and len(form) > 3 and attested.get(form, 0) >= 5:
            good.append(form)

    keys = neighbours()
    bad, pool = [], [w for w in good]
    rng.shuffle(pool)
    for word in pool:
        for _kind, wrong in typos(word, keys):
            if wrong not in attested and len(wrong) > 3:
                bad.append(wrong)
                break
        if len(bad) >= n // 2:
            break

    rng.shuffle(good)
    return ({w: "word" for w in good[:n - n // 2]}
            | {w: "fragment" for w in bad[:n // 2]})


def read_checkpoint(path: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["model"]][row["word"]] = row
    return out


def record(log, done, model_id: str, chunk: list[str], got: dict,
           deferred: list[str]) -> None:
    """Write down the answers, and set aside the candidates without one.

    Silence is not a verdict. Recording an unanswered candidate as `missing`
    would make every later run skip it, which is how a quarter of the first
    calibration's apparent refusals were produced.
    """
    for word in chunk:
        judgement = got.get(word)
        if judgement is None:
            deferred.append(word)
            continue
        done[model_id][word] = {
            "model": model_id, "word": word, "verdict": judgement.verdict,
            "lemma": judgement.lemma, "pos": judgement.pos}
        log.write(json.dumps(done[model_id][word], ensure_ascii=False) + "\n")


async def run(models: list[str], words: list[str], limiter: Limiter,
              checkpoint: Path, size: int, retries: int = 2,
              give_up: int = 4) -> dict[str, dict[str, dict]]:
    done = read_checkpoint(checkpoint)
    with checkpoint.open("a", encoding="utf-8") as log:
        for model_id in models:
            todo = [w for w in words if w not in done.get(model_id, {})]
            if not todo:
                print(f"{model_id}: already complete", file=sys.stderr)
                continue
            agent = build_agent(model_id)
            print(f"{model_id}: {len(todo):,} to do, "
                  f"{len(todo) // size + 1} requests", file=sys.stderr)
            # A model that is refused upstream is refused for the whole run,
            # and asking it four hundred more times spends the allowance that
            # the model still answering needs.
            failures = 0
            chunks = [todo[i:i + size] for i in range(0, len(todo), size)]
            stop, deferred = False, []
            for wave in range(0, len(chunks), limiter.slots._value or 1):
                group = chunks[wave:wave + (limiter.slots._value or 1)]
                results = await asyncio.gather(
                    *(ask_all(agent, limiter, c) for c in group),
                    return_exceptions=True)
                for chunk, got in zip(group, results):
                    if isinstance(got, RuntimeError):
                        print(f"  stopping: {got}", file=sys.stderr)
                        stop = True
                        break
                    if isinstance(got, BaseException):
                        # Not lost: a batch that failed on a transient upstream
                        # refusal is worth asking again once the pass is over
                        # and whatever was throttling it has had time to clear.
                        print(f"  deferred: {type(got).__name__} "
                              f"{str(got)[:90]}", file=sys.stderr)
                        deferred.extend(chunk)
                        failures += 1
                        continue
                    failures = 0
                    record(log, done, model_id, chunk, got, deferred)
                log.flush()
                seen = min((wave + len(group)) * size, len(todo))
                print(f"  {seen:,}/{len(todo):,}  "
                      f"{limiter.used} requests", end="\r", file=sys.stderr)
                if stop:
                    return done
                if failures >= give_up:
                    print(f"  {model_id} has refused {failures} batches in a "
                          f"row; leaving it for another day", file=sys.stderr)
                    break
            print(file=sys.stderr)

            # The deferred pass. Everything the main pass could not get an
            # answer for is asked again in smaller batches — a model that drops
            # four candidates out of twelve usually answers all four when they
            # are the only ones in front of it. Whatever is still unanswered is
            # deliberately not written down, so the next run asks about it
            # rather than treating silence as a verdict.
            for attempt in range(retries):
                if not deferred:
                    break
                small = max(2, size // (2 * (attempt + 1)))
                waiting, deferred = deferred, []
                print(f"  deferred pass {attempt + 1}: {len(waiting):,} left, "
                      f"batches of {small}", file=sys.stderr)
                chunks = [waiting[i:i + small]
                          for i in range(0, len(waiting), small)]
                width = limiter.slots._value or 1
                for wave in range(0, len(chunks), width):
                    group = chunks[wave:wave + width]
                    results = await asyncio.gather(
                        *(ask_all(agent, limiter, c) for c in group),
                        return_exceptions=True)
                    for chunk, got in zip(group, results):
                        if isinstance(got, RuntimeError):
                            print(f"  stopping: {got}", file=sys.stderr)
                            return done
                        if isinstance(got, BaseException):
                            deferred.extend(chunk)
                            continue
                        record(log, done, model_id, chunk, got, deferred)
                    log.flush()
            if deferred:
                print(f"  {len(deferred):,} still unanswered; left for the "
                      f"next run", file=sys.stderr)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=ROOT / "data/discovered.tsv")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/triaged.tsv")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "data/triage.jsonl")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--batch", type=int, default=20,
                    help="candidates per request")
    ap.add_argument("--rpm", type=float, default=8.0,
                    help="requests per minute, across all models")
    ap.add_argument("--cap", type=int, default=900,
                    help="stop after this many requests")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="requests in the air at once")
    ap.add_argument("--retries", type=int, default=2,
                    help="deferred passes over whatever went unanswered")
    ap.add_argument("--limit", type=int, help="only the first N candidates")
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="score the models on N strings whose answer is known")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load_env(ROOT / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("no OPENROUTER_API_KEY in the environment or in .env")

    if args.calibrate:
        truth = controls(args.calibrate, args.seed)
        words = list(truth)
        random.Random(args.seed).shuffle(words)
        checkpoint = args.checkpoint.with_suffix(".calibrate.jsonl")
    else:
        truth = {}
        words = [line.split("\t")[0] for line
                 in args.candidates.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.startswith("#")]
        if args.limit:
            words = words[:args.limit]
        checkpoint = args.checkpoint

    limiter = Limiter(args.rpm, args.cap, args.concurrency)
    done = asyncio.run(run(args.models, words, limiter, checkpoint,
                           args.batch, args.retries))
    print(f"{limiter.used} requests used", file=sys.stderr)

    if truth:
        print(f"\ncalibration on {len(words):,} strings whose answer is known")
        for model_id in args.models:
            rows = done.get(model_id, {})
            scored = [(truth[w], rows[w]["verdict"]) for w in words if w in rows]
            if not scored:
                print(f"  {model_id:<44} no answers")
                continue
            # A control is either a word or not; name and foreign are neither
            # here, because no control is one.
            said_word = [(t, v == "word") for t, v in scored]
            tp = sum(1 for t, w in said_word if t == "word" and w)
            fn = sum(1 for t, w in said_word if t == "word" and not w)
            fp = sum(1 for t, w in said_word if t != "word" and w)
            tn = sum(1 for t, w in said_word if t != "word" and not w)
            print(f"  {model_id}")
            print(f"    of {tp + fn:,} real words it kept {tp:,} "
                  f"({tp / max(tp + fn, 1):.0%})")
            print(f"    of {fp + tn:,} non-words it refused {tn:,} "
                  f"({tn / max(fp + tn, 1):.0%})")
        both = [w for w in words if all(w in done.get(m, {}) for m in args.models)]
        agreed = [w for w in both
                  if len({done[m][w]["verdict"] == "word" for m in args.models}) == 1]
        if both:
            kept = [w for w in agreed
                    if done[args.models[0]][w]["verdict"] == "word"]
            right = sum(1 for w in kept if truth[w] == "word")
            print(f"\n  the models agree on {len(agreed):,} of {len(both):,}; "
                  f"of the {len(kept):,} they agree are words, "
                  f"{right:,} really are ({right / max(len(kept), 1):.0%})")
        return 0

    # Agreement is among the models that answered, not among the models that
    # were asked. A free model that is rate-limited upstream all day answers
    # nothing, and requiring its vote turned 4,274 perfectly good verdicts into
    # 4,278 rows marked `disputed`. How many voices there were is a column, so
    # a one-model verdict can be read as the weaker thing it is.
    rows = []
    for word in words:
        said = {m: done[m][word]["verdict"] for m in args.models
                if word in done.get(m, {})}
        if not said:
            verdict = "unanswered"
        elif len(set(said.values())) == 1:
            verdict = next(iter(said.values()))
        else:
            verdict = "disputed"
        lemma = next((done[m][word]["lemma"] for m in args.models
                      if word in done.get(m, {}) and done[m][word]["lemma"]), "")
        pos = next((done[m][word]["pos"] for m in args.models
                    if word in done.get(m, {}) and done[m][word]["pos"]), "")
        rows.append((word, verdict, lemma, pos, str(len(said)),
                     "/".join(f"{m.split('/')[-1]}={v}" for m, v in said.items())))

    args.output.write_text(
        "# stem\tverdict\tlemma\tpos\tvoices\tper-model — tools/triage.py\n"
        "# `verdict` is what every model that answered agreed on, or\n"
        "# `disputed`. `voices` is how many answered: one is weaker than two.\n"
        "# Proposals. tools/regress.py decides whether they go in.\n"
        + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")

    counts = collections.Counter((r[1], r[4]) for r in rows)
    print(f"\n{len(rows):,} candidates → {args.output}")
    for (verdict, voices), n in counts.most_common():
        print(f"  {verdict:<12}{n:>6,}   on {voices} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
