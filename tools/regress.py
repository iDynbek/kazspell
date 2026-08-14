"""What changed its mind since last time, word by word, so it can be checked.

A score moving is not a review. Precision drifted from 97.1% to 96.6% over a
dozen changes to the phonology and the template, and every one of those moves
was a net of two things — misspellings newly let through, which is a fault, and
misspellings newly caught, which is not — with no way to see either. A summary
cannot tell them apart, and the whole point of the corpus-first method is that
each individual claim is checkable.

So the verdicts themselves are the artefact. `data/verdicts.tsv.gz` holds the
recogniser's answer for every probe in the harness — 20,000 attested types and
20,000 misspellings of them — and this prints the ones that differ from it:

    gained   a real word now accepted        good, and the reason should be a rule
    lost     a real word now refused         a regression unless it was never a word
    caught   a misspelling now refused       good
    leaked   a misspelling now accepted      the one to read carefully

`leaked` prints the analysis that let each one in, because that is the question
being asked: which walk accepts this, and should it. Real words are ordered by
how many of the 3,860 editions carry them, so what matters most is read first.

    python tools/regress.py                 # what changed
    python tools/regress.py --accept        # this is right, make it the baseline
"""

from __future__ import annotations

import argparse
import gzip
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse import (Analyser, read_elision, read_harmony,  # noqa: E402
                     read_lexicon)
from measure import build_sample, build_typos, read_attested  # noqa: E402
from template import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def verdicts(an: Analyser, attested: dict[str, int], sample: int, typos: int,
             seed: int, min_docs: int) -> dict[str, tuple[str, str, str]]:
    """probe -> (kind, verdict, context). The whole harness, as data."""
    rng = random.Random(seed)
    words = build_sample(attested, rng, sample, min_docs)
    cases = build_typos(words, attested, rng, typos)

    out: dict[str, tuple[str, str, str]] = {}
    for word in words:
        out[word] = ("real", "ok" if an.accepts(word) else "no",
                     str(attested.get(word, 0)))
    for kind, source, bad in cases:
        # A misspelling that is also in the recall sample is already recorded,
        # and its verdict there is the one that counts.
        out.setdefault(bad, (f"typo:{kind}", "ok" if an.accepts(bad) else "no",
                             source))
    return out


def read_baseline(path: Path) -> dict[str, tuple[str, str, str]]:
    if not path.exists():
        return {}
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                probe, kind, verdict, context = line.rstrip("\n").split("\t")
                out[probe] = (kind, verdict, context)
    return out


def write_baseline(path: Path, rows: dict[str, tuple[str, str, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("# probe\tkind\tverdict\tcontext — tools/regress.py --accept\n")
        for probe in sorted(rows):
            kind, verdict, context = rows[probe]
            fh.write(f"{probe}\t{kind}\t{verdict}\t{context}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--typos", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.tsv")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("--baseline", type=Path, default=ROOT / "data/verdicts.tsv.gz")
    ap.add_argument("--accept", action="store_true",
                    help="record the current verdicts as the baseline")
    ap.add_argument("--show", type=int, default=25,
                    help="how many of each kind to print")
    args = ap.parse_args()

    attested = read_attested(args.attested)
    an = Analyser(load(), read_lexicon(args.lexicon),
                  overrides=read_harmony(ROOT / "data/harmony.tsv"),
                  elision=read_elision(ROOT / "data/elision.tsv"))
    now = verdicts(an, attested, args.sample, args.typos, args.seed,
                   args.min_docs)
    was = read_baseline(args.baseline)

    if not was:
        write_baseline(args.baseline, now)
        print(f"no baseline; recorded {len(now):,} verdicts → {args.baseline}")
        return 0

    gained, lost, caught, leaked = [], [], [], []
    for probe, (kind, verdict, context) in sorted(now.items()):
        before = was.get(probe)
        if before is None or before[1] == verdict:
            continue
        real = kind == "real"
        bucket = ((gained if verdict == "ok" else lost) if real
                  else (leaked if verdict == "ok" else caught))
        bucket.append((probe, kind, context))

    only_now = set(now) - set(was)
    only_was = set(was) - set(now)
    if only_now or only_was:
        print(f"warning: the probe set itself moved — {len(only_now):,} new "
              f"probes, {len(only_was):,} gone. Compare only what is in both.\n",
              file=sys.stderr)

    def report(title: str, rows, note: str, analyse: bool = False) -> None:
        print(f"\n{title}: {len(rows):,}   {note}")
        if not rows:
            return
        rows = sorted(rows, key=lambda r: -int(r[2]) if r[1] == "real" else 0)
        for probe, kind, context in rows[:args.show]:
            where = (f"in {context} books" if kind == "real"
                     else f"{kind.split(':')[1]} of {context}")
            print(f"  {probe:<24}{where}")
            if analyse:
                reading = an.analyse(probe)[:1]
                if reading:
                    print(f"  {'':<24}  {' + '.join(reading[0])}")
        if len(rows) > args.show:
            print(f"  … {len(rows) - args.show:,} more")

    report("gained", gained, "real words now accepted — check each is a rule")
    report("lost", lost, "real words now refused — regressions unless not words")
    report("caught", caught, "misspellings now refused")
    report("leaked", leaked, "misspellings now accepted — read these",
           analyse=True)

    if args.accept:
        write_baseline(args.baseline, now)
        print(f"\nrecorded {len(now):,} verdicts → {args.baseline}")
    elif gained or lost or caught or leaked:
        print("\n--accept to make this the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
