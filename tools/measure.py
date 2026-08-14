"""Score the recogniser on real text and on real misspellings.

Two numbers, and they pull against each other. Recall is how much of the
language it accepts, measured on the 3,860 editions rather than on a wordlist,
weighted by how often each form is actually written. Precision is how much of a
misspelling it catches, measured by corrupting attested forms the way Kazakh
typists corrupt them — the nine letters Cyrillic added for Kazakh sit beside
their Russian neighbours on the keyboard and in the mind.

A misspelling that is itself an attested word is dropped from the test. That
correction was worth 1.4 points to hunspell-kk's reported score and it applies
here for the same reason: corrupting one real form very often lands on another.

Recall is reported twice. The books are Kazakh books and 21.2% of the types in
them are Russian — quoted, cited, code-switched — which the recogniser is right
to refuse and which counts against it anyway. `data/russian.tsv.gz` says which
types those are, and the second number is recall on the rest. Neither figure is
the true one: the first is dragged down by Russian and the second drops some
8,600 words Kazakh and Russian share, `да`, `бар` and `бала` among them. The
truth is between them, and both move when the model does.

    python tools/measure.py --sample 20000
"""

from __future__ import annotations

import argparse
import gzip
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse import (Analyser, read_elision, read_harmony,  # noqa: E402
                     read_lexicon)
from template import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The letters Kazakh added to Cyrillic against what a Russian keyboard, or a
# hurried hand, puts there instead.
CONFUSIONS = [("ә", "а"), ("ө", "о"), ("ұ", "у"), ("ү", "у"), ("і", "и"),
              ("қ", "к"), ("ғ", "г"), ("ң", "н"), ("һ", "х")]
ROWS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]


def neighbours() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in ROWS:
        for i, ch in enumerate(row):
            out[ch] = row[max(0, i - 1):i] + row[i + 1:i + 2]
    return out


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def read_russian(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {line.strip() for line in fh
                if line.strip() and not line.startswith("#")}


def typos(word: str, keys: dict[str, str]) -> list[tuple[str, str]]:
    out = []
    for i, ch in enumerate(word):
        for a, b in CONFUSIONS:
            if ch == a:
                out.append(("confusion", word[:i] + b + word[i + 1:]))
            elif ch == b:
                out.append(("confusion", word[:i] + a + word[i + 1:]))
        for near in keys.get(ch, ""):
            out.append(("keyboard", word[:i] + near + word[i + 1:]))
        out.append(("doubled", word[:i] + ch + word[i:]))
        if len(word) > 3:
            out.append(("dropped", word[:i] + word[i + 1:]))
        if i + 1 < len(word) and word[i] != word[i + 1]:
            out.append(("swapped", word[:i] + word[i + 1] + word[i] + word[i + 2:]))
    return out


def build_sample(attested: dict[str, int], rng: random.Random,
                 size: int, min_docs: int) -> list[str]:
    """The attested types to score, drawn the same way every run."""
    pool = [w for w, n in attested.items() if n >= min_docs and len(w) > 1]
    return rng.sample(pool, min(size, len(pool)))


def build_typos(sample: list[str], attested: dict[str, int],
                rng: random.Random, limit: int) -> list[tuple[str, str, str]]:
    """(kind, the real word, the misspelling) — the same set every run.

    `rng` must be the one `build_sample` was drawn with, and drawn from first:
    the shuffle continues that stream, and a fresh generator would give a
    different set of misspellings and a different score.
    """
    keys = neighbours()
    cases, seen = [], set()
    words = [w for w in sample if len(w) > 3]
    rng.shuffle(words)
    for word in words:
        for kind, bad in typos(word, keys):
            if bad not in attested and bad not in seen and bad != word:
                seen.add(bad)
                cases.append((kind, word, bad))
        if len(cases) >= limit:
            break
    return cases[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=20000,
                    help="attested types to score for recall")
    ap.add_argument("--typos", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-docs", type=int, default=2)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.tsv")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("--misses", type=Path, help="write the rejected forms here")
    ap.add_argument("--russian", type=Path, default=ROOT / "data/russian.tsv.gz",
                    help="attested types a Russian dictionary claims")
    args = ap.parse_args()

    attested = read_attested(args.attested)
    russian = read_russian(args.russian)
    an = Analyser(load(), read_lexicon(args.lexicon),
                  overrides=read_harmony(ROOT / "data/harmony.tsv"),
                  elision=read_elision(ROOT / "data/elision.tsv"))
    rng = random.Random(args.seed)

    sample = build_sample(attested, rng, args.sample, args.min_docs)

    start = time.time()
    accepted, tok_ok, tok_all, missed = 0, 0, 0, []
    kaz_ok, kaz_all, kaz_types, kaz_accepted = 0, 0, 0, 0
    for word in sample:
        weight = attested[word]
        tok_all += weight
        ours = word not in russian
        kaz_all += weight if ours else 0
        kaz_types += 1 if ours else 0
        if an.accepts(word):
            accepted += 1
            tok_ok += weight
            if ours:
                kaz_ok += weight
                kaz_accepted += 1
        else:
            missed.append(word)
    elapsed = time.time() - start

    print(f"recall, on {len(sample):,} attested types "
          f"(in {args.min_docs}+ of 3,860 books)")
    print(f"  by type   {accepted / len(sample):>7.1%}   "
          f"{accepted:,} of {len(sample):,}")
    print(f"  by book-weight {tok_ok / tok_all:>7.1%}")
    if russian:
        print(f"\n  setting aside the {len(sample) - kaz_types:,} types a Russian "
              f"dictionary claims:")
        print(f"  by type   {kaz_accepted / kaz_types:>7.1%}   "
              f"{kaz_accepted:,} of {kaz_types:,}")
        print(f"  by book-weight {kaz_ok / kaz_all:>7.1%}")
    print(f"\n  {len(sample) / max(elapsed, 1e-9):,.0f} words/second")

    cases = build_typos(sample, attested, rng, args.typos)

    caught = sum(1 for _k, _o, bad in cases if not an.accepts(bad))
    print(f"\nprecision, on {len(cases):,} misspellings of those same words")
    print(f"  caught    {caught / len(cases):>7.1%}   {caught:,} of {len(cases):,}")
    by_kind: dict[str, list[int]] = {}
    for kind, _o, bad in cases:
        row = by_kind.setdefault(kind, [0, 0])
        row[1] += 1
        if not an.accepts(bad):
            row[0] += 1
    for kind, (ok, total) in sorted(by_kind.items()):
        print(f"    {kind:<10}{ok / total:>7.1%}  ({ok:,}/{total:,})")

    if args.misses:
        args.misses.write_text("\n".join(missed) + "\n", encoding="utf-8")
        print(f"\n{len(missed):,} rejected forms → {args.misses}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
