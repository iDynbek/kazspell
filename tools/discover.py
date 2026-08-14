"""Stems the books inflect that the lexicon has never heard of.

Coverage is now a vocabulary problem rather than a grammar one. Of the missed
book-weight that is neither Russian nor a name, 48% has no usable stem at all —
no prefix of the word is an entry — so no amount of morphology will reach it.
The three sources the lexicon was built from are exhausted; the corpus is not.

What makes a string a word rather than a typo is that the language *inflects*
it. `процес` appears with `-тер`, `-ке`, `-те` and `-тің`; a misspelling appears
once, in one shape. So a candidate has to show a paradigm: several distinct
walks through the template, over several distinct attested types.

Two things keep that from finding fragments of words already known:

    only rejected types      a type the recogniser already accepts says
                             nothing about a missing entry, and letting them
                             in produced `шығ`, `сөй` and `өмі` — pieces of
                             `шығу`, `сөйле` and `өмір`, each with thirty
                             "patterns" that are all one word oversegmented

    the shortest stem        `процестерге` yields `процес` + `-тер` + `-ге`
                             and not `процестер` + `-ге`, so the nested
                             prefixes of one word stop voting for each other

This writes candidates, not entries. Nothing here goes into the lexicon until
it has been read and measured — `tools/regress.py` shows what each batch does
to the misspellings, which is the side vocabulary can only hurt.

    python tools/discover.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse import default  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ALPHABET = set("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")
KAZAKH_ONLY = set("әғқңөұүһі")
# The same Russian-keyboard substitutions `build_lexicon.py` refuses at
# admission. A candidate one edit from an attested entry is that entry
# misspelt, and admitting it makes the misspelling unflaggable forever.
TRANSLIT = {"у": "ұү", "и": "і", "к": "қ", "г": "ғ", "н": "ң", "а": "ә", "о": "ө"}


def kazakhised(word: str):
    positions = [i for i, c in enumerate(word) if c in TRANSLIT]
    if not positions or len(positions) > 10:
        return
    for r in (1, 2):
        for combo in itertools.combinations(positions, r):
            for choice in itertools.product(*[TRANSLIT[word[i]] for i in combo]):
                out = list(word)
                for i, c in zip(combo, choice):
                    out[i] = c
                yield "".join(out)


def deletions(word: str) -> set[str]:
    return {word[:i] + word[i + 1:] for i in range(len(word))}


def near(word: str, index: set[str]) -> bool:
    """Whether `word` is one edit from something in the index.

    An insertion, a deletion or a substitution all show up as a shared
    delete-one variant, which is why the index holds those rather than the
    words. `аал`, `алл` and `айа` came through the paradigm test with several
    shapes each — the corpus has enough scanning noise to inflect them — and
    each is one letter from a word the lexicon already has.
    """
    return word in index or bool(deletions(word) & index)


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def read_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {line.strip() for line in fh
                if line.strip() and not line.startswith("#")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("--russian", type=Path, default=ROOT / "data/russian.tsv.gz")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/discovered.tsv")
    ap.add_argument("--near-misses", type=Path,
                    default=ROOT / "data/near_misses.tsv",
                    help="candidates one letter from an entry, for review")
    ap.add_argument("--min-docs", type=int, default=2,
                    help="ignore a type in fewer books than this")
    ap.add_argument("--patterns", type=int, default=3,
                    help="distinct slot sequences a candidate must show")
    ap.add_argument("--forms", type=int, default=3,
                    help="distinct attested types a candidate must appear in")
    ap.add_argument("--min-stem", type=int, default=3)
    args = ap.parse_args()

    attested = read_attested(args.attested)
    russian = read_set(args.russian)
    an = default()

    pool = [w for w, n in attested.items()
            if n >= args.min_docs and w not in russian
            and set(w) <= ALPHABET and args.min_stem < len(w) <= 24
            and not an.accepts(w)]
    print(f"{len(pool):,} attested types the recogniser refuses", file=sys.stderr)

    patterns: dict[str, set[tuple]] = collections.defaultdict(set)
    forms: dict[str, set[str]] = collections.defaultdict(set)
    weight: collections.Counter = collections.Counter()
    for word in pool:
        for cut in range(args.min_stem, len(word)):
            stem, rest = word[:cut], word[cut:]
            walks = [p for track in ("n", "v")
                     for p in an._walk(stem, rest, track, None, frozenset())]
            if not walks:
                continue
            # The shortest stem that explains the word is the one that claims
            # it. Anything longer is this stem plus a suffix already counted.
            for path in walks:
                patterns[stem].add(tuple(p.split(":")[0] for p in path))
            forms[stem].add(word)
            weight[stem] += attested[word]
            break

    known = an.lexicon
    # Delete-one variants of every entry, so a candidate can be tested against
    # the whole lexicon at once instead of against each of its 128,769 words.
    index = set(known)
    for entry in known:
        index |= deletions(entry)
    rows, nearby, refused = [], [], collections.Counter()
    for stem, sigs in patterns.items():
        if stem in known:
            continue
        # A candidate the recogniser already builds is not a missing lexeme,
        # it is a word form: `үйде` is `үй` plus `-де` and is written bare in
        # 2,049 books, which made it the strongest-looking candidate of all.
        # Admitting it would let anything be a stem plus a suffix.
        if an.accepts(stem):
            refused["already a word form"] += 1
            continue
        if len(sigs) < args.patterns or len(forms[stem]) < args.forms:
            refused["too little of a paradigm"] += 1
            continue
        if not (set(stem) & KAZAKH_ONLY):
            hit = next((v for v in kazakhised(stem)
                        if v in known and v in attested), None)
            if hit:
                refused["a Russian-keyboard spelling of an entry"] += 1
                continue
        if near(stem, index):
            # Not discarded. A candidate one letter from an entry is usually
            # that entry misspelt, which is why it does not go in — but some
            # are ordinary words that happen to have a near neighbour, and
            # 4,363 of them is a pool worth reading rather than dropping.
            nearby.append((stem, len(sigs), len(forms[stem]), weight[stem],
                           attested.get(stem, 0)))
            refused["one letter from an entry"] += 1
            continue
        rows.append((stem, len(sigs), len(forms[stem]), weight[stem],
                     attested.get(stem, 0)))

    # Whether the candidate is also written on its own. A lexeme usually is —
    # `москва` is in 44 books bare — and a fragment usually is not: `москв`,
    # `кла` and `отба` are in none. It is not a rule, because a stem that only
    # ever appears inflected is a real thing too, so it is a column to sort by
    # rather than a gate to fail.
    # Everything already discovered is in the analyser by now, so a second run
    # finds nothing new — and must not therefore write an empty file over the
    # first run's work. Previous rows are carried through unchanged.
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                stem, p_, f_, w_, b_ = (line.split("\t") + ["0"] * 4)[:5]
                if stem not in {r[0] for r in rows}:
                    rows.append((stem, int(p_), int(f_), int(w_), int(b_)))

    rows.sort(key=lambda r: (-r[4], -r[3]))
    args.output.write_text(
        "# stem\tpatterns\tforms\tbook-weight\tbare — tools/discover.py\n"
        "# candidates, not entries: read them before admitting them.\n"
        "# `bare` is how many books write the stem on its own.\n"
        + "\n".join(f"{s}\t{p}\t{f}\t{w}\t{b}" for s, p, f, w, b in rows) + "\n",
        encoding="utf-8")

    nearby.sort(key=lambda r: (-r[4], -r[3]))
    args.near_misses.write_text(
        "# stem\tpatterns\tforms\tbook-weight\tbare — tools/discover.py\n"
        "# candidates held back for being one letter from an entry. Most are\n"
        "# that entry misspelt; some are words with an unlucky neighbour.\n"
        + "\n".join(f"{s}\t{p}\t{f}\t{w}\t{b}" for s, p, f, w, b in nearby)
        + "\n", encoding="utf-8")

    print(f"{len(rows):,} candidates → {args.output}")
    print(f"{len(nearby):,} held back → {args.near_misses}")
    for why, n in refused.most_common():
        print(f"  {n:>6,} refused: {why}")
    print()
    bare = sum(1 for r in rows if r[4])
    print(f"  {bare:,} of them are also written on their own\n")
    for stem, sigs, forms_n, w, b in rows[:20]:
        print(f"  {stem:<20}{sigs:>3} patterns {forms_n:>4} forms  "
              f"weight {w:>6,}  bare {b:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
