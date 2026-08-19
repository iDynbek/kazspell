"""How often each word is written with a capital, and in the middle of a line.

Names are the largest thing this recogniser still refuses, and the reason it
refuses them is that a name list cannot simply be admitted: 995 of the 23,981
names available sit one harmony edit from an ordinary word, and they are the
commonest words in the language — `болган` against `болған`, `ким` against
`кім`, `жана` against `жаңа`. Admitting those makes the commonest misspelling
of the commonest words impossible to flag.

Telling a name from a word needs the case that `data/attested.tsv.gz` threw
away: it is folded, so `Тұрсын` the name and `тұрсың` "you stand" are one row.
kazdict keeps its citation quotes verbatim, 292,305 of them over 2,515 works,
and that is running Kazakh with its capitals intact.

A capital at the start of a sentence says nothing, so only mid-sentence tokens
are counted. A word that is capitalised there and almost nowhere else is a
name; one that is capitalised sometimes is a word that also starts sentences in
quoted fragments.

    python tools/build_case.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORD = re.compile(r"[А-ЯЁӘҒҚҢӨҰҮҺІа-яёәғқңөұүһі]+")
# A token opening a sentence is capitalised by punctuation, not by being a
# name, so it carries no evidence either way.
OPENS = re.compile(r"(?:^|[.!?…»\"]\s+|\(\s*)$")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path,
                    default=ROOT.parent / "kazdict/data/build/kazdict.db")
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "data/case.tsv.gz")
    args = ap.parse_args()

    if not args.db.exists():
        sys.exit(f"no kazdict at {args.db}")
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT quote, work_title FROM citation "
        "WHERE quote IS NOT NULL AND length(quote) > 0")

    upper: collections.Counter = collections.Counter()
    lower: collections.Counter = collections.Counter()
    works: dict[str, set] = collections.defaultdict(set)
    quotes = 0
    for quote, work in rows:
        quotes += 1
        for match in WORD.finditer(quote):
            token = match.group(0)
            folded = token.lower()
            if OPENS.search(quote[:match.start()]):
                continue          # sentence-initial: no evidence
            if token[0].isupper():
                upper[folded] += 1
            else:
                lower[folded] += 1
            works[folded].add(work)

    seen = set(upper) | set(lower)
    with gzip.open(args.output, "wt", encoding="utf-8") as fh:
        fh.write("# form\tcapitalised\tlower-case\tworks — tools/build_case.py\n")
        for form in sorted(seen):
            fh.write(f"{form}\t{upper[form]}\t{lower[form]}\t{len(works[form])}\n")

    names = sum(1 for f in seen if upper[f] >= 3 and upper[f] >= lower[f] * 9)
    print(f"{quotes:,} citation quotes, {len(seen):,} distinct forms "
          f"→ {args.output}")
    print(f"  {names:,} are capitalised mid-sentence at least 3 times and "
          f"at least 9:1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
