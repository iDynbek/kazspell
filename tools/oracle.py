"""Every form apertium-kaz builds for a lemma, against every form we build.

`check_template.py` compares the two grammars as *inventories* — which suffixes
each knows about. This compares them as *machines*: apertium ships its
transducer expanded, one lemma of each part of speech taken through every
paradigm it has, 13,051 forms with no clitic or copula. Each one is a word
apertium is willing to generate together with the analysis it generates it
from, which makes it the one source here that can say what is missing rather
than only what is wrong.

It found two families the corpus then confirmed. The қалау рай derives a
nominal and takes a possessive — `көргісі келеді`, in 530 of the 3,860
editions, `айтқысы` 526, `барғым` 205 — and it was filed with the moods that
do not, so every verb-only stem lost the construction. And the `-н-` cases
follow `-дікі` as well as a possessive: `менікін` 54 books, `менікінен` 50.

The oracle is not the authority, though, and this is why every miss is looked
up in the books before it is called a gap. apertium generates
`жақсылардікінге`, and Kazakh does not write it: `менікінге` is in none of the
3,860 editions against 19 for `менікіне`. A form neither we nor the corpus has
is apertium being generous, not us being wrong.

    python tools/oracle.py
    python tools/oracle.py --show 40      # more of each cluster
"""

from __future__ import annotations

import argparse
import collections
import gzip
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from analyse import default  # noqa: E402

CYRILLIC = re.compile(r"^[а-яёәғқңөұүһі]+$")
# Tags that name the lemma's own part of speech rather than anything added to
# it, so that forms differing only in which lemma they come from cluster.
LEXICAL = {"n", "v", "adj", "prn", "tv", "iv", "subst", "np", "cop"}


def read_expansion(path: Path) -> list[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    out = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            surface, _, analysis = line.rstrip("\n").partition(":")
            # `+` joins a clitic or a copula to the form. Those are separate
            # words here, and comparing them would be comparing a decision
            # about tokenisation rather than about morphology.
            if analysis and "+" not in analysis and CYRILLIC.match(surface):
                out.append((surface, analysis))
    return out


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expansion", type=Path,
                    default=ROOT.parent / "apertium-kaz/expanded/current-state.txt.gz")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/oracle_gaps.tsv")
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    if not args.expansion.exists():
        sys.exit(f"no expansion at {args.expansion}; it ships in apertium-kaz "
                 f"under expanded/")
    forms = read_expansion(args.expansion)
    attested = read_attested(args.attested)
    an = default()

    lemmas = collections.Counter(a.split("<")[0] for _s, a in forms)
    print(f"{len(forms):,} forms over {len(lemmas)} lemmas: "
          f"{', '.join(sorted(lemmas))}")

    built, gaps, generous = 0, [], []
    clusters: dict[str, list[str]] = collections.defaultdict(list)
    totals: collections.Counter = collections.Counter()
    for surface, analysis in forms:
        tags = re.findall(r"<([a-z0-9_]+)>", analysis)
        key = ".".join(t for t in tags if t not in LEXICAL) or "bare"
        totals[key] += 1
        if an.accepts(surface):
            built += 1
            continue
        clusters[key].append(surface)
        (gaps if attested.get(surface, 0) else generous).append((surface, analysis))

    print(f"\nwe build {built:,} of {len(forms):,} ({built / len(forms):.1%})")
    print(f"  {len(gaps):,} we refuse and the books do write — ours to fix")
    print(f"  {len(generous):,} we refuse and no book writes either — "
          f"apertium being generous")

    args.output.write_text(
        "# form\tbooks\tanalysis — tools/oracle.py\n"
        "# forms apertium-kaz generates that this recogniser refuses.\n"
        "# `books` is how many of the 3,860 editions write it: 0 means the\n"
        "# disagreement is probably apertium's rather than ours.\n"
        + "\n".join(f"{s}\t{attested.get(s, 0)}\t{a}"
                    for s, a in sorted(gaps + generous,
                                       key=lambda p: -attested.get(p[0], 0)))
        + "\n", encoding="utf-8")

    if gaps:
        print("\nrefused, and attested — take these first:")
        for surface, analysis in sorted(gaps, key=lambda p: -attested[p[0]])[:args.show]:
            print(f"  {surface:<24}{attested[surface]:>5} books   {analysis}")

    print("\nby category, worst first:")
    for key, missing in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:args.show]:
        seen = sum(1 for f in missing if attested.get(f, 0))
        print(f"  {len(missing):>4}/{totals[key]:<5} {key[:44]:<46}"
              f"{seen} attested")
    print(f"\n→ {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
