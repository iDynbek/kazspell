"""Entries that appear in no book and hold nothing up.

A wordlist assembled from three sources carries entries none of them should
have offered: acronyms filed as words, headwords that are really inflected
forms, scanning damage that survived into a printed dictionary. They are not
free. Every entry is a stem, so every entry is another way for a misspelling of
a longer word to land on something the recogniser will build.

Deciding which are wrong by looking at them does not work — the ones the leaks
land on are mostly real: `ем` is in 1,791 of the 3,860 editions, `өре` in 421,
`иін` in 378. Short real words catch corruptions of long ones, and that is a
property of the language rather than a defect in the list.

What can be decided is whether an entry is doing anything. An entry that
appears in none of the books, and that no attested form is built on, is
carrying no weight at all: nothing is lost by removing it, and one more stem
stops being available to a misspelling. Both halves are needed. `ушық` appears
in no book either, and `ушығып` is in 132 of them — removing that one costs a
real word, which is what the second test is for.

    python tools/prune.py                # what it would remove
    python tools/prune.py --apply        # rewrite the lexicon
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from analyse import default  # noqa: E402
from phonology import devoice  # noqa: E402


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.tsv")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/pruned.tsv")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the lexicon instead of only listing")
    args = ap.parse_args()

    attested = read_attested(args.attested)
    an = default()
    lines = args.lexicon.read_text(encoding="utf-8").splitlines()
    header = [l for l in lines if l.startswith("#")]
    rows = [l.split("\t") for l in lines if l and not l.startswith("#")]

    idle = {r[0] for r in rows if attested.get(r[0], 0) == 0}
    print(f"{len(idle):,} of {len(rows):,} entries appear in no book",
          file=sys.stderr)

    # Which of them some attested form is actually built on. Taking every
    # reading of every attested type is the only way to ask: an entry can be
    # the stem of a word that shares no prefix with anything else it looks like.
    # A reading names the surface the suffixes attached to, which is not always
    # the entry: `қаңырығы` is built on `қаңырығ` and the entry is `қаңырық`.
    # Undoing the alternations matters — without it the entry looks idle, and
    # `қаңырығы` and `басжібін` were being removed along with it.
    holding = set()
    for i, word in enumerate(attested):
        if i % 100_000 == 0:
            print(f"  {i:,}/{len(attested):,}", end="\r", file=sys.stderr)
        for reading in an.analyse(word):
            for form in (reading[0], devoice(reading[0]),
                         an.elision.get(reading[0])):
                if form in idle:
                    holding.add(form)
    print(file=sys.stderr)

    drop = idle - holding
    keep = [r for r in rows if r[0] not in drop]

    args.output.write_text(
        "# form\tpos\tsources — entries in no book, holding up nothing\n"
        "# tools/prune.py\n"
        + "\n".join(f"{r[0]}\t{r[1] if len(r) > 1 else ''}\t"
                    f"{r[3] if len(r) > 3 else ''}"
                    for r in rows if r[0] in drop) + "\n", encoding="utf-8")

    print(f"{len(holding):,} of them are the stem of something attested — kept")
    print(f"{len(drop):,} are the stem of nothing → {args.output}")
    if args.apply:
        args.lexicon.write_text(
            "\n".join(header + ["\t".join(r) for r in keep]) + "\n",
            encoding="utf-8")
        print(f"{len(keep):,} entries left in {args.lexicon}")
    else:
        print("--apply to remove them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
