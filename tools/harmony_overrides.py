"""Which entries take the harmony column their spelling does not predict.

Vowel harmony is a rule over letters and it has exceptions, all of them in the
same place: `мораль` takes `моральға` and `медаль` takes `медалімен`, and
nothing distinguishes them but usage. `банк` is written with a back vowel and
takes `банкке`; `фильм`, `министр` and `брифинг` take front endings. There is
no rule to find, so the fact is held per lexeme — `phonology.fits` has taken an
`overrides` map from the beginning and nothing has ever filled it.

The books can fill it. Every harmony-bearing suffix comes in two columns, so
asking which spelling of `<entry> + -ға/-ге` is attested, over several suffixes
at once, asks the language directly. An entry is recorded only when the answer
disagrees with what the letters predict and the evidence is not thin.

    python tools/harmony_overrides.py
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phonology import harmony  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The suffixes that differ only in their column, so an attested spelling is
# evidence about the stem and nothing else. Frequent ones, and from four
# different slots, so no single suffix's own oddities can decide an entry.
PROBES = [
    ("лар", "лер"), ("дар", "дер"), ("тар", "тер"),      # көптік
    ("ға", "ге"), ("қа", "ке"),                          # барыс
    ("да", "де"), ("та", "те"),                          # жатыс
    ("дан", "ден"), ("тан", "тен"),                      # шығыс
    ("дың", "дің"), ("тың", "тің"), ("ның", "нің"),      # ілік
    ("ы", "і"), ("ым", "ім"), ("ың", "ің"), ("сы", "сі"),  # тәуелдік
    # And the verbal ones. Without them a verb is judged on whatever nominal
    # suffix its letters happen to collide with, which is how `си` came out
    # back on `сида` while `сиеді` and `сиіп`, 18 books and 70, went unseen.
    ("ған", "ген"), ("қан", "кен"), ("ады", "еді"), ("ып", "іп"),
    ("са", "се"), ("атын", "етін"), ("мады", "меді"), ("бады", "беді"),
]


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def read_entries(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(line.split("\t")[0])
    return out


def evidence(entry: str, attested: dict[str, int],
             headwords: frozenset[str] = frozenset()) -> tuple[int, int]:
    """Book-weight for the back column and for the front one.

    A probe form that is a headword in its own right is not evidence about this
    entry. `жиі` is the adverb "often", in 1,620 of the 3,860 editions, and it
    is not `жи` plus `-і`: counting it made `жи` look front on 1,801 books
    against 2,885, where `жиып` beats `жиіп` 1,669 to 7, `жиған` beats `жиген`
    978 to 2, and `жиса` beats `жисе` 115 to nothing. One homograph outvoted
    the paradigm.

    A loanword in `-ь` is probed without it as well: nothing attaches across a
    soft sign, so `медаль` is written `медалі` and the probes that keep the `ь`
    find nothing at all — which is how it came to have no evidence either way.
    """
    stems = [entry] + ([entry[:-1]] if entry.endswith("ь") else [])
    back = front = 0
    for b, f in PROBES:
        for stem in stems:
            for probe, side in ((stem + b, "back"), (stem + f, "front")):
                if probe in headwords:
                    continue
                if side == "back":
                    back += attested.get(probe, 0)
                else:
                    front += attested.get(probe, 0)
    return back, front


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.tsv")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/harmony.tsv")
    ap.add_argument("--min-books", type=int, default=5,
                    help="ignore an entry with less evidence than this")
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="how far the winning column must be ahead")
    ap.add_argument("--min-length", type=int, default=4,
                    help="shortest entry that may be called ambivalent")
    args = ap.parse_args()

    attested = read_attested(args.attested)
    entries = read_entries(args.lexicon)
    headwords = frozenset(entries)

    rows, agreed, thin = [], 0, 0
    for entry in entries:
        predicted = harmony(entry)
        if predicted is None:
            continue
        back, front = evidence(entry, attested, headwords)
        winner, won, lost = (("back", back, front) if back >= front
                             else ("front", front, back))
        if won < args.min_books:
            thin += 1
            continue
        if won < lost * args.ratio:
            # Both columns are well attested and neither wins. `медаль` is
            # written `медальға` and `медалі`, 120 books of back against 84 of
            # front, and refusing either would be refusing the language. This
            # is a third answer, not a missing one.
            #
            # Not for a very short entry, though. A probe is only evidence if
            # the form it builds is really this entry plus that suffix, and
            # `қа` plus `-лар` finds `қалар`, which is `қал` plus `-ар`. The
            # shorter the entry the more of the language is a false hit, and
            # this verdict is the one that switches a check off rather than
            # moving it, so it is the one that has to be sure.
            if lost >= args.min_books and len(entry) >= args.min_length:
                rows.append((entry, "both", back, front))
            else:
                thin += 1
            continue
        if winner == predicted:
            agreed += 1
            continue
        rows.append((entry, winner, back, front))

    args.output.write_text(
        "# form\tharmony\tback-weight\tfront-weight — tools/harmony_overrides.py\n"
        "# entries whose attested suffixes disagree with what their letters\n"
        "# predict. Everything not listed here follows the rule.\n"
        + "\n".join(f"{w}\t{h}\t{b}\t{f}" for w, h, b, f in rows) + "\n",
        encoding="utf-8")

    print(f"{len(entries):,} entries: {agreed:,} confirm the rule, "
          f"{thin:,} have too little evidence to say,")
    print(f"{len(rows):,} contradict it → {args.output}")
    for entry, winner, back, front in sorted(
            rows, key=lambda r: -max(r[2], r[3]))[:15]:
        print(f"  {entry:<18}{winner:<7}back {back:>5}  front {front:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
