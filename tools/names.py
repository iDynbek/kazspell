"""Proper names the corpus capitalises, reduced to the lemma that inflects.

Names are two fifths of what a Kazakh checker flags in running text and the
largest thing this one still refuses. They were left out because a name list
cannot simply be admitted — 995 of the 23,981 names available sit one harmony
edit from an ordinary word, and they are the commonest words in the language:
`болган` against `болған`, `ким` against `кім`, `жана` against `жаңа`. Letting
those in makes the commonest misspelling of the commonest words unflaggable.

What was missing was the case, and `tools/build_case.py` recovers it. A form
capitalised mid-sentence and almost never otherwise is a name; `тұрсын` is
capitalised 23 times and written in lower case 232, so it is the verb and not
the name, which is exactly the distinction the folded table could not make.

Two more things are needed to get from that to a lexicon.

A capitalised *form* is not a lemma. `абайдың` and `қазақстанда` are as
capitalised as `абай` and `қазақстан`, and admitting them as entries would let
them inflect again. So a candidate is dropped when a shorter candidate plus a
legal walk through the template already spells it.

And the harmony gate stays. A candidate one Russian-keyboard edit from an
attested ordinary word is refused however capitalised it is, because the cost
of admitting it is paid by every writer who misspells that ordinary word.

    python tools/names.py
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from analyse import (Analyser, load_lexicon, read_elision,  # noqa: E402
                     read_harmony)
from template import load as load_template  # noqa: E402

KAZAKH_ONLY = set("әғқңөұүһі")
TRANSLIT = {"у": "ұү", "и": "і", "к": "қ", "г": "ғ", "н": "ң", "а": "ә", "о": "ө"}


def kazakhised(word: str):
    spots = [i for i, c in enumerate(word) if c in TRANSLIT]
    if not spots or len(spots) > 10:
        return
    for r in (1, 2):
        for combo in itertools.combinations(spots, r):
            for choice in itertools.product(*[TRANSLIT[word[i]] for i in combo]):
                out = list(word)
                for i, c in zip(combo, choice):
                    out[i] = c
                yield "".join(out)


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def read_case(path: Path) -> dict[str, tuple[int, int, int]]:
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            form, up, low, works = line.rstrip("\n").split("\t")
            out[form] = (int(up), int(low), int(works))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=ROOT / "data/case.tsv.gz")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/names.tsv")
    ap.add_argument("--min-caps", type=int, default=3,
                    help="times a form must be capitalised mid-sentence")
    ap.add_argument("--ratio", type=float, default=9.0,
                    help="how far capitalised must beat lower case")
    ap.add_argument("--min-length", type=int, default=4)
    ap.add_argument("--lists", type=Path, nargs="*", default=[
        ROOT.parent / "hunspell-kk/data/names.txt",
        ROOT.parent / "hunspell-kk/data/names_wp.txt"],
        help="name lists to admit on their own authority, gated the same way")
    args = ap.parse_args()

    case = read_case(args.case)
    attested = read_attested(ROOT / "data/attested.tsv.gz")
    # Deliberately without the names file this writes. Using the shipped
    # analyser would read the last run's output back in, find every name
    # already buildable, and emit almost nothing — which is what it did.
    an = Analyser(load_template(),
                  load_lexicon(ROOT / "data/lexicon.tsv",
                               ROOT / "data/discovered.tsv",
                               ROOT / "data/tracks.tsv"),
                  overrides=read_harmony(ROOT / "data/harmony.tsv"),
                  elision=read_elision(ROOT / "data/elision.tsv"))

    candidates = {f for f, (up, low, _w) in case.items()
                  if up >= args.min_caps and up >= low * args.ratio
                  and len(f) >= args.min_length}
    print(f"{len(candidates):,} forms are capitalised mid-sentence and "
          f"hardly ever otherwise", file=sys.stderr)

    # A candidate a shorter candidate already explains is that one inflected.
    lemmas, inflected = [], 0
    # Sorted by length and then alphabetically: ties broken by set order made
    # the output differ between runs of the same input.
    for form in sorted(candidates, key=lambda f: (len(f), f)):
        shorter = None
        for cut in range(args.min_length, len(form)):
            head, rest = form[:cut], form[cut:]
            if head not in candidates:
                continue
            if any(an._walk(head, rest, track, None, frozenset())
                   for track in ("n", "v")):
                shorter = head
                break
        if shorter:
            inflected += 1
        else:
            lemmas.append(form)

    # The lists are names somebody annotated rather than names this corpus
    # capitalises — KazNERD's training data and Wikipedia's titles — so they
    # carry no case evidence and enter on their source's authority. Everything
    # after this point applies to them exactly as it does to the corpus's own.
    listed = 0
    for path in args.lists:
        if not path.exists():
            print(f"  no list at {path}", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            form = line.strip().lower()
            if (form and not form.startswith("#") and form.isalpha()
                    and len(form) >= args.min_length and form not in candidates):
                lemmas.append(form)
                listed += 1
    print(f"  {listed:,} more from the lists", file=sys.stderr)

    kept, refused = [], []
    for form in lemmas:
        # Membership, not buildability. A name the morphology happens to be
        # able to spell is still needed as an entry, because its own
        # inflections are built on it: dropping `төке` for being spellable
        # took `төкеңнен` and `төкені` with it.
        if form in an.lexicon:
            continue
        if not (set(form) & KAZAKH_ONLY):
            clash = next((v for v in kazakhised(form)
                          if (v in case and case[v][1] > case[v][0])
                          or attested.get(v, 0) >= 2), None)
            if clash:
                refused.append((form, clash))
                continue
        kept.append(form)

    args.output.write_text(
        "# form\tcapitalised\tlower-case\tworks — tools/names.py\n"
        "# proper names the corpus capitalises, reduced to the lemma. They go\n"
        "# in as `np`, so the nominal track and nothing else.\n"
        + "\n".join(f"{f}\t{case.get(f, (0, 0, 0))[0]}\t"
                    f"{case.get(f, (0, 0, 0))[1]}\t{case.get(f, (0, 0, 0))[2]}"
                    for f in sorted(kept,
                                    key=lambda x: (-case.get(x, (0, 0, 0))[0], x)))
        + "\n",
        encoding="utf-8")

    print(f"  {inflected:,} were a shorter name already inflected")
    print(f"  {len(refused):,} are one keyboard edit from an ordinary word:")
    for form, clash in refused[:8]:
        print(f"      {form} against {clash}")
    print(f"  {len(kept):,} names → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
