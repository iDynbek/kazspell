"""Admit Kazakh personal and place names, correcting their spelling on the way in.

Names are two fifths of what a Kazakh spellchecker flags in running text, so
leaving them out is not an option. Taking them as offered is not either: 868 of
the 951 unflaggable transliteration duplicates hunspell-kk ships came from one
Wikipedia name dump, where names are listed as they appear in Russian-language
documents — `гулжазира` for `гүлжазира`, `сандыгуль` for `сандыгүл`.

The two spellings are not equals. Both get written, but only one is Kazakh, and
a checker that accepts both can never tell a writer they typed the wrong one.
So the list is not filtered, it is *corrected*: for a name spelt without a
single Kazakh letter, the restored spellings are looked up in 3,860 books, and
where one of them is what people actually write, that is the name that goes in
and the original does not.

The decision is only made where the books can make it. A rare name attested
nowhere keeps the spelling it came with — guessing at harmony for a name we
have never seen written would invent vocabulary, which is the failure mode this
whole exercise exists to avoid.

    python tools/names.py -o data/names.tsv
"""

from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent

KAZAKH_ONLY = set("әғқңөұүһі")
ALPHABET = set("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")
LATIN = re.compile(r"[A-Za-z]")

# The substitutions that turn a Russian-document spelling back into Kazakh.
TRANSLIT = {"у": "ұү", "и": "і", "к": "қ", "г": "ғ", "н": "ң", "а": "ә", "о": "ө"}
MAX_SUBS = 3

# How much better attested a restored spelling has to be before it replaces the
# one we were given. A name is not a common noun: `абдиров` and `әбдіров` are
# both written, so a small margin would flip pairs that are genuinely in use.
MARGIN = 4
MIN_DOCS = 3

# A correction must land on something with independent reason to be a name.
# `common` holds lemmas, so it does not catch inflected forms, and the corpus
# counts cannot tell `Тұрсын` the name from `тұрсың` "you stand" — 1,141 books
# of the second are no evidence at all about the first. Either KazNERD tagged
# the target as a name in running text, or it is a name we hold that is too
# rare to be a common word form.
NAME_DOC_CAP = 200


def variants(word: str):
    positions = [i for i, c in enumerate(word) if c in TRANSLIT]
    if not positions or len(positions) > 12:
        return
    for r in range(1, MAX_SUBS + 1):
        for combo in itertools.combinations(positions, r):
            for choice in itertools.product(*[TRANSLIT[word[i]] for i in combo]):
                out = list(word)
                for i, c in zip(combo, choice):
                    out[i] = c
                yield "".join(out)


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.split("\t")[0].strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def read_kaznerd(path: Path) -> collections.Counter:
    """Names as annotated in real text, with how often each was seen.

    This list is evidence in a way a name dump is not: every token in it was
    written by someone in a sentence, and tagged by someone else as a name.
    """
    freq = collections.Counter()
    for split in ("train", "valid", "test"):
        f = path / f"IOB2_{split}.txt"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].endswith(
                    ("PERSON", "LOCATION", "ORGANISATION")):
                freq[parts[0].lower()] += 1
    return freq


def resolve(name: str, attested: dict[str, int], common: set[str],
            known_names: set[str], nerd: set[str]) -> tuple[str, str | None]:
    """(spelling to admit, the spelling it replaced) — corrects where it can.

    The book counts are case-folded, so they cannot tell a name from a noun
    that happens to be spelt the same way. Left to itself the search finds the
    commonest word within one harmony edit and "corrects" towards it: `али` to
    `әлі` (still), `ким` to `кім` (who), `жана` to `жаңа` (new), each on
    thousands of books that are not about anybody's name.

    So a correction may not land on a word the lexicon already holds as common
    vocabulary — not even when that spelling is also a name. `Әлі` is a name and
    `әлі` is an adverb in 2,907 books, and no count taken over case-folded text
    can say which one the corpus was counting. Where the target is ambiguous the
    original spelling stays; the correction is only made where it is safe.

    What survives is the case the exercise is for — `гулжазира` to `гүлжазира`,
    where the target is a name and nothing else.
    """
    if set(name) & KAZAKH_ONLY:
        return name, None                       # already Kazakh-spelt
    here = attested.get(name, 0)
    best, best_n = None, 0
    for v in variants(name):
        if v in common:
            continue                            # that is a word, not this name
        n = attested.get(v, 0)
        name_evidence = v in nerd or (v in known_names and n <= NAME_DOC_CAP)
        if not name_evidence:
            continue
        if n > best_n:
            best, best_n = v, n
    if best and best_n >= MIN_DOCS and best_n >= here * MARGIN:
        return best, name
    return name, None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/names.tsv")
    ap.add_argument("--corrections", type=Path,
                    default=ROOT / "data/name_corrections.tsv")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("--wikipedia", type=Path,
                    default=ROOT.parent / "hunspell-kk/data/names_wp.txt")
    ap.add_argument("--mined", type=Path,
                    default=ROOT.parent / "hunspell-kk/data/names.txt")
    ap.add_argument("--kaznerd", type=Path, default=ROOT.parent / "KazNERD/KazNERD")
    ap.add_argument("--apertium", type=Path,
                    default=ROOT / "data/apertium_paradigms.tsv")
    ap.add_argument("--kazdict", type=Path,
                    default=ROOT.parent / "kazdict/data/build/kazdict.db")
    args = ap.parse_args()

    attested = read_attested(args.attested)
    nerd = read_kaznerd(args.kaznerd)

    # Common vocabulary, so a correction cannot walk a name into a noun.
    from build_lexicon import read_apertium, read_kazdict
    common = {w for w, (_par, pos) in read_apertium(args.apertium).items()
              if pos and pos != "np"}
    common |= set(read_kazdict(args.kazdict))
    print(f"{len(common):,} common-vocabulary spellings guard the corrections",
          file=sys.stderr)
    candidates: dict[str, set[str]] = collections.defaultdict(set)
    for word in read_list(args.wikipedia):
        candidates[word].add("wikipedia")
    for word in read_list(args.mined):
        candidates[word].add("mined")
    for word in nerd:
        candidates[word].add("kaznerd")
    print(f"{len(candidates):,} name candidates "
          f"(wikipedia, mined, kaznerd)", file=sys.stderr)

    all_names = set(candidates)
    admitted: dict[str, set[str]] = collections.defaultdict(set)
    corrections, dropped = [], []
    for name in sorted(candidates):
        if len(name) < 2 or LATIN.search(name) or (set(name) - ALPHABET - {"-", "'"}):
            dropped.append((name, ",".join(sorted(candidates[name])), "orthography"))
            continue
        keep, replaced = resolve(name, attested, common, all_names, set(nerd))
        admitted[keep] |= candidates[name]
        if replaced:
            corrections.append((replaced, keep, str(attested.get(replaced, 0)),
                                str(attested.get(keep, 0))))

    args.output.write_text(
        "# name\tsources\tdocs — tools/names.py\n"
        + "\n".join(f"{n}\t{','.join(sorted(s))}\t{attested.get(n, 0)}"
                    for n, s in sorted(admitted.items())) + "\n", encoding="utf-8")
    args.corrections.write_text(
        "# given\tadmitted\tdocs_given\tdocs_admitted — Russian-document spellings\n"
        "# replaced by the Kazakh spelling the books show people writing\n"
        + "\n".join("\t".join(c) for c in corrections) + "\n", encoding="utf-8")

    print(f"\n{len(admitted):,} names admitted")
    print(f"  spellings corrected on the way in: {len(corrections):,}")
    print(f"  refused on orthography:            {len(dropped):,}")
    seen = sum(1 for n in admitted if n in attested)
    print(f"  attested in the books:             {seen:,} ({seen/len(admitted):.1%})")
    print("\nsample corrections (given -> admitted, books each):")
    for a, b, na, nb in sorted(corrections, key=lambda c: -int(c[3]))[:12]:
        print(f"  {a:<18} -> {b:<18} {na:>5} -> {nb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
