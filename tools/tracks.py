"""Which track a part-of-speech-less entry belongs on, asked of the books.

43.6% of the lexicon carries no part of speech, and an entry with none is
opened on both tracks — it may take a plural and a past tense, a possessive and
a converb. That is the honest thing to do with a word nobody has classified,
and it is expensive: blanking the part of speech on the 56.4% that *do* have
one costs 1.1 points of precision, which makes this the largest single thing
left between the recogniser and its target.

Filling it in is a judgement, so it looked like the one job in this project for
a language model. It is not. Asked for the part of speech of 200 entries
apertium had filed by hand, a 550B model agreed on 60% — and a dozen lines of
corpus probing agrees on 96%, over 58% of the same entries, because the
question is not really "what kind of word is this". It is "does the corpus
write this with `-ды` and `-ған`, or with `-ның` and `-лар`", and the corpus
can be asked directly.

The discovered stems need this most, because nobody classified any of them.
They carry no features at all, so both tracks are open on every one — which is
how `сейдімбек`, a personal name the corpus argued for, came to take a
causative and spell `сейдімбекіт`.

Nothing is claimed where the evidence is thin. An entry the books do not
inflect either way keeps both tracks and its silence.

    python tools/tracks.py
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from analyse import default  # noqa: E402
from phonology import VOICING  # noqa: E402

# `тотық` is written `тотығ` before a vowel, so its verbal forms have to
# be looked for under both spellings.
VOICED_FROM = {v: k for k, v in VOICING.items()}

# Endings that only one track can take. No case suffix is verbal and no tense
# suffix is nominal, so an attested `<entry>+ған` is evidence of a verb in a
# way that needs no model to interpret.
VERBAL = ["ды", "ді", "ты", "ті", "ған", "ген", "қан", "кен",
          "майды", "мейді", "ып", "іп", "у", "са", "се", "ады", "еді",
          "ар", "ер", "атын", "етін", "мақ", "мек", "сын", "сін", "ыл", "іл"]
NOMINAL = ["ның", "нің", "дың", "дің", "тың", "тің", "ға", "ге", "қа", "ке",
           "лар", "лер", "дар", "дер", "тар", "тер", "да", "де", "та", "те",
           "сы", "сі", "ы", "і", "ым", "ім", "мен", "бен", "пен",
           "дан", "ден", "тан", "тен", "ны", "ні", "сыз", "сіз",
           "лық", "лік", "дық", "дік", "тық", "тік"]


def read_attested(path: Path) -> dict[str, int]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {w: int(n) for w, n in
                (line.rstrip("\n").split("\t") for line in fh)}


def evidence(entry: str, attested: dict[str, int]) -> tuple[int, int]:
    return (sum(attested.get(entry + s, 0) for s in VERBAL),
            sum(attested.get(entry + s, 0) for s in NOMINAL))


VERBAL_POS = {"v", "v-trans", "v-intrans", "v-aux"}
NOMINAL_POS = {"n", "np", "adj", "adv", "num", "pron", "det", "post",
               "interj", "conj", "abbr"}


def labelled(lexicon: Path) -> list[tuple[str, set[str]]]:
    """Entries that do have a part of speech, and what it says."""
    out = []
    for line in lexicon.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        form, pos = (line.split("\t") + [""])[:2]
        tags = {t for t in pos.split("+") if t and t != "?"}
        if tags:
            out.append((form, tags))
    return out


def unlabelled(lexicon: Path, discovered: Path | None = None) -> list[str]:
    """Every entry with nothing said about its part of speech.

    The discovered stems are all of them. Nobody vouched for those, so they
    carry no features at all, so both tracks are open on every one — which is
    how `сейдімбек`, a personal name the corpus argued for, came to take a
    causative and spell `сейдімбекіт`.
    """
    out = []
    for line in lexicon.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        form, pos = (line.split("\t") + [""])[:2]
        if not [t for t in pos.split("+") if t and t != "?"]:
            out.append(form)
    if discovered and discovered.exists():
        for line in discovered.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                out.append(line.split("\t")[0])
    return out


def unbuilt(entry: str, suffixes: list[str], attested: dict[str, int],
            analyser) -> int:
    """Book-weight of forms on this track that the recogniser cannot build.

    Attested forms alone are no evidence. `жүре` is a noun and the books write
    `жүреді` 1,928 times, but that is `жүр` plus `-е` plus `-ді` and the
    recogniser already builds it — the weight belongs to `жүр`. Only a form
    nothing can currently spell argues that this entry is missing a track, and
    that is what `тотығып` is: `тотық` is filed a noun, `тотығып` is in 92 of
    the 3,860 editions, and no walk reaches it.
    """
    stems = [entry]
    if entry[-1:] in VOICED_FROM:
        stems.append(entry[:-1] + VOICED_FROM[entry[-1]])
    return sum(attested.get(stem + suffix, 0)
               for stem in stems for suffix in suffixes
               if attested.get(stem + suffix, 0)
               and not analyser.accepts(stem + suffix))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.tsv")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("--discovered", type=Path,
                    default=ROOT / "data/discovered.tsv")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/tracks.tsv")
    ap.add_argument("--min-books", type=int, default=2,
                    help="ignore an entry with less evidence than this")
    ap.add_argument("--add-track", type=int, default=20,
                    help="unbuildable book-weight before a labelled entry is "
                         "given the track its part of speech denies it")
    ap.add_argument("--min-stem", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=1.5,
                    help="how far the winning track must be ahead")
    args = ap.parse_args()

    attested = read_attested(args.attested)
    entries = unlabelled(args.lexicon, args.discovered)

    rows, thin = [], 0
    # An entry that names one track and inflects on the other. A part of speech
    # is one source's opinion, and the books disagree with it often enough to
    # matter: `сен` is filed a verb — сену, to believe — and is also the
    # pronoun, with `сенде`, `сенге` and `сенмен` in 1,041 books between them.
    analyser = default()
    for entry, tags in labelled(args.lexicon):
        if len(entry) < args.min_stem:
            continue
        if tags & NOMINAL_POS and not tags & VERBAL_POS:
            weight = unbuilt(entry, VERBAL, attested, analyser)
            if weight >= args.add_track:
                rows.append((entry, "v", weight, 0))
        elif tags & VERBAL_POS and not tags & NOMINAL_POS:
            weight = unbuilt(entry, NOMINAL, attested, analyser)
            if weight >= args.add_track:
                rows.append((entry, "n", 0, weight))
    added = len(rows)

    for entry in entries:
        verbal, nominal = evidence(entry, attested)
        won, lost = max(verbal, nominal), min(verbal, nominal)
        if won < args.min_books or won < lost * args.ratio:
            thin += 1
            continue
        rows.append((entry, "v" if verbal > nominal else "n", verbal, nominal))

    args.output.write_text(
        "# form\ttrack\tverbal-weight\tnominal-weight — tools/tracks.py\n"
        "# for entries with no part of speech. An entry not listed here keeps\n"
        "# both tracks, because the books do not say which one it is on.\n"
        + "\n".join(f"{f}\t{t}\t{v}\t{n}" for f, t, v, n in rows) + "\n",
        encoding="utf-8")

    verbs = sum(1 for r in rows if r[1] == "v")
    print(f"{added:,} entries were given the track their part of speech "
          f"denied them")
    print(f"{len(entries):,} entries with no part of speech")
    print(f"  {len(rows):,} placed ({len(rows) / max(len(entries), 1):.1%}): "
          f"{verbs:,} verbal, {len(rows) - verbs:,} nominal → {args.output}")
    print(f"  {thin:,} the books do not inflect either way")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
