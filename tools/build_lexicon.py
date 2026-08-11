"""Build the lexicon from primary sources, with every entry having to earn its place.

hunspell-kk assembled its wordlist by union: anything any source offered went
in, and the filtering happened afterwards or not at all. What that let through
is known and measured — 16 entries with a Latin `i` inside Cyrillic, inherited
from the 2009 release and shipped for three years, and 951 Russian-transliterated
spellings attested in none of 3,860 books whose Kazakh spelling is attested in
up to 1,827, so `сезим` stands beside `сезім` and the checker can never flag it.

Neither is expensive to fix. Both are expensive to fix *later*, once the affix
model, the paradigm assignment and the shipped artefacts have all been built on
top. So the gates are here, at the point of admission, and every rejection is
written out with its reason rather than silently dropped.

Sources, in order of what they can vouch for:

    apertium    33,569 lemmas, each hand-filed under a paradigm. The only
                source that knows transitivity.
    kazdict     86,569 single-word headwords from the sozdikqor corpus, 45,737
                carrying a part-of-speech label. Much the largest, and the only
                one that reaches modern loanwords.
    kaznerd     proper names from entity annotations.
    books       84.2M tokens over 3,860 editions. Not a vocabulary source —
                a witness. It decides the gates, it does not open them.

    python tools/build_lexicon.py -o data/lexicon.tsv
"""

from __future__ import annotations

import argparse
import collections
import gzip
import itertools
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The Kazakh Cyrillic alphabet, all 42 letters. Anything outside it is not a
# spelling of a Kazakh word, whatever source offered it.
ALPHABET = set("аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя")
EXTRA = set("-'’")
KAZAKH_ONLY = set("әғқңөұүһі")
LATIN = re.compile(r"[A-Za-z]")

# Russian-passport transliteration against the Kazakh spelling. Applied in
# combination, because a name usually carries more than one.
TRANSLIT = {"у": "ұү", "и": "і", "к": "қ", "г": "ғ", "н": "ң", "а": "ә", "о": "ө"}
MAX_SUBS = 3


def kazakhised(word: str):
    """Every spelling of `word` that restores Kazakh letters, up to MAX_SUBS."""
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


def read_attested(path: Path) -> set[str]:
    if not path.exists():
        sys.exit(f"no attestation set at {path}")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {line.rstrip("\n") for line in fh}


def read_apertium(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            lemma, paradigms, pos = line.split("\t")
            out[lemma.lower()] = (paradigms, pos)
    return out


KDPOS = {"зат": "n", "сын": "adj", "ет": "v", "үст": "adv", "есім": "n",
         "сан": "num", "шыл": "post", "одағ": "interj", "ес": "pron",
         "ел": "interj", "мод": "adv"}


def read_kazdict(path: Path) -> dict[str, set[str]]:
    """headword -> parts of speech. Case-folded in Python: SQLite's lower() is
    ASCII-only and silently matches 33 Cyrillic headwords instead of 45,737."""
    if not path.exists():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out: dict[str, set[str]] = collections.defaultdict(set)
    rows = con.execute("""
        SELECT h.form, (SELECT group_concat(DISTINCT l.code)
                          FROM entry e JOIN sense s ON s.entry_id = e.id
                          JOIN sense_label sl ON sl.sense_id = s.id
                          JOIN label l ON l.id = sl.label_id
                         WHERE e.headword_id = h.id)
          FROM headword h""")
    for form, codes in rows:
        word = form.lower().strip()
        if word:
            out[word] |= {KDPOS[c] for c in (codes or "").split(",") if c in KDPOS}
    return out


def read_baseline(path: Path) -> set[str]:
    """The 2009 release's wordlist: 54,063 entries, assembled by hand.

    It is the dictionary being replaced and it is still the only hand-checked
    Kazakh wordlist there is. Its known defects — the Latin `i` inside Cyrillic
    names — are exactly what the gates catch, so it can be taken as a source
    without being taken on trust.
    """
    if not path.exists():
        return set()
    return {line.partition("/")[0].strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()[1:]
            if line.strip()}


def read_kaznerd(path: Path) -> set[str]:
    names = set()
    for split in ("train", "valid", "test"):
        f = path / f"IOB2_{split}.txt"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[-1].endswith(("PERSON", "LOCATION",
                                                       "ORGANISATION")):
                names.add(parts[0].lower())
    return names


def gate(word: str, sources: set[str], attested: set[str],
         candidates: set[str]) -> str | None:
    """The reason to refuse this entry, or None to admit it."""
    if not word:
        return "empty"
    # Multi-word headwords are a real part of the language and no business of a
    # single-word speller. They are set aside, not thrown away: 60,903 of them
    # are the raw material for whatever handles compounds and phrases later.
    if " " in word:
        return "multiword"
    if len(word) > 32:
        return "length"
    if LATIN.search(word):
        return "mixed-script" if set(word) & ALPHABET else "not-cyrillic"
    bad = set(word) - ALPHABET - EXTRA
    if bad:
        return f"outside-alphabet:{''.join(sorted(bad))}"
    if word.strip(''.join(EXTRA)) != word:
        return "edge-punctuation"
    # A spelling with no Kazakh letter, whose Kazakh spelling is a candidate
    # and is attested while this one is not, is the Russian-keyboard form of a
    # word we already have. Admitting it means never being able to flag it.
    if not (set(word) & KAZAKH_ONLY) and word not in attested:
        for variant in kazakhised(word):
            if variant in candidates and variant in attested:
                return f"translit-of:{variant}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/lexicon.tsv")
    ap.add_argument("--rejects", type=Path, default=ROOT / "data/rejected.tsv")
    ap.add_argument("--apertium", type=Path,
                    default=ROOT / "data/apertium_paradigms.tsv")
    ap.add_argument("--kazdict", type=Path,
                    default=ROOT.parent / "kazdict/data/build/kazdict.db")
    ap.add_argument("--kaznerd", type=Path, default=ROOT.parent / "KazNERD/KazNERD")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.txt.gz")
    ap.add_argument("--baseline", type=Path,
                    default=ROOT.parent / "hunspell-kk/baseline/kk_KZ.dic")
    args = ap.parse_args()

    attested = read_attested(args.attested)
    apertium = read_apertium(args.apertium)
    kazdict = read_kazdict(args.kazdict)
    kaznerd = read_kaznerd(args.kaznerd)
    baseline = read_baseline(args.baseline)
    print(f"apertium {len(apertium):,}  kazdict {len(kazdict):,}  "
          f"kaznerd {len(kaznerd):,}  baseline {len(baseline):,}  "
          f"attested {len(attested):,}", file=sys.stderr)

    sources: dict[str, set[str]] = collections.defaultdict(set)
    for word in apertium:
        sources[word].add("apertium")
    for word in kazdict:
        sources[word].add("kazdict")
    for word in kaznerd:
        sources[word].add("kaznerd")
    for word in baseline:
        sources[word].add("baseline")

    candidates = set(sources)
    admitted, rejected = [], []
    for word in sorted(candidates):
        why = gate(word, sources[word], attested, candidates)
        if why:
            rejected.append((word, ",".join(sorted(sources[word])), why))
            continue
        par, pos = apertium.get(word, ("", ""))
        kd = kazdict.get(word, set())
        if not pos and kd:
            pos = "+".join(sorted(kd))
        if not pos and "kaznerd" in sources[word]:
            pos = "np"
        admitted.append((word, pos or "?", par,
                         ",".join(sorted(sources[word])),
                         "1" if word in attested else "0"))

    args.output.write_text(
        "# form\tpos\tparadigms\tsources\tattested — tools/build_lexicon.py\n"
        + "\n".join("\t".join(r) for r in admitted) + "\n", encoding="utf-8")
    multiword = [r for r in rejected if r[2] == "multiword"]
    rejected = [r for r in rejected if r[2] != "multiword"]
    args.rejects.write_text(
        "# form\tsources\treason — refused by tools/build_lexicon.py\n"
        + "\n".join("\t".join(r) for r in rejected) + "\n", encoding="utf-8")
    (args.rejects.parent / "multiword.tsv").write_text(
        "# form\tsources — set aside by tools/build_lexicon.py, out of scope for a\n"
        "# single-word speller and kept for whatever handles phrases later\n"
        + "\n".join(f"{w}\t{s}" for w, s, _r in multiword) + "\n", encoding="utf-8")
    print(f"{len(multiword):,} multi-word headwords set aside → multiword.tsv")

    print(f"\n{len(candidates):,} candidates → {len(admitted):,} admitted, "
          f"{len(rejected):,} refused")
    reasons = collections.Counter(r[2].split(":")[0] for r in rejected)
    for why, n in reasons.most_common():
        print(f"  {why:<22}{n:>7,}")
    known = sum(1 for r in admitted if r[1] != "?")
    seen = sum(1 for r in admitted if r[4] == "1")
    print(f"\n  with a part of speech  {known:>7,}  {known/len(admitted):>6.1%}")
    print(f"  attested in the books  {seen:>7,}  {seen/len(admitted):>6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
