"""Lift apertium-kaz's per-lexeme paradigm assignment out of its lexc source.

This is the information hunspell-kk does not have and cannot mine. Its affix
file asks what a stem sounds like — harmony, and a five-way class for the
consonant the next suffix should start with. What decides whether a suffix may
attach is what the stem *is*, and apertium-kaz has been carrying that answer,
by hand, for 34,880 stems: every entry names a continuation lexicon, and the
lexicon is a paradigm.

The distinction that matters most is one we have never encoded. `V-TV` and
`V-IV` are transitive and intransitive, 1,769 and 1,823 stems, and only the
transitive ones form a passive — `жаз` gives `жазылды`, `отыр` does not give
`*отырылды`. hunspell-kk hands every voice to every verb.

    python tools/apertium_paradigms.py --lexc ../apertium-kaz/apertium-kaz.kaz.lexc

Writes data/apertium_paradigms.tsv: lemma, paradigms, part of speech.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# An entry line is `upper:lower CONTINUATION ;` with an optional `! comment`.
# `%` escapes the next character, including `:` and `;`, so it has to be
# honoured or entries like `%:%<punct%>` split in the wrong place.
ENTRY = re.compile(r"^(?P<body>(?:%.|[^!;])*);")

# Which part of speech each continuation lexicon represents. Anything not named
# here is carried through verbatim: the point is to lose nothing, and an
# unmapped paradigm is visible rather than silently nominal.
POS_OF = {
    "N1": "n", "N5": "n", "N1-NAT": "n", "NP-COMMON": "n",
    "V-IV": "v-intrans", "V-TV": "v-trans", "V-TD": "v-trans",
    "Vinfl-AUX": "v-aux",
    "A1": "adj", "A2": "adj", "A3": "adj", "A4": "adj",
    "ADV": "adv", "ADV-LANG": "adv", "ADV-WITH-KI": "adv",
    "NUM": "num", "NUM-DIGIT": "num",
    "DET-QNT": "det",
    "ABBR": "abbr",
    "INTERJ": "interj", "IDEO": "interj",
    "POST": "post", "POST-NOM": "post", "POST-DAT": "post",
    "CC-SOYED": "conj",
}
PROPER = re.compile(r"^NP-")


def parse(path: Path) -> dict[str, set[str]]:
    """lemma -> the continuation lexicons it is filed under."""
    out: dict[str, set[str]] = collections.defaultdict(set)
    lexicon = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("LEXICON"):
            lexicon = line.split()[1]
            continue
        if not line or line.startswith("!"):
            continue
        m = ENTRY.match(line)
        if not m:
            continue
        body = m.group("body").strip()
        if not body:
            continue
        parts = body.split()
        if len(parts) < 2:
            continue
        form, cont = parts[0], parts[-1]
        if not re.match(r"^[A-Z][A-Za-z0-9_-]*$", cont):
            continue
        # `upper:lower` — the analysis side is the lemma we want. Unescape.
        lemma = re.split(r"(?<!%):", form)[0]
        lemma = re.sub(r"%(.)", r"\1", lemma).strip()
        # Continuation entries carry tags rather than lemmas — `+е<cop><aor>`,
        # `+лы<post>`. They are morphology, not vocabulary, and letting them
        # through put `+ма<qst>+е<cop><aor><evid>` into the wordlist.
        if not lemma or lemma.startswith(("<", "+")) or "<" in lemma:
            continue
        out[lemma].add(cont)
    return out


def pos_of(paradigms: set[str]) -> str:
    tags = set()
    for p in paradigms:
        if p in POS_OF:
            tags.add(POS_OF[p])
        elif PROPER.match(p):
            tags.add("np")
    if not tags:
        return "?"
    # A lemma really can be both; `бала` is a noun and a verb. Keep both, in a
    # stable order, rather than picking a winner here — the family assignment
    # is what has to decide, and it should see the ambiguity.
    return "+".join(sorted(tags))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexc", type=Path,
                    default=ROOT.parent / "apertium-kaz/apertium-kaz.kaz.lexc")
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "data/apertium_paradigms.tsv")
    args = ap.parse_args()

    if not args.lexc.exists():
        sys.exit(f"no lexc at {args.lexc}")
    entries = parse(args.lexc)
    print(f"{len(entries):,} lemmas, "
          f"{len({p for v in entries.values() for p in v}):,} paradigms",
          file=sys.stderr)

    rows = []
    for lemma in sorted(entries):
        paradigms = entries[lemma]
        rows.append((lemma, ",".join(sorted(paradigms)), pos_of(paradigms)))

    args.output.write_text(
        "# lemma\tparadigms\tpos — from apertium-kaz's lexc, "
        "tools/apertium_paradigms.py\n"
        + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")

    dist = collections.Counter(r[2] for r in rows)
    print(f"\n{'part of speech':<22}{'lemmas':>8}")
    for tag, n in dist.most_common():
        print(f"  {tag:<20}{n:>8,}")
    print(f"\n→ {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
