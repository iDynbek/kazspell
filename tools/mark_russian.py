"""Mark the attested types that are Russian, so recall can be read honestly.

The 3,860 editions are Kazakh books and they contain Russian — quoted, cited,
code-switched, and in whole passages. Every Russian word in them is a type the
recogniser is asked to accept and should not, so it counts as a miss twice
over: once for being wrong to accept, once for being counted against recall
when refused. At the last measurement 64.5% of the missed book-weight was
valid Russian, which means the headline recall number was mostly reporting how
much Russian is in the corpus.

The remedy is to say which types those are, once, and let the harness set them
aside. A type is set aside when a Hunspell Russian dictionary accepts it —
including the ones Kazakh also has, `да`, `бар`, `бала`, some 8,600 of them.
Dropping those costs a little coverage that is really ours, and it is the only
way to keep the exclusion from being circular: excluding only the *missed*
Russian words would define the failures away and could never lower the score.

The dictionary is a parameter, not a file in this repo. Any Hunspell ru_RU will
do — the LibreOffice one, or a distribution's `hunspell-ru` package — and the
output here is a fact about this corpus rather than a copy of anyone's
dictionary.

    python tools/mark_russian.py --dict /usr/share/hunspell/ru_RU
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The nine letters Cyrillic added for Kazakh. No Russian dictionary should
# accept a word carrying one, and at the last run none did — this is here to
# fail loudly if the dictionary passed in is not the one it claims to be.
KAZAKH_ONLY = set("әғқңөұүһі")


def read_attested(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [line.split("\t")[0] for line in fh]


def accepted_by(words: list[str], dictionary: Path) -> set[str]:
    """The words a Hunspell dictionary considers correct.

    `hunspell -l` prints the ones it rejects, so the answer is everything else.
    """
    if not shutil.which("hunspell"):
        sys.exit("hunspell is not installed")
    if not Path(f"{dictionary}.dic").exists():
        sys.exit(f"no dictionary at {dictionary}.dic")
    out = subprocess.run(
        ["hunspell", "-d", str(dictionary), "-l", "-i", "utf-8"],
        input="\n".join(words), capture_output=True, text=True, check=True)
    return set(words) - set(out.stdout.split())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", type=Path, required=True,
                    help="a Hunspell ru_RU, without the .dic/.aff suffix")
    ap.add_argument("--attested", type=Path, default=ROOT / "data/attested.tsv.gz")
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "data/russian.tsv.gz")
    args = ap.parse_args()

    words = read_attested(args.attested)
    russian = accepted_by(words, args.dict)

    wrong = sorted(w for w in russian if set(w) & KAZAKH_ONLY)
    if wrong:
        for word in wrong[:10]:
            print(f"  {word}", file=sys.stderr)
        sys.exit(f"{len(wrong):,} words with a Kazakh-only letter were accepted "
                 f"as Russian; {args.dict} is not a Russian dictionary")

    with gzip.open(args.output, "wt", encoding="utf-8") as fh:
        fh.write(f"# attested types a Russian dictionary accepts — "
                 f"tools/mark_russian.py --dict {args.dict}\n")
        fh.write("\n".join(sorted(russian)) + "\n")

    print(f"{len(words):,} attested types, {len(russian):,} of them Russian "
          f"({len(russian) / len(words):.1%}) → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
