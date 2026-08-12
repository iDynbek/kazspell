"""Take a word apart into a stem and a legal walk through the template.

This is the reference implementation of the recogniser: the thing the Rust core
has to agree with, written to be read rather than to be fast. A word is
accepted when some entry in the lexicon, followed by some ordered sequence of
slots, each in the shape that stem licenses, spells it exactly.

Everything the architecture is for shows up here as an ordinary condition:

    the slot order          a suffix may only follow one of lower order
    the group               one of the finiteness family, never two
    `requires`              the passive is refused on an intransitive entry
    `restart`               after a participle the walk continues nominal
    the phonology           `мектепдер` is refused because `мектеп` takes `т`

    python tools/analyse.py мектептерімізде барғаным жазылмады
    python tools/analyse.py --check мектепдер     # expect no analysis
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phonology import fits  # noqa: E402
from template import Template, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_SLOTS = 6


def read_lexicon(path: Path) -> dict[str, frozenset[str]]:
    """form -> the features it carries, for `requires` to test against."""
    out: dict[str, frozenset[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        form, pos = parts[0], parts[1] if len(parts) > 1 else "?"
        features = set()
        for tag in pos.split("+"):
            if tag and tag != "?":
                features.add(tag)
        # An entry with no part of speech could be either track, and saying so
        # is honest: 43.6% of the lexicon is in that state until the corpus
        # assignment lands. A transitivity-gated slot still refuses it, because
        # `v-trans` is a claim nobody has made about this word.
        out[form] = frozenset(features)
    return out


def tracks_of(features: frozenset[str]) -> tuple[str, ...]:
    """Which template tracks an entry can start on."""
    verbal = {"v", "v-trans", "v-intrans", "v-aux"}
    nominal = {"n", "np", "adj", "adv", "num", "pron", "det", "post", "interj"}
    out = []
    if features & nominal or not features:
        out.append("n")
    if features & verbal or not features:
        out.append("v")
    return tuple(out)


class Analyser:
    def __init__(self, tpl: Template, lexicon: dict[str, frozenset[str]],
                 overrides: dict[str, str] | None = None):
        self.tpl = tpl
        self.lexicon = lexicon
        self.overrides = overrides or {}
        self.max_morpheme = max(len(m) for m in tpl.by_morpheme)

    def analyse(self, word: str) -> list[list[str]]:
        """Every (stem, slot...) reading of `word`. Empty means not a word."""
        word = word.lower()
        out: list[list[str]] = []
        seen = set()
        for cut in range(1, len(word) + 1):
            stem, rest = word[:cut], word[cut:]
            features = self.lexicon.get(stem)
            if features is None:
                continue
            for track in tracks_of(features):
                for path in self._walk(stem, rest, track, None, features):
                    reading = tuple([stem] + path)
                    if reading not in seen:
                        seen.add(reading)
                        out.append(list(reading))
        return out

    def _walk(self, surface: str, rest: str, track: str, prev, features):
        """Sequences of slots spelling `rest`, attaching after `surface`."""
        if not rest:
            yield []
            return
        if prev is not None and len(rest) > self.max_morpheme * MAX_SLOTS:
            return
        for cut in range(min(len(rest), self.max_morpheme), 0, -1):
            morpheme = rest[:cut]
            for slot in self.tpl.by_morpheme.get(morpheme, ()):
                if prev is None:
                    if slot.track != track or slot.after:
                        continue
                elif not self.tpl.may_follow(prev, slot):
                    continue
                if slot.requires and not set(slot.requires) <= features:
                    continue
                if not fits(surface, morpheme, slot.morphemes, self.overrides):
                    continue
                nxt = slot.restart or slot.track
                for tail in self._walk(surface + morpheme, rest[cut:],
                                       nxt, slot, features):
                    yield [f"{slot.id}:{morpheme}"] + tail

    def accepts(self, word: str) -> bool:
        return bool(self.analyse(word))


@functools.lru_cache(maxsize=1)
def default() -> Analyser:
    return Analyser(load(), read_lexicon(ROOT / "data/lexicon.tsv"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any word is accepted")
    ap.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.tsv")
    args = ap.parse_args()

    an = Analyser(load(), read_lexicon(args.lexicon))
    bad = 0
    for word in args.words:
        readings = an.analyse(word)
        if not readings:
            print(f"{word}\t—")
            continue
        bad += 1
        for reading in readings[:6]:
            print(f"{word}\t{' + '.join(reading)}")
        if len(readings) > 6:
            print(f"{word}\t… {len(readings) - 6} more")
    return 1 if (args.check and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
