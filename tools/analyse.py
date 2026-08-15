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

from phonology import VOWELS, devoice, fits  # noqa: E402
from template import Template, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MAX_SLOTS = 6

# How many stem-deriving suffixes one walk may use. Kazakh stacks derivation —
# `дос-тық-тың` is ordinary — but not without limit, and a walk that reaches for
# a third is usually spelling a misspelling: among the accepted misspellings in
# the harness, 33 need two derivations or more against 11 real words, and 3 need
# three against none.
MAX_DERIVATIONS = 2
DERIVING = {"sozjasam", "etis_jasam"}


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


def read_discovered(path: Path) -> dict[str, frozenset[str]]:
    """Stems the books inflect that no source vouched for.

    Kept in their own file rather than merged into the lexicon, because they
    are not the same kind of thing: every entry in `lexicon.tsv` was offered by
    apertium, kazdict or the 2009 release and then gated, and these were argued
    for by the corpus alone. They carry no part of speech, so both tracks are
    open to them and `requires` refuses them, which is the right amount of
    confidence to extend to a word nobody has vouched for.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            out[line.split("\t")[0]] = frozenset()
    return out


def read_tracks(path: Path) -> dict[str, str]:
    """form -> the track the books put it on, for entries with no part of speech."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            form, track = line.split("\t")[:2]
            out[form] = track
    return out


def load_lexicon(lexicon: Path, discovered: Path,
                 tracks: Path | None = None) -> dict[str, frozenset[str]]:
    """The vouched-for entries, plus the discovered ones that add something.

    An entry with no part of speech is opened on both tracks, which is honest
    and expensive. Where the corpus inflects it one way and not the other, that
    settles it — see tools/tracks.py — and the other track closes.
    """
    out = read_lexicon(lexicon)
    for form, features in read_discovered(discovered).items():
        out.setdefault(form, features)
    for form, track in read_tracks(tracks or Path("/nonexistent")).items():
        if form in out and not out[form]:
            out[form] = frozenset({track})
    return out


def read_elision(path: Path) -> dict[str, str]:
    """elided stem -> the lemma it belongs to.

    A closed class drops the vowel of its last syllable before a vowel-initial
    suffix: `мойын` is `мойны`, `халық` is `халқы`. The elided form is not a
    word on its own and never stands without a suffix, and the suffix must be
    the vowel-initial one that caused the elision — `мойында` is the plain stem
    plus a locative, not `мойн` plus anything.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            lemma, elided = line.split("\t")
            out[elided] = lemma
    return out


def read_harmony(path: Path) -> dict[str, str]:
    """entry -> the column it takes, where the letters predict the other one.

    `мораль` is back and `медаль` is front, `тарих` is back and `хал` is front,
    and no rule over the letters recovers any of it. tools/harmony_overrides.py
    reads the answers off the books; this is where they are used.
    """
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            form, column = line.split("\t")[:2]
            out[form] = column
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
                 overrides: dict[str, str] | None = None,
                 elision: dict[str, str] | None = None):
        self.tpl = tpl
        self.lexicon = lexicon
        self.overrides = overrides or {}
        self.elision = elision or {}
        self.max_derivations = MAX_DERIVATIONS
        self.max_morpheme = max(len(m) for m in tpl.by_morpheme)

    def analyse(self, word: str) -> list[list[str]]:
        """Every (stem, slot...) reading of `word`. Empty means not a word."""
        word = word.lower()
        out: list[list[str]] = []
        seen = set()
        for cut in range(1, len(word) + 1):
            stem, rest = word[:cut], word[cut:]
            for lemma, tail in self._seams(stem, rest):
                merged = self.lexicon.get(lemma)
                if merged is None:
                    continue
                for track in tracks_of(merged):
                    for path in self._walk(lemma, tail, track, None, merged):
                        reading = tuple([lemma] + path)
                        if reading not in seen:
                            seen.add(reading)
                            out.append(list(reading))
            features = self.lexicon.get(stem)
            elided_from = self.elision.get(stem)
            if features is None and elided_from is not None:
                features = self.lexicon.get(elided_from)
            voiced_from = None
            if features is None:
                voiced_from = devoice(stem)
                if voiced_from is not None:
                    features = self.lexicon.get(voiced_from)
            if features is None:
                continue
            # An alternated stem is not a word on its own, and the alternation
            # only happens because a vowel-initial suffix followed it.
            if (elided_from is not None and stem not in self.lexicon) or \
                    voiced_from is not None:
                if not rest or rest[0] not in VOWELS:
                    continue
            for track in tracks_of(features):
                for path in self._walk(stem, rest, track, None, features):
                    reading = tuple([stem] + path)
                    if reading not in seen:
                        seen.add(reading)
                        out.append(list(reading))
        return out

    def _seams(self, stem: str, rest: str):
        """(lemma, suffix) pairs where the join is spelt differently from both.

        These are not alternations of the stem or of the suffix — both are
        unchanged, and it is only the seam between them that a letter is
        written across. Undoing the spelling gives back the ordinary walk.
        """
        # `ы`/`і` and a following `й` are written as one `и`: `оқы` plus
        # `-йды` is `оқиды`, in 882 books against 7 for `оқыйды`.
        if stem.endswith("и"):
            for vowel in ("ы", "і"):
                yield stem[:-1] + vowel, "й" + rest
        # `й` and a following `а` are written as one `я`: `жай` plus `-атын`
        # is `жаятын`, in 139 books against none for `жайатын`.
        if stem.endswith("я"):
            yield stem[:-1] + "й", "а" + rest
        # A Russian loanword in `-ь` loses it before a vowel: `медаль` gives
        # `медалімен`, `секретарь` gives `секретары`. The soft sign is not a
        # segment and nothing can be attached across it.
        if rest[:1] in VOWELS:
            yield stem + "ь", rest

    def _walk(self, surface: str, rest: str, track: str, prev, features,
              derivations: int = 0):
        """Sequences of slots spelling `rest`, attaching after `surface`."""
        if not rest:
            yield []
            return
        if derivations > self.max_derivations:
            return
        if prev is not None and len(rest) > self.max_morpheme * MAX_SLOTS:
            return
        for cut in range(min(len(rest), self.max_morpheme), 0, -1):
            written = rest[:cut]
            # A derived stem voices its final stop before a vowel exactly as an
            # entry does — `-дық` plus `-ы` is `дығы` — and until now only
            # entries could. `болатындығын`, in 326 of 3,860 books, and every
            # other -лық noun carrying a possessive was unbuildable; what did
            # get through, `достығымыз`, got through by being read as `-ты`
            # plus `-ғы`, two derivations that are not there.
            spellings = [written]
            if rest[cut:cut + 1] in VOWELS:
                voiced = devoice(written)
                if voiced is not None:
                    spellings.append(voiced)
            for morpheme, slot in ((m, s) for m in spellings
                                   for s in self.tpl.by_morpheme.get(m, ())):
                if prev is None:
                    if slot.track != track or slot.after:
                        continue
                elif not self.tpl.may_follow(prev, slot):
                    continue
                if slot.requires and not set(slot.requires) <= features:
                    continue
                # The override is a fact about the entry, so it applies to the
                # suffix the entry takes and to nothing after it. Looking it up
                # by surface let it reach any intermediate string that happened
                # to be another entry: `кіріп` is one, and its column refused
                # `кіріппіз` the ordinary `кір` plus `-іп` plus `-піз`. Once a
                # suffix is on, the surface has vowels of its own.
                if not fits(surface, morpheme, slot.morphemes,
                            self.overrides if prev is None else None,
                            slot.a_after):
                    continue
                nxt = slot.restart or slot.track
                for tail in self._walk(surface + written, rest[cut:], nxt,
                                       slot, features,
                                       derivations + (slot.id in DERIVING)):
                    yield [f"{slot.id}:{written}"] + tail

    def accepts(self, word: str) -> bool:
        return bool(self.analyse(word))


@functools.lru_cache(maxsize=1)
def default() -> Analyser:
    return Analyser(load(), load_lexicon(ROOT / "data/lexicon.tsv",
                                        ROOT / "data/discovered.tsv",
                                        ROOT / "data/tracks.tsv"),
                    overrides=read_harmony(ROOT / "data/harmony.tsv"),
                    elision=read_elision(ROOT / "data/elision.tsv"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any word is accepted")
    ap.add_argument("--lexicon", type=Path, default=ROOT / "data/lexicon.tsv")
    args = ap.parse_args()

    an = Analyser(load(), load_lexicon(args.lexicon,
                                       ROOT / "data/discovered.tsv",
                                       ROOT / "data/tracks.tsv"),
                  overrides=read_harmony(ROOT / "data/harmony.tsv"),
                  elision=read_elision(ROOT / "data/elision.tsv"))
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
