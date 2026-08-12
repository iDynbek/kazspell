"""Which shape a suffix takes, and whether a given shape is the right one.

The template says a plural is one of `лар лер дар дер тар тер`. Which one is not
a choice — the stem decides, twice. Vowel harmony picks the column from the
stem's last vowel; the stem's final segment picks the row. Getting either wrong
produces a string that looks like a word and is not, `мектепдер` for
`мектептер`, which is what the misspelling benchmarks are made of.

Which finals license which shape is not one table. `-лар` follows a vowel and
`-дар` follows `л`; the genitive `-ның` follows a nasal where the plural takes
`-дар`. Assuming a single three-way split rejected `жыланның`, in 422 of 3,860
books, and every word shaped like it.

So the series are not declared — they are read off the template. Morphemes in a
slot that differ only in their first consonant are one alternation, and their
number decides how the groups divide. Adding an allomorph to `template.toml`
therefore cannot leave the phonology out of step with it.
"""

from __future__ import annotations

import collections
import functools

BACK = set("аоұыу")
FRONT = set("әөүіеё")
VOWELS = BACK | FRONT

# Kazakh does not have one three-way split, it has several, and which one
# applies is decided by the series itself. The plural takes `-лар` after a
# vowel but `-дар` after `л`: `балалар`, `елдер`. The genitive takes `-ның`
# after a nasal where the plural takes `-дар`: `жыланның`, `жыландар`. The
# instrumental counts every sonorant as one group. Assuming a single partition
# rejected `жыланның` — 422 of 3,860 books — and every word like it.
#
# So the partition is chosen by the A-member's initial consonant, which is what
# names the series.
A_INITIALS = set("лнмс")
B_INITIALS = set("дбғг")
C_INITIALS = set("тпқк")

SONORANT_RUJ = set("руй")
SONORANT_L = set("л")
NASAL = set("мнң")
FRICATIVE = set("жз")
VOICED = SONORANT_RUJ | SONORANT_L | NASAL | FRICATIVE

# Everything else that ends a word is voiceless for this purpose, б в г д
# included: they devoice finally, so `клуб` takes `клубқа`.
def _a_set(initial: str) -> set[str]:
    if initial == "л":
        return VOWELS | SONORANT_RUJ
    if initial in "нс":
        return VOWELS | (NASAL if initial == "н" else set())
    if initial == "м":
        return VOWELS | SONORANT_RUJ | SONORANT_L | NASAL
    return VOWELS


def _b_set(initial: str) -> set[str]:
    return (VOWELS | VOICED) - _a_set(initial)


# A stem-final voiceless stop voices before a vowel-initial suffix: `мектеп` is
# `мектебі`, `амандық` is `амандығы`, `тілек` is `тілегі`. The surface a suffix
# attaches to is therefore not the entry, and a recogniser that only ever looks
# the entry up rejects `мектебін` — which it did, along with everything else
# whose stem ends in one of these three letters.
VOICING = {"б": "п", "ғ": "қ", "г": "к"}


def devoice(stem: str) -> str | None:
    """The entry a voiced surface stem could have come from."""
    if stem and stem[-1] in VOICING:
        return stem[:-1] + VOICING[stem[-1]]
    return None


# Suffixes whose vowels do not harmonise, so their vowels are no evidence about
# the stem. Reading `-мен` as front made every stem carrying one look front.
INVARIANT = {"мен", "бен", "пен", "менен", "бенен", "пенен",
             "нікі", "дікі", "тікі"}


def harmony(word: str) -> str | None:
    for ch in reversed(word):
        if ch in BACK:
            return "back"
        if ch in FRONT:
            return "front"
    return None


def morpheme_harmony(morpheme: str) -> str | None:
    return None if morpheme in INVARIANT else harmony(morpheme)


@functools.lru_cache(maxsize=None)
def series_finals(morphemes: tuple[str, ...]) -> dict[str, frozenset[str]]:
    """morpheme -> the stem-final letters it may follow, read off the slot.

    Morphemes differing only in their initial consonant are one alternation.
    Three members divide the finals three ways, with the split chosen by the
    A-member; two members divide voiced against voiceless.
    """
    families: dict[tuple[str, str | None], list[str]] = collections.defaultdict(list)
    for m in morphemes:
        if m and m[0] not in VOWELS:
            families[(m[1:], morpheme_harmony(m))].append(m)

    out: dict[str, frozenset[str]] = {}
    for members in families.values():
        initials = {m[0] for m in members}
        a_initial = next((i for i in initials if i in A_INITIALS), None)
        for m in members:
            first = m[0]
            if first in A_INITIALS:
                out[m] = frozenset(_a_set(first))
            elif first in B_INITIALS:
                out[m] = frozenset(_b_set(a_initial) if a_initial
                                   else VOWELS | VOICED)
            elif first in C_INITIALS:
                out[m] = frozenset()          # the voiceless finals: see below
    return out


def licensed(stem: str, morpheme: str, morphemes: tuple[str, ...]) -> bool:
    finals = series_finals(morphemes)
    if morpheme not in finals:
        return True
    last = stem[-1:]
    if morpheme[0] in C_INITIALS:
        return last not in (VOWELS | VOICED)   # only the voiceless finals
    return last in finals[morpheme]


@functools.lru_cache(maxsize=None)
def linking_pairs(morphemes: tuple[str, ...]) -> frozenset[str]:
    """The bare members of a pair like `ым`/`м`, which need a vowel-final stem."""
    have = set(morphemes)
    return frozenset(m for m in morphemes
                     if ("ы" + m) in have or ("і" + m) in have)


def fits(stem: str, morpheme: str, slot_morphemes: tuple[str, ...],
         overrides: dict[str, str] | None = None) -> bool:
    """Whether `morpheme` is the shape this stem licenses for its slot.

    `overrides` carries stems whose harmony the spelling does not predict:
    `банк` is written with a back vowel and takes `банкке`, while `фильм`,
    `министр` and `брифинг` take front endings. No rule over the letters
    recovers that, so it is held per lexeme.
    """
    if not morpheme:
        return True
    want = morpheme_harmony(morpheme)
    if want is not None:
        have = (overrides or {}).get(stem) or harmony(stem)
        if have is not None and have != want:
            return False

    ends_vowel = stem[-1:] in VOWELS
    if morpheme[0] in VOWELS:
        return not ends_vowel          # `ат-ым`, never `бала-ым`
    if morpheme in linking_pairs(slot_morphemes):
        return ends_vowel              # `бала-м`, never `ат-м`

    return licensed(stem, morpheme, slot_morphemes)


def realise(stem: str, slot_morphemes: tuple[str, ...],
            overrides: dict[str, str] | None = None) -> list[str]:
    """The shapes of this slot that the stem licenses, longest first."""
    return sorted((m for m in slot_morphemes if fits(stem, m, slot_morphemes, overrides)),
                  key=len, reverse=True)
