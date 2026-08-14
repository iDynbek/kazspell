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

Morphemes in a slot that differ only in their first consonant are one
alternation; which finals each member follows is stated by the slot, in
`a_after`, because the letter cannot carry it. The genitive and the accusative
both alternate `н`/`д`/`т` and they divide the finals differently — `жанның`
but `жанды` — so nothing about the letter `н` predicts either one.
"""

from __future__ import annotations

import collections
import functools

# Kazakh spells two glides with letters that also spell vowels: `у` is /w/ in
# `тау` and `и` is /j/ in `такси`, and `ю` ends in /w/ too. A stem ending in one
# takes the consonant-final shape — `тауды` not `*тауны`, `тауым` not `*таум` —
# so they are not vowels for the purpose of picking a suffix. `я` is not one of
# them: it ends in /a/, and `армияны` behaves as vowel-final.
VOWELS = set("аәоөұүыіеёя")
GLIDES = set("уийю")

# Kazakh does not have one three-way split, it has several, and which one
# applies is not recoverable from the alternation's letters. The plural takes
# `-лар` after a vowel but `-дар` after `л`: `балалар`, `елдер`. The genitive
# takes `-ның` after a nasal where the plural takes `-дар`: `жыланның`,
# `жыландар`. And the accusative alternates on the very same `н`/`д`/`т` as the
# genitive while dividing the finals differently — `жанның` against `жанды`,
# 1,392 books against 1,851 — so reading the partition off the letter `н` gets
# one of the two wrong whichever way it is written.
#
# So each slot states what its A-member follows, as `a_after` in the template,
# in these classes. The B-member takes the rest of the voiced finals and the
# C-member the voiceless ones, which do follow from the alternation.
A_INITIALS = set("лнмс")
B_INITIALS = set("дбғг")
C_INITIALS = set("тпқк")

# The four consonant series Kazakh actually alternates on. Nothing outside them
# is an alternation, however alike two morphemes look: `-мыз` and `-сыз` differ
# in one letter and are the first plural and the second formal, not two shapes
# of one ending, and `-йін` and `-сін` likewise. Grouping either pair put a
# real ending behind the other's conditions — and because the group was keyed
# on a set of letters, which of the two lost was decided by iteration order.
SERIES = (frozenset("лдт"), frozenset("ндт"),
          frozenset("мбп"), frozenset("ғгқк"))

RHOTIC = set("р")
LIQUID = set("л")
NASAL = set("мнң")
FRICATIVE = set("жз")
VOICED = GLIDES | RHOTIC | LIQUID | NASAL | FRICATIVE

CLASSES = {"vowel": VOWELS, "glide": GLIDES, "r": RHOTIC, "liquid": LIQUID,
           "nasal": NASAL, "fricative": FRICATIVE}

# Everything else that ends a word is voiceless for this purpose, б в г д
# included: they devoice finally, so `клуб` takes `клубқа`.
def _a_set(initial: str) -> set[str]:
    if initial == "л":
        return VOWELS | GLIDES | RHOTIC
    if initial in "нс":
        return VOWELS | (NASAL if initial == "н" else set())
    if initial == "м":
        return VOWELS | GLIDES | RHOTIC | LIQUID | NASAL
    return VOWELS


def _b_set(a_set: set[str]) -> set[str]:
    return (VOWELS | VOICED) - a_set


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
             "нікі", "дікі", "тікі",
             "у"}   # the тұйық етістік: `келу` is front, `оқу` is back


# Harmony classes. `и`, `я` and `ю` belonged to neither, so `такси`, `армия`
# and `аю` had no harmony at all and took both columns of every suffix.
BACK = set("аоұыуяю")
FRONT = set("әөүіеёи")


def harmony(word: str) -> str | None:
    """Which column of a suffix this stem takes, from its last vowel.

    A word-final `у` is not that vowel. `келуге` is front and `оқуға` is back,
    so the тұйық етістік carries no harmony of its own and the stem under it
    decides; reading it as back put every verbal noun in the wrong column. In
    any other position `у` is an ordinary back vowel — `институтқа`, not
    `*институтке` — and when it is the only one the word is back: `суға`.
    """
    stem = word[:-1] if word.endswith("у") else word
    for ch in reversed(stem):
        if ch in BACK:
            return "back"
        if ch in FRONT:
            return "front"
    return "back" if stem != word else None


# `қ` and `ғ` are the back consonants and `к` and `г` the front ones, which is
# the only thing that distinguishes the first plural of `бардық` from that of
# `келдік`: the morpheme is a single letter and carries no vowel to harmonise.
VELAR_HARMONY = {"қ": "back", "ғ": "back", "к": "front", "г": "front"}


def morpheme_harmony(morpheme: str) -> str | None:
    if morpheme in INVARIANT:
        return None
    return harmony(morpheme) or VELAR_HARMONY.get(morpheme)


def _alternations(members: list[str]) -> list[list[str]]:
    """Split morphemes that differ only in their initial into real series.

    Largest series first, so `-ның/-дың/-тың` is read as the н series and not
    as the л one minus its `-лың`. What no series claims stands alone, and
    standing alone is what makes it unconditioned.
    """
    out, rest = [], list(members)
    while rest:
        initials = {m[0] for m in rest}
        series = max(SERIES, key=lambda s: len(s & initials))
        if len(series & initials) < 2:
            return out + [[m] for m in rest]
        out.append([m for m in rest if m[0] in series])
        rest = [m for m in rest if m[0] not in series]
    return out


@functools.lru_cache(maxsize=None)
def series_finals(morphemes: tuple[str, ...],
                  a_after: tuple[str, ...] = ()) -> dict[str, frozenset[str]]:
    """morpheme -> the stem-final letters it may follow.

    Morphemes differing only in their initial consonant are one alternation.
    `a_after` names the classes the A-member follows; the B-member takes the
    remaining voiced finals and the C-member the voiceless ones. Without it the
    partition is guessed from the A-member's initial, which is right for the
    slots that have no homograph to be confused with and wrong for the two that
    do.
    """
    declared = set().union(*(CLASSES[c] for c in a_after)) if a_after else None

    families: dict[tuple[str, str | None], list[str]] = collections.defaultdict(list)
    for m in morphemes:
        if not m or m[0] in VOWELS:
            continue
        # A one-letter morpheme is no evidence of anything. `-м`, `-ң`, `-қ`
        # and `-к` are four persons of the past, not four shapes of one
        # suffix, and reading them as a series put `-қ` among the voiceless
        # finals — where `барды` cannot reach it, so `бардық` was not a word.
        # Where a single letter really does alternate the slot declares it.
        if not m[1:] and declared is None:
            continue
        families[(m[1:], morpheme_harmony(m))].append(m)

    out: dict[str, frozenset[str]] = {}
    for members in (alt for family in families.values()
                    for alt in _alternations(family)):
        # One member is not an alternation. `-сың` is the whole second person
        # and nothing in жіктік competes with it, so reading its `с` as the
        # A-member of a series confined it to vowel-final stems and took
        # `келмейсің` and `барғансың` out of the language. A slot that does
        # alternate on a single written member says so with `a_after`.
        if len(members) == 1 and declared is None:
            continue
        initials = {m[0] for m in members}
        a_initial = next((i for i in initials if i in A_INITIALS), None)
        a_set = declared if declared is not None else (
            _a_set(a_initial) if a_initial else None)
        for m in members:
            first = m[0]
            if first in A_INITIALS:
                out[m] = frozenset(a_set if a_set is not None else VOWELS)
            elif first in B_INITIALS:
                out[m] = frozenset(_b_set(a_set) if a_set is not None
                                   else VOWELS | VOICED)
            elif first in C_INITIALS:
                out[m] = frozenset()          # the voiceless finals: see below
    return out


def licensed(stem: str, morpheme: str, morphemes: tuple[str, ...],
             a_after: tuple[str, ...] = ()) -> bool:
    finals = series_finals(morphemes, a_after)
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


@functools.lru_cache(maxsize=None)
def epenthetic(morphemes: tuple[str, ...]) -> frozenset[str]:
    """Members like `сы` beside `ы`, whose consonant is there to break hiatus.

    The same repair as `ым`/`м` and the other way round: a consonant-final stem
    takes `аты`, a vowel-final one takes `баласы`. It is not an alternation
    between two consonants, so the series machinery cannot see it, and without
    it `*атсы` is as licensed as `баласы`.
    """
    have = set(morphemes)
    return frozenset(m for m in morphemes
                     if len(m) > 1 and m[0] not in VOWELS
                     and m[1] in VOWELS and m[1:] in have)


def fits(stem: str, morpheme: str, slot_morphemes: tuple[str, ...],
         overrides: dict[str, str] | None = None,
         a_after: tuple[str, ...] = ()) -> bool:
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
    # No geminate glide across a boundary. `болу` does not take a second
    # тұйық етістік to give `*болуу`, and `сүй` takes `сүйеді`, not `*сүййді`.
    if morpheme[0] in GLIDES and stem[-1:] == morpheme[0]:
        return False
    if morpheme in linking_pairs(slot_morphemes):
        return ends_vowel              # `бала-м`, never `ат-м`
    if morpheme in epenthetic(slot_morphemes):
        return ends_vowel              # `бала-сы`, never `ат-сы`

    return licensed(stem, morpheme, slot_morphemes, a_after)


def realise(stem: str, slot_morphemes: tuple[str, ...],
            overrides: dict[str, str] | None = None,
            a_after: tuple[str, ...] = ()) -> list[str]:
    """The shapes of this slot that the stem licenses, longest first."""
    return sorted((m for m in slot_morphemes
                   if fits(stem, m, slot_morphemes, overrides, a_after)),
                  key=len, reverse=True)
