"""Which shape each slot takes on a given stem.

Every case is decided by the 3,860 editions rather than by a grammar's say-so,
and the book counts are quoted because several of these were wrong in a way
only a corpus catches.

What is tested is the allomorph choice alone, not whether the whole word is
accepted: `қалады` is a real word by another reading — `қал` plus `-ады` — and
asking the analyser about it would answer a different question than the one
this file is for. So each case names the slot, the stem, and the shape that
slot must and must not take there.

    python tests/test_phonology.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from analyse import (Analyser, read_elision, read_harmony,  # noqa: E402
                     read_lexicon)
from phonology import harmony, realise  # noqa: E402
from template import load  # noqa: E402

# (slot, stem, the shape it takes, the shape it does not, the word at issue)
CASES = [
    # The genitive and the accusative alternate on the same н/д/т and divide
    # the finals differently. Reading the partition off the letter `н` gets one
    # of the two wrong whichever way it is written, and it got the accusative
    # wrong: `жанды` is in 1,851 books and `жанны` in 9.
    ("tabys", "жан",  "ды",  "ны",  "жанды"),
    ("tabys", "күн",  "ді",  "ні",  "күнді"),
    ("ilik",  "жан",  "ның", "дың", "жанның"),
    ("tabys", "қала", "ны",  "ды",  "қаланы"),
    ("ilik",  "қала", "ның", "дың", "қаланың"),

    # `у` spells a glide, not a vowel, so a stem ending in one is
    # consonant-final: `тауды` 757 books against `тауны` 3, `тауым` 215
    # against `таум` 0. `и` and `ю` are the same; `я` is not, it ends in /a/.
    ("tabys",    "тау",    "ды",  "ны",  "тауды"),
    ("ilik",     "тау",    "дың", "ның", "таудың"),
    ("tauelik",  "тау",    "ым",  "м",   "тауым"),
    ("tabys",    "су",     "ды",  "ны",  "суды"),
    ("ilik",     "аю",     "дың", "ның", "аюдың"),
    ("tabys",    "такси",  "ді",  "ды",  "таксиді"),
    ("ilik",     "үй",     "дің", "нің", "үйдің"),
    ("tabys",    "армия",  "ны",  "ды",  "армияны"),
    # But the plural and the instrumental count the glides with the sonorants.
    ("koptik",   "тау",    "лар", "дар", "таулар"),
    ("komektes", "тау",    "мен", "бен", "таумен"),

    # The ablative after a nasal is -нан/-нен, and it was missing outright, so
    # every noun ending in one lost its ablative: `құмнан` is in 222 books and
    # `құмдан` in none. It is not the -нан that follows a possessive.
    ("shygys", "құм",  "нан", "дан", "құмнан"),
    ("shygys", "әлем", "нен", "ден", "әлемнен"),
    ("shygys", "қала", "дан", "нан", "қаладан"),
    ("shygys", "тау",  "дан", "нан", "таудан"),
    ("shygys", "ел",   "ден", "нен", "елден"),

    # `-нікі` follows a vowel and `-дікі` a nasal, which is the opposite of
    # what the genitive does on the same letter: `әкемдікі` 6 books, `әкемнікі`
    # none. (`менікі` is the pronoun being irregular, and is held as an entry.)
    ("tauelik_of", "әкем", "дікі", "нікі", "әкемдікі"),
    ("tauelik_of", "бала", "нікі", "дікі", "баланікі"),

    # The other hiatus repair, the same as `ым`/`м` the other way round: a
    # consonant-final stem takes `аты`, a vowel-final one takes `баласы`.
    ("tauelik", "бала", "сы", "ы",  "баласы"),
    ("tauelik", "ат",   "ы",  "сы", "аты"),
    # And the -н cases, which only ever follow a possessive and so only ever
    # follow a vowel.
    ("tabys_px", "қаласы", "н", None, "қаласын"),

    # A word-final `у` carries no harmony of its own; anywhere else it is an
    # ordinary back vowel. `келуге` is in 817 books and `келуға` in none.
    ("barys", "келу",   "ге", "ға", "келуге"),
    ("barys", "оқу",    "ға", "ге", "оқуға"),
    ("barys", "су",     "ға", "ге", "суға"),
    ("barys", "армия",  "ға", "ге", "армияға"),
    ("barys", "мектеп", "ке", "ға", "мектепке"),

    # Negation takes -ба after a nasal and -ма after a liquid, which is the
    # plural's partition and not the instrumental's, though all three alternate
    # on м/б/п. `сенбейді` is in 483 books and `сенмейді` in none; `көнбейді`
    # 504 and `көнмейді` none; `келмейді` 1,899 and `келбейді` none.
    ("bolymsyz", "сен", "бе", "ме", "сенбейді"),
    ("bolymsyz", "көн", "бе", "ме", "көнбейді"),
    ("bolymsyz", "жан", "ба", "ма", "жанбайды"),
    ("bolymsyz", "кел", "ме", "бе", "келмейді"),
    ("bolymsyz", "бер", "ме", "бе", "бермейді"),
    ("bolymsyz", "оқы", "ма", "ба", "оқыма"),
    ("bolymsyz", "жаз", "ба", "ма", "жазбайды"),
    ("bolymsyz", "айт", "па", "ба", "айтпайды"),

    # `-дай` on a bare stem and after a possessive: `баладай`, `баласындай`.
    ("salystyru",    "бала",   "дай",  "тай",  "баладай"),
    ("salystyru",    "ат",     "тай",  "дай",  "аттай"),
    ("salystyru",    "ер",     "дей",  "тей",  "ердей"),
    ("salystyru_px", "баласы", "ндай", "ндей", "баласындай"),

    # The long participle is front on `-етұғын` and back on `-атұғын`, which
    # only a suffix read from its first vowel can tell apart: both end in `ұғын`.
    ("esimshe", "кел", "етұғын", "атұғын", "келетұғын"),
    ("esimshe", "бол", "атұғын", "етұғын", "болатұғын"),
    # And `-ушы`/`-уші` still divide the other way, on their last vowel, which
    # is why the `у` is skipped rather than counted.
    ("esimshe", "оқы", "ушы", "уші", "оқушы"),
    ("esimshe", "кел", "уші", "ушы", "келуші"),

    # A soft sign is not a segment: it is the `л` of `мораль` that picks `-ға`.
    ("barys", "мораль", "ға", "қа", "моральға"),
    ("ilik",  "февраль", "дың", "тың", "февральдың"),
]

# No geminate glide across a boundary: `болу` does not take a second тұйық
# етістік, and `сүй` takes `сүйеді` rather than `*сүййді`.
NO_GEMINATE = [("tuiyq", "болу", "у"), ("shaq_osy", "сүй", "йді")]

# Four persons, not four shapes of one suffix, so nothing may confine `-қ` to
# the finals a series would give it — `барды` has to reach it. What does apply
# is harmony, on the consonant, because neither morpheme carries a vowel.
PERSONAL = [
    ("jiktik_otken", "барды", "қ",  "к",  "бардық"),
    ("jiktik_otken", "келді", "к",  "қ",  "келдік"),
    ("jiktik_otken", "барса", "қ",  "к",  "барсақ"),
    ("jiktik_otken", "келсе", "к",  "қ",  "келсек"),
    ("jiktik_otken", "барды", "м",  None, "бардым"),
    ("jiktik_otken", "барды", "ңыз", "ңіз", "бардыңыз"),
    # And the second person of the other series is not a series member either:
    # `-сың` is the whole ending, so it follows a consonant perfectly well.
    ("jiktik_v", "келмей", "сің", None, "келмейсің"),
    ("jiktik_n", "барған", "сың", None, "барғансың"),
]

# The three alternations that change the stem rather than the suffix, tested as
# whole words because that is the only place they show.
#
# Only one way round. An alternation adds a spelling of the seam; it does not
# make the unalternated one impossible, and the unalternated one is usually a
# legal walk on its own account — `оқыйды` is `оқы` plus the көсемше `-й` plus
# `-ды`, and `мойынына` is `мойын` with an ordinary possessive. Whether those
# should be refused is a question about the walks, not about the seam.
STEMS = [
    # A closed class drops the vowel of its last syllable before a vowel.
    ("мойнына",  "мойын drops its vowel"),
    ("халқы",    "халық likewise"),
    # A final voiceless stop voices before a vowel.
    ("мектебін",  "мектеп voices its п"),
    ("амандығын", "амандық voices its қ"),
    # A final ы or і and a following й are written as one и: `оқиды` is in 882
    # books and `оқыйды` in 7.
    ("оқиды",  "оқы + -йды"),
    ("ашитын", "ашы + -йтын"),
    ("ериді",  "ері + -йді"),
    ("оқиын",  "оқы + -йын"),
    ("құритын", "құры + -йтын"),
    # `й` and a following `а` are written as one `я`.
    ("жаятын", "жай + -атын"),
    ("тыятын", "тый + -атын"),
    # A loanword in `-ь` loses it before a vowel.
    ("секретары", "секретарь + -ы"),
]

# Whole words again, for the two slots whose point is where they sit in the
# chain rather than which shape they take.
WALKS = [
    ("баласындай",  "-дай reaches past тәуелдік"),
    ("қолындай",    "and past the -н that follows it"),
    ("анаңдай",     "and after a second-person possessive"),
    ("барсаңшы",    "-шы comes after the person, not before it"),
    ("болсайшы",    "and takes a й after a vowel"),
    ("келсеңші",    "front"),
    ("ойнайық",     "-айық loses its vowel after a vowel-final stem"),
    ("көтерелік",   "-елік is the same person again"),
    ("сенбейді",    "negation after a nasal"),
    ("моральға",    "and a soft sign decides nothing"),
    # Per-lexeme harmony, read off the books because no rule finds it.
    ("тарихқа",     "тарих is back though its last vowel is front"),
    ("миының",      "and so is ми"),
    ("банкке",      "банк is front though its only vowel is back"),
    ("октябрьден",  "октябрь is front"),
]

HARMONY = [
    ("такси",       "front"),   # таксиді
    ("институт",    "back"),    # институтқа — on the у, not the и
    ("келу",        "front"),   # келуге
    ("оқу",         "back"),    # оқуға
    ("су",          "back"),    # суға
    ("тау",         "back"),    # тауға
    ("армия",       "back"),    # армияға
    ("аю",          "back"),    # аюға
    ("университет", "front"),   # университетке
]


def main() -> int:
    tpl = load()
    bad = []

    def shapes(slot_id: str, stem: str) -> list[str]:
        slot = tpl.by_id[slot_id]
        return realise(stem, slot.morphemes, None, slot.a_after)

    for slot_id, stem, want, refuse, word in CASES + PERSONAL:
        got = shapes(slot_id, stem)
        if want not in got:
            bad.append(f"{slot_id} on {stem!r} will not take {want!r} "
                       f"({word}); it offers {got}")
        if refuse is not None and refuse in got:
            bad.append(f"{slot_id} on {stem!r} takes {refuse!r}, "
                       f"but the word is {word}")
    for slot_id, stem, refuse in NO_GEMINATE:
        if refuse in shapes(slot_id, stem):
            bad.append(f"{slot_id} on {stem!r} takes {refuse!r}, "
                       f"doubling a glide across the boundary")
    an = Analyser(tpl, read_lexicon(ROOT / "data/lexicon.tsv"),
                  overrides=read_harmony(ROOT / "data/harmony.tsv"),
                  elision=read_elision(ROOT / "data/elision.tsv"))
    for word, why in STEMS + WALKS:
        if not an.accepts(word):
            bad.append(f"{word!r} refused — {why}")
    for word, want in HARMONY:
        got = harmony(word)
        if got != want:
            bad.append(f"harmony({word!r}) is {got}, want {want}")

    for line in bad:
        print(f"  {line}")
    if bad:
        print(f"\n{len(bad)} failures")
        return 1
    checked = sum(2 if refuse is not None else 1
                  for _s, _st, _w, refuse, _word in CASES + PERSONAL)
    print(f"{checked + len(NO_GEMINATE) + len(STEMS) + len(WALKS) + len(HARMONY)} "
          f"cases pass: {len(CASES)} allomorph choices, "
          f"{len(PERSONAL)} personal endings, "
          f"{len(NO_GEMINATE)} geminates refused, "
          f"{len(STEMS)} stem alternations, {len(WALKS)} walks, "
          f"{len(HARMONY)} harmony classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
