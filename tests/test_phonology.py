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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

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

    # A word-final `у` carries no harmony of its own; anywhere else it is an
    # ordinary back vowel. `келуге` is in 817 books and `келуға` in none.
    ("barys", "келу",   "ге", "ға", "келуге"),
    ("barys", "оқу",    "ға", "ге", "оқуға"),
    ("barys", "су",     "ға", "ге", "суға"),
    ("barys", "армия",  "ға", "ге", "армияға"),
    ("barys", "мектеп", "ке", "ға", "мектепке"),
]

# No geminate glide across a boundary: `болу` does not take a second тұйық
# етістік, and `сүй` takes `сүйеді` rather than `*сүййді`.
NO_GEMINATE = [("tuiyq", "болу", "у"), ("shaq", "сүй", "йді")]

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

    for slot_id, stem, want, refuse, word in CASES:
        got = shapes(slot_id, stem)
        if want not in got:
            bad.append(f"{slot_id} on {stem!r} will not take {want!r} "
                       f"({word}); it offers {got}")
        if refuse in got:
            bad.append(f"{slot_id} on {stem!r} takes {refuse!r}, "
                       f"but the word is {word}")
    for slot_id, stem, refuse in NO_GEMINATE:
        if refuse in shapes(slot_id, stem):
            bad.append(f"{slot_id} on {stem!r} takes {refuse!r}, "
                       f"doubling a glide across the boundary")
    for word, want in HARMONY:
        got = harmony(word)
        if got != want:
            bad.append(f"harmony({word!r}) is {got}, want {want}")

    for line in bad:
        print(f"  {line}")
    if bad:
        print(f"\n{len(bad)} failures")
        return 1
    print(f"{len(CASES) * 2 + len(NO_GEMINATE) + len(HARMONY)} cases pass: "
          f"{len(CASES)} allomorph choices both ways, "
          f"{len(NO_GEMINATE)} geminates refused, "
          f"{len(HARMONY)} harmony classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
