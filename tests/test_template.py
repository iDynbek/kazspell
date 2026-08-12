"""What the template must and must not allow, in word forms rather than tables.

Every case is a real Kazakh word or a real error, written as the slot sequence
it decomposes into. The template is the thing being tested; the phonology that
chooses between allomorphs is the engine's, and is not exercised here.

    python tests/test_template.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from template import load  # noqa: E402

# (word, starting track, slot sequence) — sequences the template must permit.
LEGAL = [
    # мектеп-тер-іміз-де: the form the whole two-level design existed to reach.
    ("мектептерімізде", "n", ["koptik", "tauelik", "septik"]),
    ("баламен",         "n", ["septik"]),
    ("баламын",         "n", ["jiktik_n"]),
    ("қалалардың",      "n", ["koptik", "septik"]),
    ("достығымыз",      "n", ["sozjasam", "tauelik"]),
    # The attributive attaches after the case, not before it, and the result
    # inflects again: үй-де-гі-лер-дің.
    ("үйдегі",          "n", ["septik", "attr"]),
    ("үйдегілердің",    "n", ["septik", "attr", "koptik", "septik"]),
    ("әкемдікі",        "n", ["tauelik", "tauelik_of"]),
    ("менікін",         "n", ["tauelik_of", "septik"]),
    # A participle becomes a nominal and inflects as one: бар-ған-ым.
    ("барғаным",        "v", ["esimshe", "tauelik"]),
    ("барғандардың",    "v", ["esimshe", "koptik", "septik"]),
    # A verbal noun does the same: оқу-ы-на.
    ("оқуына",          "v", ["tuiyq", "tauelik", "septik"]),
    # Voice, then negation, then a finite tense.
    ("жазылды",         "v", ["etis_pass", "shaq"]),
    ("жазылмады",       "v", ["etis_pass", "bolymsyz", "shaq"]),
    ("айтқызды",        "v", ["etis_caus", "shaq"]),
    ("көрісті",         "v", ["etis_recip", "shaq"]),
    ("айтпайды",        "v", ["bolymsyz", "shaq"]),
    ("келсең",          "v", ["rai", "jiktik_v"]),
    ("айтып",           "v", ["kosemshe"]),
    # Denominal verb formation restarts the chain on the verbal track.
    ("трансформациялау", "n", ["etis_jasam", "tuiyq"]),
    ("реттелді",        "n", ["etis_jasam", "etis_pass", "shaq"]),
]

# Sequences the template must refuse, and why.
ILLEGAL = [
    ("*мектепдетер",  "n", ["septik", "koptik"],
     "көптік comes before септік, never after"),
    ("*баламызлар",   "n", ["tauelik", "koptik"],
     "тәуелдік comes after көптік"),
    ("*барғанатын",   "v", ["esimshe", "esimshe"],
     "one form from the finiteness group, not two"),
    ("*келсеп",       "v", ["rai", "kosemshe"],
     "рай and көсемше are both finiteness; they exclude each other"),
    ("*жазғанды",     "v", ["esimshe", "shaq"],
     "after a participle the chain is nominal; шақ is verbal"),
    ("*балады",       "n", ["shaq"],
     "a finite tense does not attach to a noun"),
    ("*жаздылар",     "v", ["shaq", "koptik"],
     "көптік is nominal and шақ does not restart the chain"),
]

# Only a transitive verb has a passive.
FEATURES = [
    ("жазылды",  "v", frozenset({"v-trans"}),   "etis_pass", True,
     "жаз is transitive, so the passive exists"),
    ("*отырылды", "v", frozenset({"v-intrans"}), "etis_pass", False,
     "отыр is intransitive; there is no passive to form"),
]


def walk(tpl, track, ids, features=frozenset()):
    """Whether the sequence of slot ids is a legal walk from `track`."""
    prev = None
    for slot_id in ids:
        slot = tpl.by_id.get(slot_id)
        if slot is None:
            return False, f"no slot {slot_id!r}"
        if slot.requires and not set(slot.requires) <= features:
            return False, f"{slot_id} requires {slot.requires}"
        if prev is None:
            if slot.track != track:
                return False, f"{slot_id} is track {slot.track}, not {track}"
        elif not tpl.may_follow(prev, slot):
            return False, f"{slot_id} may not follow {prev.id}"
        prev = slot
    return True, ""


def main() -> int:
    tpl = load()
    failures = []

    for word, track, ids in LEGAL:
        ok, why = walk(tpl, track, ids, frozenset({"v-trans"}))
        if not ok:
            failures.append(f"{word}: should be legal but {why}")

    for word, track, ids, reason in ILLEGAL:
        ok, _ = walk(tpl, track, ids, frozenset({"v-trans"}))
        if ok:
            failures.append(f"{word}: should be refused — {reason}")

    for word, track, features, slot_id, expected, reason in FEATURES:
        ok, _ = walk(tpl, track, [slot_id], features)
        if ok != expected:
            failures.append(f"{word}: {reason}")

    # Every morpheme the template knows must be reachable from some stem.
    for slot in tpl.slots:
        for morpheme in slot.morphemes:
            if tpl.opening(morpheme, slot.track,
                           frozenset({"v-trans", "v-intrans"})) is None:
                failures.append(f"{slot.id}: {morpheme!r} is unreachable")

    total = len(LEGAL) + len(ILLEGAL) + len(FEATURES)
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(failures)} failures over {total} cases")
        return 1
    print(f"{total} cases pass: {len(LEGAL)} legal forms, {len(ILLEGAL)} refused, "
          f"{len(FEATURES)} feature-gated, every morpheme reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
