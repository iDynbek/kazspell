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

from analyse import tracks_of  # noqa: E402
from template import load  # noqa: E402

# Every part of speech that appears in data/lexicon.tsv.
EVERY_POS = ["n", "np", "adj", "adv", "num", "pron", "det", "post", "interj",
             "conj", "abbr", "v", "v-trans", "v-intrans", "v-aux"]

# (word, starting track, slot sequence) — sequences the template must permit.
LEGAL = [
    # мектеп-тер-іміз-де: the form the whole two-level design existed to reach.
    ("мектептерімізде", "n", ["koptik", "tauelik", "jatys"]),
    ("баламен",         "n", ["komektes"]),
    ("баламын",         "n", ["jiktik_n"]),
    ("қалалардың",      "n", ["koptik", "ilik"]),
    ("достығымыз",      "n", ["sozjasam", "tauelik"]),
    # The attributive attaches after the case, not before it, and the result
    # inflects again: үй-де-гі-лер-дің.
    ("үйдегі",          "n", ["jatys", "attr"]),
    ("үйдегілердің",    "n", ["jatys", "attr", "koptik", "ilik"]),
    ("әкемдікі",        "n", ["tauelik", "tauelik_of"]),
    ("менікін",         "n", ["tauelik_of", "tabys"]),
    # A participle becomes a nominal and inflects as one: бар-ған-ым.
    ("барғаным",        "v", ["esimshe", "tauelik"]),
    ("барғандардың",    "v", ["esimshe", "koptik", "ilik"]),
    # A verbal noun does the same: оқу-ы-на.
    ("оқуына",          "v", ["tuiyq", "tauelik", "barys_px"]),
    # Voice, then negation, then a finite tense.
    ("жазылды",         "v", ["etis_pass", "shaq_jedel"]),
    ("жазылмады",       "v", ["etis_pass", "bolymsyz", "shaq_jedel"]),
    ("айтқызды",        "v", ["etis_caus", "shaq_jedel"]),
    ("көрісті",         "v", ["etis_recip", "shaq_jedel"]),
    ("айтпайды",        "v", ["bolymsyz", "shaq_osy"]),
    # The past and the conditional take their own personal series: бар-ды-қ,
    # кел-се-к. It is not the -мын/-сың of барармын, and nothing else takes it.
    ("келсең",          "v", ["rai_shart", "jiktik_otken"]),
    ("бардық",          "v", ["shaq_jedel", "jiktik_otken"]),
    ("барармын",        "v", ["esimshe", "jiktik_n"]),
    ("барасың",         "v", ["kosemshe", "jiktik_v"]),
    ("айтып",           "v", ["kosemshe"]),
    # Denominal verb formation restarts the chain on the verbal track.
    ("трансформациялау", "n", ["etis_jasam", "tuiyq"]),
    ("реттелді",        "n", ["etis_jasam", "etis_pass", "shaq_jedel"]),
]

# Sequences the template must refuse, and why.
ILLEGAL = [
    ("*мектепдетер",  "n", ["jatys", "koptik"],
     "көптік comes before септік, never after"),
    ("*баламызлар",   "n", ["tauelik", "koptik"],
     "тәуелдік comes after көптік"),
    ("*барғанатын",   "v", ["esimshe", "esimshe"],
     "one form from the finiteness group, not two"),
    ("*келсеп",       "v", ["rai_shart", "kosemshe"],
     "рай and көсемше are both finiteness; they exclude each other"),
    ("*жазғанды",     "v", ["esimshe", "shaq_jedel"],
     "after a participle the chain is nominal; шақ is verbal"),
    ("*балады",       "n", ["shaq_jedel"],
     "a finite tense does not attach to a noun"),
    ("*жаздылар",     "v", ["shaq_jedel", "koptik"],
     "көптік is nominal and шақ does not restart the chain"),
    ("*барадым",      "v", ["shaq_osy", "jiktik_otken"],
     "the -м series follows the past and the conditional, not the present"),
    ("*барсыным",     "v", ["rai", "jiktik_otken"],
     "nor the imperative, which already carries its person"),
    ("*қаланан",      "n", ["shygys_px"],
     "the -н ablative appears only after a possessive"),
    ("*үйдесі",       "n", ["jatys", "tauelik"],
     "тәуелдік comes before септік, not after"),
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
            if slot.after:
                return False, f"{slot_id} may only follow {slot.after}"
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

    # Every part of speech the lexicon uses has to put an entry on some track.
    # Naming neither list left `не`, `және` and `түгіл` — 3,554, 2,740 and
    # 1,304 of the 3,860 editions — on no track at all, unbuildable even bare.
    for pos in EVERY_POS:
        if not tracks_of(frozenset({pos})):
            failures.append(f"an entry tagged {pos!r} is put on no track")
    if tracks_of(frozenset({"a-tag-nobody-has-added-yet"})) != ("n", "v"):
        failures.append("an unknown part of speech should open both tracks "
                        "rather than close both")

    total = len(LEGAL) + len(ILLEGAL) + len(FEATURES) + len(EVERY_POS) + 1
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        print(f"\n{len(failures)} failures over {total} cases")
        return 1
    print(f"{total} cases pass: {len(LEGAL)} legal forms, {len(ILLEGAL)} refused, "
          f"{len(FEATURES)} feature-gated, {len(EVERY_POS) + 1} parts of "
          f"speech placed, every morpheme reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
