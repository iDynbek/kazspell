"""Load and check the suffix template, and say what it can and cannot describe.

`data/template.toml` is the specification; this reads it, refuses it if it is
incoherent, and offers the two questions everything else asks of it: which slot
does a morpheme belong to, and may this slot follow that one.

Keeping the answers here rather than in each tool is the point. hunspell-kk
learned the same grammar three times — once in the affix miner, once in the
paradigm grids, once in the family classifier — and they disagreed, which is
how `-ғы` came to be a nominal suffix in one place and the қалау рай in
another, and how a correct form was deleted for being an error.

    python tools/template.py            # validate and describe
    python tools/template.py --chains   # enumerate the legal slot sequences
"""

from __future__ import annotations

import argparse
import collections
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Slot:
    id: str
    name: str
    gloss: str
    track: str
    order: int
    morphemes: tuple[str, ...]
    group: str | None = None
    requires: tuple[str, ...] = ()
    restart: str | None = None
    after: tuple[str, ...] = ()


@dataclass
class Template:
    slots: tuple[Slot, ...]
    by_id: dict[str, Slot] = field(default_factory=dict)
    by_morpheme: dict[str, tuple[Slot, ...]] = field(default_factory=dict)

    def __post_init__(self):
        self.by_id = {s.id: s for s in self.slots}
        index: dict[str, list[Slot]] = collections.defaultdict(list)
        for slot in self.slots:
            for morpheme in slot.morphemes:
                index[morpheme].append(slot)
        self.by_morpheme = {m: tuple(v) for m, v in index.items()}

    def opening(self, suffix: str, track: str,
                features: frozenset[str] = frozenset()) -> Slot | None:
        """The slot the first morpheme of `suffix` belongs to, longest match.

        Longest-first matters and is not a tie-break: `лар` and `ларың` are both
        real, and cutting at the shorter one files a possessive as a plural.
        """
        for cut in range(len(suffix), 0, -1):
            for slot in self.by_morpheme.get(suffix[:cut], ()):
                if slot.track != track:
                    continue
                if slot.requires and not set(slot.requires) <= features:
                    continue
                return slot
        return None

    def may_follow(self, first: Slot, then: Slot) -> bool:
        """Whether `then` may attach after `first` within one stem's chain."""
        if then.after and first.id not in then.after:
            return False
        if first.restart:
            return then.track == first.restart      # the chain began again
        if then.track != first.track:
            return False
        if first.group and first.group == then.group:
            return False                            # one of the group, not two
        # `-нан` is the ablative a possessive leaves behind, and only that.
        if then.after and first.id not in then.after:
            return False
        return then.order > first.order


def load(path: Path | None = None) -> Template:
    path = path or ROOT / "data/template.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    slots = tuple(Slot(
        id=s["id"], name=s["name"], gloss=s["gloss"], track=s["track"],
        order=s["order"], morphemes=tuple(s["morphemes"]),
        group=s.get("group"), requires=tuple(s.get("requires", ())),
        restart=s.get("restart"), after=tuple(s.get("after", ())),
    ) for s in raw["slot"])
    problems = check(slots)
    if problems:
        for p in problems:
            print(f"template: {p}", file=sys.stderr)
        sys.exit(f"{len(problems)} problems in {path}")
    return Template(slots)


def check(slots: tuple[Slot, ...]) -> list[str]:
    """Everything that would make the template describe a language it should not."""
    problems = []
    seen = set()
    for slot in slots:
        if slot.id in seen:
            problems.append(f"duplicate slot id {slot.id!r}")
        seen.add(slot.id)
        if slot.track not in ("n", "v"):
            problems.append(f"{slot.id}: track {slot.track!r} is not n or v")
        if not slot.morphemes:
            problems.append(f"{slot.id}: no morphemes")
        if len(set(slot.morphemes)) != len(slot.morphemes):
            dupes = [m for m, n in collections.Counter(slot.morphemes).items() if n > 1]
            problems.append(f"{slot.id}: repeated morphemes {dupes}")
        if slot.restart and slot.restart not in ("n", "v"):
            problems.append(f"{slot.id}: restart {slot.restart!r} is not n or v")
        for other in slot.after:
            if other not in {x.id for x in slots}:
                problems.append(f"{slot.id}: after names unknown slot {other!r}")
    # A group whose members sit at different orders is not a choice between
    # them; the order would decide, and the exclusivity would never be reached.
    groups = collections.defaultdict(set)
    for slot in slots:
        if slot.group:
            groups[slot.group].add(slot.order)
    for group, orders in groups.items():
        if len(orders) > 1:
            problems.append(f"group {group!r} spans orders {sorted(orders)}; "
                            "members of a group must share one")
    # A track with no terminal slot can never end a word.
    for track in ("n", "v"):
        if not any(s.track == track for s in slots):
            problems.append(f"no slots on track {track!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=ROOT / "data/template.toml")
    ap.add_argument("--chains", action="store_true",
                    help="enumerate the legal slot sequences, one per line")
    args = ap.parse_args()

    tpl = load(args.template)
    print(f"{len(tpl.slots)} slots, {len(tpl.by_morpheme):,} distinct morphemes, "
          f"template valid\n")
    print(f"{'slot':<13}{'track':<7}{'order':>6}  {'group':<11}"
          f"{'requires':<11}{'restart':<8}{'forms':>6}  name")
    for slot in sorted(tpl.slots, key=lambda s: (s.track, s.order, s.id)):
        print(f"{slot.id:<13}{slot.track:<7}{slot.order:>6}  "
              f"{slot.group or '-':<11}{','.join(slot.requires) or '-':<11}"
              f"{slot.restart or '-':<8}{len(slot.morphemes):>6}  {slot.name}")

    shared = {m: [s.id for s in v] for m, v in tpl.by_morpheme.items() if len(v) > 1}
    print(f"\n{len(shared)} morphemes are claimed by more than one slot — the "
          "homographs\nthat make a per-rule condition unable to tell them apart:")
    for morpheme, ids in sorted(shared.items())[:14]:
        print(f"  {morpheme:<8} {', '.join(ids)}")

    if args.chains:
        print("\nlegal slot sequences:")
        for line in enumerate_chains(tpl):
            print(f"  {line}")
    return 0


def enumerate_chains(tpl: Template, depth: int = 4) -> list[str]:
    """Every slot sequence the template permits, to `depth` slots."""
    out = []

    def walk(prefix: list[Slot], track: str):
        if prefix:
            out.append(" → ".join(s.id for s in prefix))
        if len(prefix) >= depth:
            return
        for slot in tpl.slots:
            if prefix and not tpl.may_follow(prefix[-1], slot):
                continue
            if not prefix and slot.track != track:
                continue
            walk(prefix + [slot], track)

    for track in ("n", "v"):
        walk([], track)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
