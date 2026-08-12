"""Check the suffix template against apertium-kaz's grammar, morpheme by morpheme.

`data/template.toml` is a reconstruction of Kazakh grammar, and the last time a
reconstruction like it was trusted, three separate omissions — the қалау рай
`-ғы`, the imperative `-ыңыз`, the denominal `-ла` — each silently deleted real
words, and only a paradigm grid caught them. The template now sits upstream of
the lexicon, the engine and every number this project reports, so an error in it
propagates everywhere and is visible nowhere.

apertium-kaz is a hand-written grammar by linguists. Its suffix entries are
`%<tag%>:%>surface`, where the surface is spelt in archiphonemes that its twol
rules resolve — `%<pl%>:%>%{L%}%{A%}р` is `-лар/-лер/-дар/-дер/-тар/-тер` in one
line. Expanding those gives an inventory to hold ours against, decided by
somebody else, for their own reasons, years ago.

What the diff means is asymmetric. A form apertium has and we do not is a gap in
our template and probably a bug. A form we have and apertium does not is not
automatically wrong — its lexicon has known holes and covers 95.5% of tokens —
but it is a claim that now has to be defended rather than assumed.

    python tools/check_template.py
"""

from __future__ import annotations

import argparse
import collections
import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from template import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The archiphonemes, as apertium-kaz's own header documents them. Expanding to
# every realisation over-generates on purpose: this is an inventory to compare
# against, not a generator, and twol decides which one surfaces where.
# Harmony-neutral archiphonemes: the choice is voicing or assimilation, and
# every realisation is possible on some stem.
NEUTRAL = {
    "L": "лдт", "N": "ндт", "M": "мбп", "D": "дт",
    "S": ["с", ""], "n": ["н", ""], "l": "лн",
    "а": "а", "э": "е", "ә": "ә", "й": "й", "л": "л", "н": "н",
    "з": "з", "т": "т", "с": "с", "і": "і", "д": "д",
}

# Harmony-sensitive ones. Expanding these independently is what produced
# `бақшы` and `ганша` in the first run — strings no Kazakh word contains,
# because one word does not carry both harmonies. They are resolved together,
# back or front, once per form.
HARMONIC = {
    "A": ("а", "е"), "I": ("ы", "і"), "K": ("қ", "к"),
    "G": ("қғ", "кг"), "E": ("ай", "ей"), "y": ("ы", "і"),
}

# apertium's tag against the slot in our template that should hold the suffix.
TAG_SLOT = {
    "pl": "koptik",
    "gen": "septik", "dat": "septik", "acc": "septik", "abl": "septik",
    "loc": "septik", "ins": "septik",
    "px1sg": "tauelik", "px2sg": "tauelik", "px3sp": "tauelik",
    "px1pl": "tauelik", "px2pl": "tauelik",
    "neg": "bolymsyz",
    "pass": "etis_pass", "refl": "etis_refl", "coop": "etis_recip",
    "caus": "etis_caus",
    "ifi": "shaq", "aor": "shaq", "past": "shaq", "pres": "shaq",
    "fut": "shaq", "fut_plan": "shaq",
    "imp": "rai", "opt": "rai", "gna_cond": "rai",
    "ger": "tuiyq", "ger_impf": "tuiyq", "ger_past": "tuiyq", "ger_perf": "tuiyq",
    "prc_perf": "esimshe", "prc_impf": "esimshe", "prc_fplan": "esimshe",
    "prc_vol": "esimshe", "gpr_past": "esimshe", "gpr_impf": "esimshe",
    "gpr_fut": "esimshe", "gpr_pot": "esimshe",
    "gna_perf": "kosemshe", "gna_impf": "kosemshe", "gna_cont": "kosemshe",
    "gna_until": "kosemshe", "gna_after": "kosemshe",
}

ENTRY = re.compile(r"^(%<[^ :]*%>):%>(\S+)")


def tokenise(surface: str) -> list[str] | None:
    """The spelling as a list of literals and `{X}` archiphoneme names."""
    out, i = [], 0
    while i < len(surface):
        if surface[i] == "%" and i + 1 < len(surface):
            if surface[i + 1] == "{":
                end = surface.find("%}", i)
                if end == -1:
                    return None
                out.append("{" + surface[i + 2:end].replace("%", "") + "}")
                i = end + 2
                continue
            out.append(surface[i + 1])          # %-escaped literal
            i += 2
            continue
        out.append(surface[i])
        i += 1
    return out


def expand(surface: str) -> set[str]:
    """Every letter string this archiphoneme spelling can stand for.

    Only the first morpheme is taken. `%>` is lexc's boundary marker, so an
    entry like `бақ%>шы` is two morphemes and the template holds them in two
    slots; joining them would compare a chain against a morpheme and report a
    gap that is really a difference of granularity.
    """
    tokens = tokenise(surface)
    if tokens is None:
        return set()
    if tokens and tokens[0] == ">":
        tokens = tokens[1:]
    if ">" in tokens:
        tokens = tokens[:tokens.index(">")]
    tokens = [t for t in tokens if t not in ("#", "%", "")]
    if not tokens:
        return set()

    forms = set()
    for harmony in (0, 1):
        parts: list[list[str]] = []
        for token in tokens:
            if token.startswith("{"):
                name = token[1:-1]
                if name in HARMONIC:
                    parts.append(list(HARMONIC[name][harmony]))
                elif name in NEUTRAL:
                    parts.append(list(NEUTRAL[name]))
                else:
                    return set()
            else:
                parts.append([token])
        if sum(len(p) for p in parts) > 48:
            return set()
        forms |= {"".join(c) for c in itertools.product(*parts)}
    return {f for f in forms if f and all(ch.isalpha() for ch in f)}


def apertium_suffixes(lexc: Path) -> dict[str, set[str]]:
    """slot id -> the surface forms apertium's grammar gives it."""
    out: dict[str, set[str]] = collections.defaultdict(set)
    for line in lexc.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = ENTRY.match(line)
        if not m:
            continue
        tags, surface = m.groups()
        heads = re.findall(r"%<([a-z0-9_]+)%>", tags)
        slot = next((TAG_SLOT[h] for h in heads if h in TAG_SLOT), None)
        if not slot:
            continue
        for form in expand(surface):
            if form:
                out[slot].add(form)
    return out


def composes(tpl, form: str, track: str) -> list[str] | None:
    """The slot walk that builds `form` from two or more morphemes, if any.

    apertium-kaz spells some combinations as one lexc entry — `%<pl%>%<px3sp%>`
    is `лары`, локатив plus attributive is `дағы`. Those are not morphemes we
    are missing; they are chains we compose, and reporting them as gaps would
    mean padding the template with every combination the language allows.
    """
    def walk(rest: str, track: str, prev, path: list[str]):
        if not rest:
            return path if len(path) > 1 else None
        for cut in range(len(rest), 0, -1):
            for slot in tpl.by_morpheme.get(rest[:cut], ()):
                if slot.requires:
                    continue
                if prev is None:
                    if slot.track != track:
                        continue
                elif not tpl.may_follow(prev, slot):
                    continue
                nxt = slot.restart or slot.track
                got = walk(rest[cut:], nxt, slot, path + [slot.id])
                if got:
                    return got
        return None
    return walk(form, track, None, [])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexc", type=Path,
                    default=ROOT.parent / "apertium-kaz/apertium-kaz.kaz.lexc")
    ap.add_argument("-o", "--output", type=Path,
                    default=ROOT / "data/template_gaps.tsv")
    args = ap.parse_args()

    tpl = load()
    theirs = apertium_suffixes(args.lexc)
    ours = {s.id: set(s.morphemes) for s in tpl.slots}
    # Everything the template knows, anywhere: a form filed under a different
    # slot is a disagreement about analysis, not a missing morpheme, and the
    # two are worth telling apart.
    everywhere = {m for v in ours.values() for m in v}

    rows, missing_total, elsewhere_total, composed_total = [], 0, 0, 0
    print(f"{'slot':<13}{'ours':>6}{'theirs':>8}{'shared':>8}"
          f"{'missing':>9}{'composed':>10}{'elsewhere':>11}")
    for slot in sorted(theirs):
        mine, their = ours.get(slot, set()), theirs[slot]
        raw_missing = their - everywhere
        missing, composed = set(), set()
        for form in raw_missing:
            track = "n" if slot in ("koptik", "tauelik", "septik", "jiktik_n",
                                    "sozjasam", "attr", "tauelik_of") else "v"
            (composed if composes(tpl, form, track) else missing).add(form)
        elsewhere = (their - mine) - raw_missing
        missing_total += len(missing)
        elsewhere_total += len(elsewhere)
        composed_total += len(composed)
        print(f"{slot:<13}{len(mine):>6}{len(their):>8}{len(mine & their):>8}"
              f"{len(missing):>9}{len(composed):>10}{len(elsewhere):>11}")
        for form in sorted(missing):
            rows.append((slot, form, "missing"))
        for form in sorted(elsewhere):
            held = ",".join(sorted(s for s, v in ours.items() if form in v))
            rows.append((slot, form, f"filed-under:{held}"))

    args.output.write_text(
        "# slot\tform\tstatus — apertium-kaz has this form for this slot;\n"
        "# `missing` means the template does not have it at all,\n"
        "# `filed-under` means we hold it in a different slot\n"
        + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")

    print(f"\n{missing_total} forms apertium has that the template cannot build")
    print(f"{composed_total} it spells as one entry and we compose from slots")
    print(f"{elsewhere_total} it files under a different slot than we do")
    if missing_total:
        print("\nmissing, by slot:")
        for slot in sorted({r[0] for r in rows if r[2] == "missing"}):
            forms = [r[1] for r in rows if r[0] == slot and r[2] == "missing"]
            print(f"  {slot:<13}{' '.join(sorted(forms)[:18])}")
    print(f"\n→ {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
