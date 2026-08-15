# Parked

Things that were built, measured, and did not earn a place in the pipeline.
Kept because the measurement is worth more than the code, and because a
negative result nobody wrote down gets tried again.

## `triage.py`, `optimise_triage.py` — asking a model instead of the corpus

Three questions in this project look like judgements a language model should be
better at than a table of suffixes. It was better at none of them.

| question | model | corpus |
|---|---|---|
| is this string a word? | 75% precise on a half-junk pool | 24 words gained per leak |
| which harmony column? | not tried | categorical: `оқиды` 882 books, `оқыйды` 7 |
| which track, n or v? | 60% | 96% |

On the near-miss pool the model's `word` verdict removed 164 real words to
avoid 24 leaks — the ratio of the pool it was filtering, which is what sampling
looks like. Its mistakes were the ones that matter: it approved `аал`, `даса`
and `кұда`, the last being `құда` with a Russian `к`, which is exactly what
`build_lexicon.py` refuses at admission because admitting it makes the
misspelling unflaggable.

The reason is not that the model is bad. It is that inflection is evidence and
recognition is a guess, and this corpus has 84.2M tokens of evidence. A
question of the form *what kind of thing is this* can nearly always be rewritten
as *what does the corpus write after it*, and then it is not a judgement at all.

What survived: the calibration habit. Every one of those numbers exists because
the tool scored itself against answers already known — apertium's hand-filed
parts of speech, misspellings attested in none of the 3,860 editions — before
being believed. That is worth keeping whatever is being asked.

Where a model might still earn its place, if this is picked up again:

- the 40,195 entries the books do not inflect either way, where there is no
  corpus evidence by construction and it is the model or nothing
- lemma grouping: the corpus can say `қайрыл` inflects, not that it belongs
  under `қайру`. `data/triaged.tsv` holds proposals for 4,274 stems.
- reading the junk out of the existing lexicon, where being wrong is cheap and
  `tools/regress.py` catches it

To run either tool again: `uv venv .venv && uv pip install --python .venv
pydantic-ai-slim[openai] gepa`, an `OPENROUTER_API_KEY` in `.env`, and
`--calibrate N` before anything else.
