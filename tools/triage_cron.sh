#!/usr/bin/env bash
# One slice of the triage backlog, sized to leave the daily allowance intact.
#
# The free tier gives about a thousand requests a day. The timer fires every
# three hours, so a run may take an eighth of that and no more — 120 requests,
# 960 a day at the worst, and the pool is done in a day of them. It does not
# need to be greedy: it takes its slice, writes what it learned to the
# checkpoint, and stops.
# The next run resumes from there, and anything left unanswered by the deferred
# passes is asked again rather than recorded as a refusal.
set -uo pipefail
cd /var/home/idyn/projects/kazspell || exit 1

LOG=data/triage-cron.log
exec >>"$LOG" 2>&1
echo "=== $(date -Is) starting ==="

timeout 5400 .venv/bin/python -u tools/triage.py \
    --candidates data/near_misses.tsv \
    --batch 12 --concurrency 6 --rpm 20 --cap "${CAP:-300}" --retries 2
status=$?

remaining=$(.venv/bin/python - <<'PY'
import json, pathlib
seen = set()
p = pathlib.Path("data/triage.jsonl")
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seen.add(json.loads(line)["word"])
todo = [l.split("\t")[0] for l in
        pathlib.Path("data/near_misses.tsv").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")]
print(sum(1 for w in todo if w not in seen))
PY
)
echo "=== $(date -Is) exit $status, $remaining candidates still to do ==="
