# Project Working Instructions

- Preserve the existing rice/non-rice gate and segmentation behavior unless a regression test proves a change is necessary.
- Never present synthetic/demo metrics as real-data validation. Mark unsupported predictions `Experimental`, `Proxy`, or `Undetermined` with the reason.
- Do not assert current KMS limits or official lot compliance without a verified authoritative source and equivalent measurement basis.
- Before ending any coding task, append a dated entry to [`progress/WORK_LOG.md`](progress/WORK_LOG.md) with the task summary, files/components changed, tests/builds run and results, blockers, and next steps. Keep prior entries; do not overwrite the log.
- If the task is incomplete, record it as in progress and say what remains. Do not claim completion beyond the evidence.
