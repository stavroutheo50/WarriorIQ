# Warrior IQ repository rules

Warrior IQ is an existing production-oriented combat-sports analysis product. Its public domain is `https://WarriorIQ.eu`. Preserve working behavior and backward compatibility.

## Working rules

- Inspect the relevant implementation, tests, and current Git diff before editing.
- Make the smallest focused change that solves the verified problem. Do not modify unrelated files.
- Do not redesign the site, replace the frontend, migrate frameworks, rename routes, or reorganize the repository unless explicitly requested.
- Reuse existing components and follow the surrounding style. Keep code understandable and human-maintainable.
- Never fabricate fight events, statistics, confidence, progress, or scorecard evidence.
- Keep fight-analysis logic separate from UI presentation where practical, while preserving current interfaces.
- Never hard-code or log secrets. Use environment variables and keep private `.env` files untracked.
- Do not silently change public APIs or database schemas. Make required transitions explicit and backward-compatible.
- Avoid new dependencies unless the existing standard library and project packages cannot solve the problem cleanly.
- Add or update focused tests whenever scoring, statistics, deterministic analysis, identity/state ownership, or aggregation changes.
- Verify affected routes and important user flows after changes. Never claim a check was run unless it actually passed.
- Preserve user-owned changes. Do not deploy, push, rewrite Git history, delete data, or alter live services without explicit authorization.

## Architecture map

- `app/main.py`: FastAPI routes, authentication boundaries, upload/selection flow, analysis orchestration, and HTML responses.
- `app/state.py`: per-job runtime state, durable session files, worker claims, leases, and stale-run protection.
- `app/templates/` and `app/static/`: existing server-rendered interface and interactions; preserve its visual identity.
- `worker.py`: optional external analysis worker. Local development uses the in-process worker by default.
- `core/analyzer.py`: analysis pipeline coordinator; it connects video, tracking, identity, actions, contact, metrics, scoring, and reports.
- `core/video.py`, `pose_tracker.py`, `identity.py`, `sam_recovery.py`: ingestion, pose/tracking, fighter attribution, and recovery.
- `core/action.py`, `contact.py`, `defense.py`, `fight_stats.py`, `scoring.py`: deterministic action/outcome, statistics, combinations, rounds, and estimated scoring.
- `core/report.py` and `coaching.py`: evidence-gated report and training output. They must consume real analysis results only.
- `core/db.py`: SQLite persistence and explicit compatibility migrations; do not change schema casually.
- `core/config.py`: environment-backed configuration and runtime data roots.
- `tests/`: focused unit and application regression tests. `tools/verify_project.py` is the broad local verification entry point.

Correct fighter identity and evidence integrity take priority over speed. Optimize only measured waste such as duplicate decoding, model initialization, inference, copying, or I/O; do not lower analysis quality to improve timing.
