# LLM Wiki

The LLM Wiki is the project-memory layer for the composite partial-discharge
diagnosis system.

It records what the system is trying to build, which contracts are stable, what
evidence exists, and which implementation boundaries future agents must respect.

It is not a replacement for PRDs, source code, tests, or raw experiment
artifacts.

## Read Order

1. `INDEX.md`
   - Entry point for current project state and wiki navigation.
2. `../PROJECT_STRUCTURE.md`
   - Folder and code responsibility map.
3. `concepts/current_development_findings.md`
   - Current synthesized interpretation of the project.
4. `concepts/composite_diagnosis_architecture.md`
   - How time-series, VLM, reviewer, trace, and service layers fit together.
5. `../VLM_TRAINING_GUIDE.md`
   - Current VLM training and activation workflow.
6. `source_cards/`
   - Stable cards for tools, model families, SDKs, and implementation sources.
7. `experiment_notes/`
   - Run-level notes for actual model experiments.
8. `LOG.md`
   - Chronological wiki update record.

## Layer Responsibilities

| Path | Responsibility |
|---|---|
| `raw_sources/` | Source metadata, citations, and short excerpts |
| `source_cards/` | One-card summaries of papers, tools, model families, SDKs |
| `concepts/` | Synthesized interpretation across docs, code, and experiments |
| `experiment_notes/` | Model-run notes, commands, metrics, limitations |
| `experiment_notes/artifacts/` | Sanitized snapshots copied from ignored results |
| `templates/` | Reusable source-card and experiment-note templates |
| `INDEX.md` | Human and LLM entry point |
| `LOG.md` | Chronological change record |

## Boundaries

Numeric model comparisons must link to a specific experiment note or sanitized
artifact. Do not compare results across different splits, sample sizes, model
contracts, or prompt schemas as if they were a single leaderboard.
