# Draft: PD Model Evaluation Plan

## Requirements (confirmed)
- User asked in Korean to evaluate the current dataset shape, recommend time-series or other models before trying VLM, and evaluate the current model experiment plan.

## Technical Decisions
- Treat this as an architecture/research evaluation plan over an existing ML scaffold.
- Keep work read-only except `.omo/drafts` and `.omo/plans` artifacts.
- Prioritize leakage-safe non-VLM baselines before VLM.

## Research Findings
- CodeGraph: project is Python-only with 49 indexed files.
- CodeGraph: dataset loader normalizes each CSV sample and returns channel-first tensors by default.
- CodeGraph: current scaffold already includes MiniRocket, MultiRocket, HYDRA, sktime classifier, TS2Vec, feature baseline, transformer, and foundation-model scripts or wrappers.

## Open Questions
- Whether the user wants a written recommendation only or an executable evaluation plan; defaulting to an executable plan because current mode is planning.

## Scope Boundaries
- INCLUDE: dataset-shape assessment, model priority order, experiment-plan critique, verification strategy.
- EXCLUDE: source-code implementation, running model training, changing data files.
