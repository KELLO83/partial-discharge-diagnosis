# Draft: Partial Discharge Model Experiment Plan

## Requirements (confirmed)
- Evaluate the current partial-discharge dataset shape.
- Recommend time-series or non-VLM models to try before VLM.
- Evaluate the current model experiment plan.
- Respond in Korean.

## Technical Decisions
- Treat the first track as supervised 5-class time-series classification, not forecasting.
- Use one CSV file as one sample; do not split the 20 CSV rows across train/valid/test.
- Prioritize leakage-safe fixed split creation before model ranking.
- Run feature/classical time-series baselines before heavy deep/foundation models.
- Treat VLM as a later reporting/multimodal fusion stage, not the first classifier.

## Research Findings
- `Train/manifest.csv` has 30,010 rows, 5 balanced labels of 6,002 each, and all rows currently marked `split=train`.
- Local validation report shows all 30,010 rows valid, no invalid shapes, and leakage-risk columns present in the manifest.
- CSV shape is enforced as `(20, 7680)` in `ml/src/data/loader.py`; model input is `[B, 20, 7680]` or `[B, 7680, 20]` depending on model wrapper.
- Current registry includes GRU, TCN, InceptionTime, ResNet1D, MiniROCKET, ModernTCN, PatchTST, iTransformer, TimesNet, TimeMixer, MOMENT, UniTS, GPT4TS, and TS2Vec.
- Current docs place MiniROCKET/MultiROCKET/HYDRA and feature baselines as optional CPU-only extended candidates, but these should move earlier as calibration baselines.

## Open Questions
- Whether the user wants a saved execution plan file or only strategic guidance.
- Whether evaluation must prioritize Kaggle/portfolio speed, research rigor, or final deployment realism.

## Scope Boundaries
- INCLUDE: dataset shape assessment, leakage/split risks, model priority, experiment wave plan, VLM timing.
- EXCLUDE: code edits, running long training jobs, changing raw data, implementing missing model wrappers.
