# AGENTS: Time-Series Track

## Scope

Work in this folder when changing CSV time-series datasets, feature baselines, model wrappers, training runners, metrics, or evidence exports.

Primary paths:

```text
ml/timeseries/train.py
ml/timeseries/src/
ml/timeseries/scripts/
ml/timeseries/tests/
ml/timeseries/requirements.txt
```

## Rules

- Use `data/manifest.csv` as the source of truth.
- Treat CSV signals as `(20, 7680)` unless a script explicitly transposes them.
- Prefer `label_id` from the manifest as the supervised target.
- Never use path strings, filenames, sample IDs, label names, or defect details as features.
- Keep `max_discharge_value` out of the default metadata whitelist.
- Run one concrete model per command.
- Use official libraries or wrappers for known paper models when available.
- If an optional dependency is missing, raise or skip with a clear reason; do not silently substitute a different model.

## Imports

Use the package root:

```python
from ml.timeseries.src.data.loader import load_manifest
from ml.timeseries.src.models.registry import create_model
```

Do not use old imports:

```python
from ml.src...
from ml.scripts...
```

## Validation

Before training:

```powershell
python ml/timeseries/scripts/validate_dataset.py --fail-on-invalid
```

For local tests:

```powershell
pytest ml/timeseries/tests
```

Optional-dependency model tests may skip when official packages or cloned repositories are unavailable.

## Output

Write experiment outputs under:

```text
results/
results/timeseries/
```

Do not write generated artifacts into `Train/`.
