# PRD: Time-Series Partial-Discharge Model Track

## 1. Purpose

This track develops CSV-based partial-discharge classifiers.

It owns:

```text
data/manifest.csv
-> timeseries_path CSV
-> tensor / feature extraction
-> 5-class prediction
-> probabilities, confidence, and evidence export
```

The output is machine diagnostic evidence that can be consumed by `ml/vlm` and the service workflow.

## 2. Input

Source:

```text
data/manifest.csv
timeseries_path
```

CSV shape:

```text
(20, 7680)
```

Interpretation:

- `20` is a pseudo-channel or measurement-segment dimension.
- `7680` is the time axis.
- Do not assume the 20 rows are independent physical sensors unless dataset documentation proves it.

Allowed metadata:

- safe numeric/equipment metadata only when explicitly enabled
- no path, filename, label text, sample ID, or defect detail columns

## 3. Output

Required model output:

```text
label_id
prob_0..prob_4
confidence
model_name
split
```

Recommended evidence output:

```text
rms
std
abs_p99
pulse_rate
spectral_energy
phase-bin pulse features
numeric PRPD histogram features
```

Export context for downstream VLM:

```text
results/timeseries/evidence_context.csv
```

Service adapter contract:

```text
ml/timeseries/src/service_adapter.py
load_adapter(context) -> backend
backend.predict_csv(TimeSeriesToolInput) -> dict
```

The returned dict must contain `label_id`, `confidence`, `probabilities`, and `features`. The service normalizes this into `TimeSeriesResult`.

## 4. Models

Core GPU models:

```text
GRU
InceptionTime
PatchTST
TimesNet
MOMENT
```

Extended models:

```text
TCN
ResNet1D
ModernTCN
iTransformer
TimeMixer
UniTS
GPT4TS / One-Fits-All
TS2Vec
```

CPU/classical baselines:

```text
feature_logistic
feature_svm
feature_random_forest
feature_tabpfn
MiniROCKET
MultiROCKET
HYDRA
sktime summary / catch22
```

## 5. Training Order

1. Validate dataset.
2. Run feature baseline smoke.
3. Train GRU smoke.
4. Train one core neural model on a fixed split.
5. Add official-library models only after smoke tests pass.
6. Export validation predictions for VLM and service evaluation.

## 6. Evaluation

Required metrics:

```text
accuracy
macro_f1
weighted_f1
balanced_accuracy
per_class_precision
per_class_recall
per_class_f1
confusion_matrix
actual_PD_predicted_as_normal_count
```

Important confusion checks:

- normal vs noise
- noise vs actual PD
- surface vs corona
- void vs other PD

## 7. CLI

Main neural model runner:

```powershell
python ml/timeseries/train.py --model gru --sample-size 100
```

Dedicated CPU/classical runners:

```powershell
python ml/timeseries/scripts/run_feature_baseline.py --model logistic
python ml/timeseries/scripts/run_minirocket.py
python ml/timeseries/scripts/run_multirocket.py
python ml/timeseries/scripts/run_sktime_classifier.py --model catch22
```

Dataset validation:

```powershell
python ml/timeseries/scripts/validate_dataset.py --fail-on-invalid
```

## 8. Non-Goals

- Forecasting future signal values
- Using PRPD PNG images directly
- VLM prompt generation
- Vision encoder LoRA
- Label-leaking metadata experiments as default results
