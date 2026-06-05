# PRD: Partial-Discharge Time-Series and Small VLM Diagnosis Project

## 1. Purpose

This project uses the AI-Hub industrial partial-discharge dataset to build two model tracks:

1. A 5-class partial-discharge classifier that uses only CSV time-series data.
2. A small VLM diagnosis model that combines PRPD images, metadata, and time-series summaries.

This is not an image-only vision classification project. ResNet/EfficientNet-style PRPD image classifiers are excluded from the core track.

## 2. Data Format

Local training data lives under `Train/`.

Current `Train/manifest.csv` summary:

- Total samples: `30,010`
- Label distribution: `6,002` samples for each label `0` through `4`
- Each sample connects three files:
  - `timeseries_path`: partial-discharge CSV time-series
  - `image_path`: PRPD PNG image
  - `json_path`: label and metadata JSON
- CSV shape: `(20, 7680)`
  - `20`: pseudo-channel or measurement-segment dimension
  - `7680`: time axis

Label mapping:

| ID | Class |
| --- | --- |
| 0 | normal |
| 1 | noise |
| 2 | surface_discharge |
| 3 | corona_discharge |
| 4 | void_discharge |

Experiment code must use `label_id` from `manifest.csv`. When regenerating a manifest, derive labels from JSON `label.PD_type`.

## 3. Time-Series Track

The time-series task is classification, not forecasting.

Default input:

```text
CSV -> tensor shape (20, 7680), or transposed to (7680, 20)
```

Core models:

```text
GRU
InceptionTime
PatchTST
TimesNet
MOMENT
```

Extended GPU models:

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

CPU-only optional baselines:

```text
MiniROCKET
MultiROCKET
HYDRA
feature_logistic
feature_svm
feature_random_forest
feature_tabpfn
```

Models that may have especially long training time:

```text
TimeMixer
UniTS
GPT4TS
TS2Vec
```

Run these models on a small smoke subset before using all `30,010` samples.

## 4. Feature Baseline Track

The feature baseline is not a raw time-series model. It is a tabular classifier that consumes features extracted from each CSV.

Pipeline:

```text
CSV signal
-> amplitude / pulse / cycle / phase-bin / FFT / numeric PRPD histogram features
-> Logistic / Linear SVM / RandomForest / TabPFN
-> 5-class prediction
```

One CSV becomes one feature vector. The default is `--feature-set small`.

```text
Raw input per sample: (20, 7680) CSV
Converted input per sample: 64 tabular feature columns

Full Train shape:
X.shape = (30010, 64)
y.shape = (30010,)
target = label_id
```

Feature-set sizes:

| Feature set | CSV feature columns | With metadata | Purpose |
| --- | ---: | ---: | --- |
| `small` | 64 | 74 | Fast baseline using global/stat, FFT, amplitude histogram, pulse, cycle, and half-cycle features |
| `medium` | 128 | 138 | Balanced baseline with phase-bin count/max features |
| `full` | 182 | 192 | Includes a compact 96-bin numeric PRPD histogram |

If `--include-metadata` is enabled, add only the safe numeric metadata whitelist. Do not use file paths, file names, sample IDs, label names, or defect details as features.

## 5. VLM Track

Do not place full raw CSV rows in the VLM prompt.

Recommended VLM input:

```text
PRPD PNG image
+ JSON metadata text
+ time-series model prediction / confidence
+ time-series feature summary
```

Primary candidate:

```text
Qwen3-VL-2B-Instruct
```

Fallback or comparison candidate:

```text
Qwen2.5-VL-3B-Instruct
```

Training strategy:

```text
QLoRA SFT
Freeze the vision encoder at first
Apply LoRA primarily to LLM/projector layers
Prefer structured JSON diagnosis output over free-form natural language
```

Example target VLM output:

```json
{
  "label_id": 3,
  "label_name": "corona_discharge",
  "risk_level": "caution",
  "reason": "The PRPD pattern, metadata, and time-series evidence are consistent with corona discharge."
}
```

## 6. Initial EDA

Run initial EDA once on the current Train data and manifest. Do not repeat it before every model run unless data extraction, `manifest.csv`, label mapping, feature design, or split policy changes.

Default command:

```powershell
python ml/scripts/run_eda.py
```

The default summarizes the full 30,010-row manifest and reads a class-balanced `sample-size=500` subset of actual CSV signals for faster visualization.

Default output location:

```text
results/eda/
```

Main outputs:

| File | Purpose |
| --- | --- |
| `eda_summary.json` | Row count, label distribution, leakage-risk columns, and missing paths |
| `label_distribution.csv/png` | Class balance check |
| `metadata_distributions.png` | Insulator, equipment, and sensor distributions |
| `signal_summary_sample.csv` | RMS, p99, max_abs, pulse_rate, and related per-sample statistics |
| `signal_stats_by_class.png` | Class-wise signal-statistics boxplots |
| `phase_pulse_distribution.png` | 60Hz phase-bin pulse distribution |
| `sample_waveforms_by_class.png` | Example raw waveforms by class |
| `class_mean_abs_waveform.png` | Mean absolute waveform by class |

Run full signal-level EDA over all CSV files only when explicitly needed:

```powershell
python ml/scripts/run_eda.py --full-signal-eda
```

Initial EDA must check:

- Whether labels `0` through `4` are balanced
- Whether leakage-prone columns such as paths, file names, and label text are excluded from features
- Whether every CSV shape is `(20, 7680)`
- How RMS, p99, max_abs, and pulse_rate differ by class
- Whether 60Hz phase-bin pulse distributions differ by class
- Whether specific equipment, sensors, or insulators are concentrated in specific labels

## 7. Execution Rules

`train.py` must train exactly one neural model per run.

Allowed:

```powershell
python train.py --model gru --sample-size 100
python train.py --model moderntcn --sample-size 100
```

CPU-only baselines are not trained directly by `train.py`. The `cpu_only` entries in `train.py --list-models` are informational. Run each model through its dedicated runner:

```powershell
python ml/scripts/run_feature_baseline.py --model logistic
python ml/scripts/run_minirocket.py
python ml/scripts/run_multirocket.py
python ml/scripts/run_sktime_classifier.py --model catch22
```

Forbidden:

```powershell
python train.py --model core
python train.py --model extended
python train.py --model gru,patchtst
```

GPU neural models use `.venv`.

CPU-only baselines may use `.venv` or `.venv314t`.

- `.venv`: stable default environment
- `.venv314t`: free-threaded Python 3.14t environment for CPU-only smoke and multi-thread experiments

## 8. Success Criteria

Minimum success:

- Manifest-based dataset loading works.
- At least one Core time-series model trains and evaluates.
- Leakage-safe feature baseline works.
- Metrics include accuracy, macro F1, per-class F1, and confusion matrix.
- VLM strategy and data-conversion plan are documented.

Strong portfolio success:

- Core time-series models are compared.
- MiniROCKET/MultiROCKET and feature baselines are compared.
- At least one pretrained or foundation time-series model is fine-tuned or probed.
- A VLM instruction dataset is built from PRPD image, metadata, and time-series summaries.
- A small VLM generates structured JSON diagnosis output.

## 9. Detailed Documents

- `docs/PRD.md`: detailed project PRD
- `docs/DATASET_EXPLAIN.md`: integrated data structure and manifest notes
- `docs/TIMESERIES_MODELS.md`: time-series model candidates and experiment order
- `docs/MODEL_IMPLEMENTATION_SOURCES.md`: official implementation and wrapper policy
- `docs/VLM_STRATEGY.md`: VLM model and fine-tuning strategy
- `AGENT.md`: coding and execution rules
