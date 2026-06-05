# AGENT: AI Coding Agent Guidelines

## 1. Purpose

This document defines development rules for AI coding agents working on a partial-discharge diagnosis project using the AI-Hub industrial electrical fire prevention dataset.

The primary goal is to build an experiment pipeline for 5-class partial-discharge classification from CSV time-series data. The secondary goal is to build a small multimodal VLM diagnosis model that combines PRPD images, JSON metadata, and time-series summaries to generate either natural-language diagnostics or structured JSON output.

Before writing code, agents must review these documents in order:

1. `PRD.md`: root-level project goals, experiment scope, and execution rules
2. `docs/PRD.md`: detailed goals, scope, experiment stages, and success criteria
3. `docs/DATASET_EXPLAIN.md`: integrated data structure, CSV/PNG/JSON layout, labels, and manifest rules
4. `docs/TIMESERIES_MODELS.md`: candidate time-series models and experiment order
5. `docs/VLM_STRATEGY.md`: VLM candidate models, input/output design, and QLoRA strategy
6. `AGENT.md`: code style, runtime environment, resource usage, and validation rules

## Dataset Structure

See `docs/DATASET_EXPLAIN.md`.

## Project Scope

Current implementation priorities:

1. Inspect the `Train/` data structure and build manifest-based data mapping.
2. Run initial EDA and check data quality, labels, and leakage risks.
3. Implement CSV time-series datasets and DataLoaders.
4. Train and evaluate a GRU baseline.
5. Compare non-Transformer, Transformer/SOTA, and foundation/pretrained time-series models.
6. Export summary features or diagnostic context for VLM inputs based on time-series experiments.
7. Build a small Qwen-family VLM diagnosis model that combines PRPD images, metadata, and time-series summaries.

Forecasting is out of scope. The time-series task is not future-value prediction. It is 5-class classification: `normal`, `noise`, `surface_discharge`, `corona_discharge`, and `void_discharge`.

## Data Interpretation Rules

Use `Train/manifest.csv` as the source of truth for connecting CSV, PNG, and JSON files.

Current local `Train/manifest.csv` observations:

- Total samples: `30,010`.
- Label distribution is balanced: `6,002` samples for each label `0` through `4`.
- Each sample connects `timeseries_path`, `image_path`, and `json_path`.
- CSV time-series shape is `(20, 7680)`.
- Main manifest columns include `sample_id`, `split`, `json_path`, `image_path`, `timeseries_path`, `label_id`, `label_name`, equipment/environment metadata, and `max_discharge_value`.

Rules:

- Read source CSV files as headerless integer time-series arrays.
- Treat the confirmed CSV shape as `(20, 7680)`.
- Do not assume `20` means physical sensor channels. Treat it as a measurement-segment or pseudo-channel dimension.
- Treat `7680` as the time axis.
- For RNN/Transformer inputs, transpose to `(time, pseudo_channel) = (7680, 20)` when needed.
- For Conv/TCN/Patch-style inputs, use `(pseudo_channel, time) = (20, 7680)` as the default candidate format.
- Use JSON `PD_type` as the target label.
- Prefer manifest `label_id` in code. When regenerating a manifest, derive labels from JSON `label.PD_type`.
- Do not include label-leaking fields in VLM prompts, including `PD_type`, label-bearing paths, or label-bearing file names.
- Do not use string columns that can contain class or target information as feature-baseline inputs, including `sample_id`, `timeseries_path`, `image_path`, `json_path`, `label_name`, and `defect_details`.
- JSON metadata may be used as LLM-side text context during the VLM stage.

## Directory Boundaries

Separate code by purpose:

```text
scripts/
  Data checks, manifest generation, and reusable utility scripts

ml/
  Time-series datasets, preprocessing, model wrappers, training, evaluation, and leaderboard code

vlm/
  VLM data conversion, prompt templates, LoRA/SFT training, and inference

results/
  Experiment result CSV/JSON files, metrics, model artifacts, and log summaries

docs/
  Project requirements, dataset notes, and model documentation
```

Rules:

- Write time-series training code under `ml/`.
- Write VLM data conversion and training code under `vlm/`.
- Put repeatable data-checking and manifest-generation utilities under `scripts/`.
- Store experiment outputs and model artifacts under `results/`.
- Keep raw data under `Train/`; never overwrite source data.
- For Transformer/foundation models that have pip packages or official open-source implementations, do not reimplement the model body in this repository. Import them through model-specific wrappers.
- Keep wrappers in per-model files such as `gru.py`, `tcn.py`, `patchtst.py`, and `moment.py`.
- Except for GRU, prioritize official repositories, Hugging Face implementations, PyTorch primitives, or validated libraries for well-known paper models.
- If an official dependency is missing, raise an explicit `ImportError` explaining how to install or clone it. Do not silently replace it with an ad hoc fallback.
- Connect official repository clones through environment variables such as `TSLIB_REPO`, `ITRANSFORMER_REPO`, `TIMEMIXER_REPO`, `MODERNTCN_REPO`, `UNITS_REPO`, `ONE_FITS_ALL_REPO`, and `TS2VEC_REPO`.
- Review `docs/VLM_STRATEGY.md` before writing VLM-related code or datasets.
- Implement the VLM as a diagnosis-report generator using PRPD image, JSON metadata, and time-series summary information, not as an image-only classifier.
- Do not place full raw CSV data in VLM prompts. Provide compressed text context such as time-series model prediction, confidence, class probabilities, and statistical features.
- Prefer Qwen3-VL-2B-Instruct for the first VLM candidate. Use Qwen2.5-VL-3B-Instruct as a fallback or comparison candidate.
- Start VLM training with QLoRA SFT. Freeze the vision encoder first and apply LoRA mainly to LLM/projector layers.
- Prefer structured JSON diagnosis output over free-form natural language.

## Model Experiment Principles

Time-series model experiments are grouped as follows:

- Non-Transformer: GRU, TCN, InceptionTime, ResNet1D, ModernTCN
- Transformer / Modern SOTA: PatchTST, iTransformer, TimesNet, TimeMixer
- Foundation / Pretrained: MOMENT, UniTS, GPT4TS / One-Fits-All
- Representation Learning: TS2Vec
- CPU-only optional baseline: MiniROCKET, MultiROCKET, sktime feature-based classifiers, ROCKET, Arsenal, HYDRA, feature baseline, TabPFN

Rules:

- Start with GRU as the first baseline.
- Do not implement every model at once. Complete Core models first, then extend.
- `TimeMixer`, `UniTS`, `GPT4TS`, and `TS2Vec` can be expensive. Run a small smoke subset and an intermediate subset before running all 30k working samples.
- `iTransformer` and `ModernTCN` can also be expensive on long sequences. Start with shorter `seq_len` or `--sample-size`.
- Compare all models on the same split, label mapping, and metrics.
- Required classification metrics include accuracy, macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, and the number of true discharge samples predicted as normal.
- Before training, run `python ml/scripts/validate_dataset.py --fail-on-invalid` to validate manifest paths, labels, CSV shapes, NaN/inf values, and constant signals.
- If a manifest has both `split=train` and `split=valid`, all runners should use those splits. Reuse the same split manifest for model comparisons.
- Do not include `max_discharge_value` in the default metadata whitelist for feature baselines because it may be a leakage proxy. Use it only in a separate ablation.
- Exclude forecasting-only models such as TimesFM, Chronos, and Lag-Llama from the current scope.
- Use pretrained checkpoints for foundation models when possible, either with a downstream classification head or through fine-tuning.
- Record from-scratch and pretrained fine-tuning results separately in the leaderboard.
- For classical TSC models with official `sktime` classifiers, use an `sktime` runner rather than implementing them manually. Do not run `RandomInterval`, `TSFresh`, `FreshPRINCE`, or `Arsenal` without both `--allow-expensive` and a small subset.
- `train.py` is a single-experiment CLI for GPU neural/foundation models. CPU-only baselines such as `MiniROCKET`, `MultiROCKET`, `HYDRA`, `feature_*`, and `sktime_*` must use dedicated `ml/scripts/run_*.py` runners.
- Run only one CPU-only baseline per command. Each runner should have safe smoke defaults.
- Items marked `cpu_only` in `train.py --list-models` are discovery/help entries. If passed to `--model`, the CLI should print the dedicated runner command and exit.
- Run initial EDA once on the current `Train/` data and manifest to check label distribution, metadata distribution, CSV shape, signal statistics, phase-bin pulse distribution, and leakage-risk columns. Repeat EDA only when data structure, manifest generation, label mapping, feature design, or split policy changes.

## Python Runtime Policy

Use the project-root `.venv` as the default environment.

### ML Training Environment

GPU time-series/VLM training and large preprocessing use:

```text
.venv
```

Purpose:

- Standard CPython 3.14 with the GIL enabled
- CUDA-enabled PyTorch GPU training
- Large CSV loading, manifest validation, feature extraction, and DataLoader execution
- CPU multi-thread workloads with pandas, numpy, and scikit-learn

Warnings:

- If a library does not support Python 3.14, run that model in a compatible separate environment and record the reason.
- If package install or wheel loading fails in `.venv`, do not force a workaround. Run the package in an officially supported Python environment and record the environment difference.
- Do not build packages from source just because compatible wheels are unavailable in `.venv`, especially PyTorch, CUDA, or time-series foundation-model packages, unless the user explicitly requests it.
- Record the runtime environment for each model.

CPU-baseline exceptions:

- The main experiments are deep-learning time-series classification, so LightGBM/CatBoost-style tabular baselines are not part of the default scope.
- CPU-based classical time-series baselines such as MiniROCKET, MultiROCKET, HYDRA, feature baseline, TabPFN, shapelets, and sklearn classifiers may still use `.venv` by default.
- If a package does not support Python 3.14, run it in a compatible environment and record the difference.
- CPU fallbacks use internal multi-threading; specify `n_jobs=14` or library-specific thread options when practical.

Neural/Transformer/Foundation exceptions:

- GRU, TCN, InceptionTime, ResNet1D, ModernTCN, PatchTST, iTransformer, TimesNet, TimeMixer, MOMENT, UniTS, and GPT4TS normally use GPU training or GPU inference.
- Install and run these models in `.venv` because PyTorch/CUDA/pretrained-checkpoint compatibility matters.
- Do not force source builds of PyTorch, CUDA extensions, or time-series foundation packages inside `.venv`.
- Treat `ml/requirements.txt` as the dependency list for `.venv`.
- Consider CPU fallback only when GPU is unavailable, and record the environment and reason.
- For models requiring pretrained checkpoints, use Hugging Face tokens, local checkpoint paths, or explicit cache paths rather than browser-login prompts.

### Server/API/Admin Runtime

FastAPI backends, admin APIs, and server code also use:

```text
.venv
```

Purpose:

- Standard CPython 3.14
- FastAPI, SQLAlchemy, Alembic, PostgreSQL, and admin APIs
- Avoid dependency conflicts between ML training and service runtime

Frontend code uses the Node.js package environment under `frontend/`, not a Python virtual environment.

## ML Training Resource Policy

### Progress Display

Long-running ML/AI code must expose progress.

Applies to:

- Large CSV loading and chunk sampling
- Preprocessing, encoding, and split generation
- Epoch/iteration loops
- Batch prediction
- Hyperparameter sweeps
- Foundation model sample evaluation
- Leaderboard generation

Rules:

- Use `tqdm` progress bars when practical.
- Models with callbacks/logging should provide either a progress bar or periodic logs.
- CLI scripts may expose `--no-progress`, but progress should be enabled by default.
- Progress output must not pollute metric/result CSV files.
- Long jobs should report the current stage, processed rows, total rows or chunks, and elapsed time.
- Server/API runtime should expose progress through structured logs or DB job status rather than terminal progress bars.

### Terminal Logging

ML/AI development code must log important execution state at `INFO` level.

Log:

- Data file loading start/end
- Split, input shape, label mapping, and metadata usage
- Train/validation row counts
- Sample size, seed, and per-label row counts
- Model name, runtime environment, Python executable/version
- Important hyperparameters, sequence length, and pseudo-channel count
- GPU usage, CUDA device, and mixed precision
- GPU/CPU worker settings
- Training start/end and elapsed time
- Prediction start/end and elapsed time
- Metric results
- Result CSV/JSON paths
- Model artifact save/load paths when implemented

Rules:

- Prefer the standard `logging` module or a shared project logger over bare `print()`.
- Default log level is `INFO`.
- Use `warning` for suspicious situations and `error` or exceptions for failures.
- Avoid garbled progress/log output by using `tqdm.write()` or appropriate logging handlers.
- Store metrics/config in experiment result CSV files; use terminal logs for human-readable execution tracing.

### Experiment Unit Policy

Each training script run must execute exactly one experiment.

Rules:

- 1 script run = 1 model = 1 input config = 1 sample/full-data setting = 1 seed = 1 training job.
- Do not run multiple models sequentially from one CLI command.
- Do not run multiple sample sizes sequentially from one CLI command.
- Do not run multiple input configs sequentially from one CLI command.
- Do not run multiple seeds or hyperparameter sweeps sequentially from one CLI command.
- Do not build sweep, grid-search, AutoML, or batch leaderboard runners.
- Leaderboard generation must be post-processing over completed `results/experiments.csv` only.
- Large training must happen only through an explicit single command from the user.
- Do not automatically continue to another experiment after failure.
- Retries must also be explicitly requested.

Rationale:

- Consecutive experiments on the 30,010-sample working dataset and larger raw data can overload GPU, CPU, or memory.
- Mixing multiple experiments in one process makes logs, progress, failures, and result CSV interpretation harder.
- Long-running Transformer/foundation jobs need explicit user control over experiment boundaries.

### GPU Training

Use GPU whenever possible for GPU-capable models.

Default policy:

```text
target_gpu_memory_utilization: 0.90
```

Rules:

- Target up to 90% of available GPU memory during training.
- If `train.py` omits `--batch-size`, estimate a per-model batch size using a synthetic forward/backward probe.
- If `--batch-size` is provided, treat it as a manual experiment setting and disable automatic batch sizing.
- On OOM, reduce batch size first, then embedding dimension, then model depth.
- Record GPU memory usage in experiment outputs.
- Lower the 90% target when other services share the same GPU.

### CPU Training

CPU preprocessing and classical time-series baselines should also use `.venv` by default. PyTorch neural/Transformer/foundation models also run in `.venv`.

Rules:

- The main training path is GPU-based, so LightGBM-style CPU multi-thread optimization is not a default focus.
- Use PyTorch DataLoader `num_workers=0` by default for local Windows experiments.
- Increase `num_workers` only after confirming CSV loading is the bottleneck; try 2, 4, then 8, and use up to 14 only after stability is verified.
- Specify `n_jobs` or worker options only for CPU preprocessing, feature extraction, or batch prediction tasks that need parallelism.
- Reduce workers and record the reason when system load or memory pressure is excessive.
