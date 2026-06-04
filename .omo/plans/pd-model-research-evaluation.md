# Partial Discharge Model Research And Experiment Evaluation Plan

## TL;DR
> **Summary**: The current VLM-after-time-series direction is correct, but the experiment order should change. Before VLM, lock a leakage-safe split, run feature/classical TSC baselines, then compare selected neural/foundation models under one protocol.
> **Deliverables**:
> - Dataset shape/risk audit report
> - Revised pre-VLM model priority and experiment contract
> - Current plan critique with concrete keep/change/defer decisions
> - VLM readiness gate
> **Effort**: Medium
> **Parallel**: YES - 4 waves
> **Critical Path**: dataset audit -> split contract -> cheap baselines -> neural/foundation shortlist -> VLM gate

## Context
### Original Request
User asked, in Korean, to look at the current dataset shape, investigate whether more time-series or other models should be recommended before VLM, and evaluate the current model experiment plan.

### Interview Summary
No blocking user decision is required. Default assumption: produce a decision-complete research/evaluation plan, not run model training yet.

### Metis Review (gaps addressed)
- Plan critique must be framed as protocol critique, not performance judgment, because `results/experiments.csv` does not exist.
- Split/leakage policy is the highest-risk missing piece.
- Foundation/VLM work must be gated behind cheap baseline results and exported time-series summaries.
- Normalization, metadata, and leakage-prone columns need explicit ablations and guardrails.

## Work Objectives
### Core Objective
Create a reproducible evaluation path that decides which non-VLM models deserve experiments before VLM and whether the current experiment plan is credible.

### Deliverables
- `.omo/evidence/task-1-dataset-audit.md`
- `.omo/evidence/task-2-split-contract.md`
- `.omo/evidence/task-3-baseline-results.md`
- `.omo/evidence/task-4-neural-shortlist.md`
- `.omo/evidence/task-5-plan-evaluation.md`
- `.omo/evidence/task-6-vlm-readiness.md`

### Definition of Done (verifiable conditions with commands)
- `python ml/scripts/validate_dataset.py --fail-on-invalid` exits 0.
- `python train.py --list-models` shows the expected GPU and CPU-only model groups.
- A written recommendation ranks pre-VLM models and explicitly marks `VLM_READY=true/false`.
- The plan evaluation states keep/change/defer decisions for current Core, Extended, CPU-only, and VLM tracks.

### Must Have
- Use current repo evidence: `Train/manifest.csv`, `docs/DATASET_EXPLAIN.md`, `docs/TIMESERIES_MODELS.md`, `docs/VLM_STRATEGY.md`, `docs/MODEL_IMPLEMENTATION_SOURCES.md`, `train.py`, `ml/scripts/*`.
- Main-line model comparison must keep metadata off unless explicitly running ablation.
- Main metrics: accuracy, macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, `pd_to_normal_error_count`.
- Treat path/name/label text/defect fields and `max_discharge_value` as leakage or ablation-only inputs.

### Must NOT Have
- Do not start VLM training before exporting time-series predictions, probabilities, and safe signal summaries.
- Do not rank models from accuracy until a fixed split and preprocessing contract exists.
- Do not use forecasting-only models such as TimesFM, Chronos, Lag-Llama, Moirai, TTM, Time-MoE, Timer as first-line classification candidates.
- Do not put raw CSV values into VLM prompts.

## Current Findings
- Dataset shape: `Train/manifest.csv` has 30,010 rows, 33 columns, and five balanced classes of 6,002 rows each.
- Signal shape: each CSV is `(20, 7680)`; current docs and code treat `20` as pseudo-channel/segment dimension.
- Data quality: reports show 30,010 valid rows, no missing paths, and no invalid shape/NaN/inf issues.
- Confounding risk: `sensor_type` has 29,010 HFCT and 1,000 UHF; UHF appears in a way that may create shortcut risk, so metadata belongs in ablation only.
- Current experiment outputs: `results/experiments.csv` is absent, so the repo has a roadmap and runners but no comparable leaderboard yet.
- Current model set is broad enough; the main correction is priority/order, not adding many new families.

## Recommendation Snapshot
- **Move earlier**: feature baseline, MiniROCKET, MultiROCKET, HYDRA, Catch22/Summary.
- **Keep core, but reorder**: InceptionTime/TCN/ModernTCN should be treated as strong waveform baselines; GRU is a sanity sequence baseline, not the main strength candidate.
- **Keep after baselines**: PatchTST, TimesNet, MOMENT.
- **Defer**: iTransformer, TimeMixer, TS2Vec, UniTS, GPT4TS until baseline and core results justify cost.
- **Optional add**: LSTM-FCN/ALSTM-FCN only if a CNN+RNN hybrid is desired for PD-literature alignment; it is not mandatory because GRU + InceptionTime + TCN cover the space.
- **Do not add now**: VLM image classification or forecasting foundation models.

## Evidence Sources
- Repo docs: `docs/DATASET_EXPLAIN.md`, `docs/TIMESERIES_MODELS.md`, `docs/VLM_STRATEGY.md`, `docs/MODEL_IMPLEMENTATION_SOURCES.md`
- Repo code: `ml/src/data/loader.py`, `ml/src/eval/metrics.py`, `train.py`, `ml/scripts/run_feature_baseline.py`, `ml/scripts/run_minirocket.py`, `ml/scripts/run_multirocket.py`, `ml/scripts/run_hydra.py`, `ml/scripts/run_sktime_classifier.py`
- External references:
  - MiniROCKET: https://arxiv.org/abs/2012.08791
  - MultiROCKET: https://arxiv.org/abs/2102.00457
  - HYDRA: https://arxiv.org/abs/2203.13652
  - InceptionTime: https://arxiv.org/abs/1909.04939
  - LSTM-FCN: https://arxiv.org/abs/1709.05206
  - TS2Vec: https://arxiv.org/abs/2106.10466
  - PatchTST classification docs: https://huggingface.co/docs/transformers/model_doc/patchtst
  - TimesNet: https://arxiv.org/abs/2210.02186
  - iTransformer: https://arxiv.org/abs/2310.06625
  - MOMENT: https://arxiv.org/abs/2402.03885
  - UniTS: https://arxiv.org/abs/2403.00131
  - TimeMixer: https://arxiv.org/abs/2405.14616
  - sktime classifiers: https://www.sktime.net/en/stable/api_reference/classification.html
  - aeon classifiers: https://www.aeon-toolkit.org/en/stable/api_reference/classification.html

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after for research artifacts; no production code changes in this plan.
- QA policy: Every task has CLI/data-surface scenarios with concrete commands.
- Evidence: `.omo/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
Wave 1: Tasks 1-2
Wave 2: Task 3
Wave 3: Task 4
Wave 4: Tasks 5-6

### Dependency Matrix
- Task 1 blocks Tasks 2-6.
- Task 2 blocks Tasks 3-6.
- Task 3 blocks Tasks 4-6.
- Task 4 blocks Tasks 5-6.
- Task 5 blocks Task 6.

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: References + Acceptance Criteria + QA Scenarios.

- [ ] 1. Dataset Shape And Risk Audit

  **What to do**: Produce `.omo/evidence/task-1-dataset-audit.md` summarizing manifest rows/columns, label distribution, CSV shape, sensor/equipment distributions, quality report status, leakage-risk columns, and whether `(20, 7680)` should be modeled as `(pseudo_channel, time)`.
  **Must NOT do**: Do not train models. Do not use path/name/label text as features.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2,3,4,5,6 | Blocked By: none

  **References**:
  - Pattern: `docs/DATASET_EXPLAIN.md` - dataset shape and class definitions.
  - Pattern: `ml/src/data/loader.py` - `read_timeseries_csv()` enforces `(20, 7680)`.
  - Pattern: `reports/data_quality_summary.json` - quality/leakage summary.
  - Pattern: `results/eda_full/eda_summary.json` - signal statistics by label.

  **Acceptance Criteria**:
  - [ ] `python ml/scripts/validate_dataset.py --fail-on-invalid` exits 0.
  - [ ] Evidence file records rows=`30010`, labels=`6002` per class, CSV shape=`(20, 7680)`, and leakage-risk columns.
  - [ ] Evidence file states that metadata and `max_discharge_value` are ablation-only unless proven safe.

  **QA Scenarios**:
  ```text
  Scenario: Dataset validation happy path
    Tool: bash
    Steps: python ml/scripts/validate_dataset.py --fail-on-invalid
    Expected: exit code 0 and summary reports zero invalid rows
    Evidence: .omo/evidence/task-1-dataset-validate.txt

  Scenario: Manifest summary check
    Tool: bash
    Steps: python - <<'PY'
import pandas as pd
df = pd.read_csv('Train/manifest.csv')
print(df.shape)
print(df['label_id'].value_counts().sort_index().to_dict())
print(df['sensor_type'].value_counts(dropna=False).to_dict())
PY
    Expected: `(30010, 33)`, five classes with 6002 rows each, sensor distribution printed
    Evidence: .omo/evidence/task-1-manifest-summary.txt
  ```

  **Commit**: NO | Message: `docs(research): audit partial-discharge dataset shape` | Files: `.omo/evidence/task-1-*`

- [ ] 2. Split And Leakage Contract

  **What to do**: Define the split contract for credible model comparison in `.omo/evidence/task-2-split-contract.md`: development split, leakage-resistant split, metadata ablation rules, and duplicate/near-duplicate checks.
  **Must NOT do**: Do not report final model superiority from random split alone.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3,4,5,6 | Blocked By: 1

  **References**:
  - Pattern: `ml/src/data/loader.py` - `make_stratified_split()` fallback behavior.
  - Pattern: `ml/scripts/make_splits.py` - existing split manifest script.
  - Pattern: `results/metadata_crosstab/metadata_crosstab_summary.csv` - metadata/class shortcut checks.

  **Acceptance Criteria**:
  - [ ] Evidence file defines two split tiers: `random_stratified_dev` and `group_or_session_stress`.
  - [ ] Evidence file marks `sensor_type`, paths, label strings, `defect_details`, `defect_nums`, and `max_discharge_value` as excluded from main features.
  - [ ] Evidence file defines at least three seeds for final model comparison.

  **QA Scenarios**:
  ```text
  Scenario: Current split behavior check
    Tool: bash
    Steps: python - <<'PY'
import pandas as pd
df = pd.read_csv('Train/manifest.csv')
print(df['split'].value_counts(dropna=False).to_dict())
PY
    Expected: current split values are printed; if only `train`, plan records random split fallback risk
    Evidence: .omo/evidence/task-2-current-split.txt

  Scenario: Metadata shortcut check
    Tool: bash
    Steps: python - <<'PY'
import pandas as pd
df = pd.read_csv('Train/manifest.csv')
print(pd.crosstab(df['label_id'], df['sensor_type']))
PY
    Expected: crosstab printed and any label-specific sensor shortcut noted
    Evidence: .omo/evidence/task-2-sensor-crosstab.txt
  ```

  **Commit**: NO | Message: `docs(research): define leakage-safe split contract` | Files: `.omo/evidence/task-2-*`

- [ ] 3. Cheap Strong Baseline Priority

  **What to do**: Run or at minimum command-verify the first pre-VLM baselines in this exact order: feature logistic/SVM/RF, MiniROCKET, MultiROCKET, HYDRA, Catch22/Summary. Record which commands are ready, which dependencies are missing, expected runtime, and result columns.
  **Must NOT do**: Do not place these after foundation/VLM models in the recommendation.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 4,5,6 | Blocked By: 1,2

  **References**:
  - Pattern: `ml/scripts/run_feature_baseline.py` - feature baseline runner.
  - Pattern: `ml/scripts/run_minirocket.py` - MiniROCKET runner.
  - Pattern: `ml/scripts/run_multirocket.py` - MultiROCKET runner.
  - Pattern: `ml/scripts/run_hydra.py` - HYDRA runner.
  - External: https://arxiv.org/abs/2012.08791 - MiniROCKET.
  - External: https://arxiv.org/abs/2102.00457 - MultiROCKET.
  - External: https://arxiv.org/abs/2203.13652 - HYDRA.

  **Acceptance Criteria**:
  - [ ] Evidence ranks `feature_*`, `MiniROCKET`, `MultiROCKET`, `HYDRA`, `Catch22/Summary` before expensive neural/foundation models.
  - [ ] Evidence records exact smoke commands with `--sample-size 100` or the runner default if safe.
  - [ ] Evidence records that metadata is off for the main line.

  **QA Scenarios**:
  ```text
  Scenario: CPU baseline discovery
    Tool: bash
    Steps: python train.py --list-models
    Expected: output contains `cpu_only:` and runner hints for `minirocket`, `multirocket`, `hydra`, `feature_logistic`
    Evidence: .omo/evidence/task-3-list-models.txt

  Scenario: Feature baseline smoke command
    Tool: bash
    Steps: python ml/scripts/run_feature_baseline.py --model logistic --sample-size 100 --feature-set small --output .omo/evidence/task-3-feature-smoke.csv
    Expected: exit code 0 and output CSV contains one row with valid macro F1/balanced accuracy fields
    Evidence: .omo/evidence/task-3-feature-smoke.txt
  ```

  **Commit**: NO | Message: `docs(research): prioritize classical pre-vlm baselines` | Files: `.omo/evidence/task-3-*`

- [ ] 4. Neural And Foundation Shortlist

  **What to do**: Convert the current broad roadmap into a gated shortlist. Recommended order: InceptionTime/TCN/ModernTCN, GRU sanity, PatchTST, TimesNet, MOMENT, then extended iTransformer/TimeMixer/TS2Vec/UniTS/GPT4TS only if baseline gaps justify cost.
  **Must NOT do**: Do not recommend running every model full-scale in one wave.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 5,6 | Blocked By: 3

  **References**:
  - Pattern: `train.py` - core/extended model groups and presets.
  - Pattern: `docs/TIMESERIES_MODELS.md` - current model plan.
  - Pattern: `docs/MODEL_IMPLEMENTATION_SOURCES.md` - official implementation policy.
  - External: https://arxiv.org/abs/1909.04939 - InceptionTime.
  - External: https://huggingface.co/docs/transformers/model_doc/patchtst - PatchTST classification.
  - External: https://arxiv.org/abs/2210.02186 - TimesNet.
  - External: https://arxiv.org/abs/2402.03885 - MOMENT.

  **Acceptance Criteria**:
  - [ ] Evidence assigns each current model one of: `run_now`, `run_after_baseline`, `defer`, `avoid_now`.
  - [ ] Evidence includes normalization ablations: no normalization, per-sample z-score, train-global/per-channel, robust/amplitude-preserving.
  - [ ] Evidence defines subset gates: 100, 1,000, 5,000, then full 30,010 only after stable smoke.

  **QA Scenarios**:
  ```text
  Scenario: GPU model discovery
    Tool: bash
    Steps: python train.py --list-models
    Expected: output contains core models `gru`, `inception_time`, `patchtst`, `timesnet`, `moment`
    Evidence: .omo/evidence/task-4-list-gpu-models.txt

  Scenario: One-model rule check
    Tool: bash
    Steps: python train.py --model gru,patchtst --sample-size 10
    Expected: nonzero exit and error explaining one train.py run must train exactly one model
    Evidence: .omo/evidence/task-4-one-model-rule.txt
  ```

  **Commit**: NO | Message: `docs(research): define gated neural shortlist` | Files: `.omo/evidence/task-4-*`

- [ ] 5. Current Experiment Plan Evaluation

  **What to do**: Write `.omo/evidence/task-5-plan-evaluation.md` with a direct critique of the current plan: keep, change, defer, avoid. The evaluation must explicitly answer whether more time-series/other models are recommended before VLM.
  **Must NOT do**: Do not claim measured superiority without experiment results.

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 6 | Blocked By: 4

  **References**:
  - Pattern: `docs/PRD.md` - current staged strategy.
  - Pattern: `docs/TIMESERIES_MODELS.md` - current Core/Extended/CPU-only lineup.
  - Pattern: `docs/VLM_STRATEGY.md` - VLM role and input/output design.
  - Pattern: `results/feature_eda_small/feature_separability.csv` - existing signal feature separability.

  **Acceptance Criteria**:
  - [ ] `Keep`: classification-not-forecasting framing, VLM as reporting/diagnosis, official wrappers, metric suite.
  - [ ] `Change`: make split/leakage audit first, move feature/ROCKET/HYDRA earlier, add normalization ablations, add seed/runtime gates.
  - [ ] `Defer`: UniTS, GPT4TS, TS2Vec, iTransformer, TimeMixer until cheap/core baselines are established.
  - [ ] `Avoid now`: PRPD-only image classifier as main goal, raw CSV prompt VLM, forecasting foundation models as first-line classifiers.

  **QA Scenarios**:
  ```text
  Scenario: Current docs consistency check
    Tool: bash
    Steps: python - <<'PY'
from pathlib import Path
for p in ['docs/PRD.md','docs/TIMESERIES_MODELS.md','docs/VLM_STRATEGY.md']:
    text = Path(p).read_text(encoding='utf-8')
    print(p, 'classification' in text.lower(), 'VLM' in text or 'vlm' in text.lower())
PY
    Expected: each document prints true signals for relevant strategy terms
    Evidence: .omo/evidence/task-5-docs-consistency.txt

  Scenario: Experiment results absence check
    Tool: bash
    Steps: test -f results/experiments.csv && wc -l results/experiments.csv || echo 'NO results/experiments.csv'
    Expected: current evidence records whether a leaderboard exists before judging performance
    Evidence: .omo/evidence/task-5-results-status.txt
  ```

  **Commit**: NO | Message: `docs(research): evaluate current model experiment plan` | Files: `.omo/evidence/task-5-*`

- [ ] 6. VLM Readiness Gate

  **What to do**: Define a binary VLM gate in `.omo/evidence/task-6-vlm-readiness.md`. VLM is allowed only after a leakage-safe time-series baseline exports prediction label, confidence/probabilities, selected signal features, and failure cases.
  **Must NOT do**: Do not recommend VLM as a replacement for untested time-series classifiers.

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: final verification | Blocked By: 5

  **References**:
  - Pattern: `docs/VLM_STRATEGY.md` - VLM is multimodal report generation, not image-only classifier.
  - Pattern: `ml/src/eval/metrics.py` - confidence/metric fields that should inform VLM context.
  - Pattern: `docs/PRD.md` - Strategy A and Strategy B VLM path.

  **Acceptance Criteria**:
  - [ ] Evidence contains `VLM_READY=false` until at least one leakage-safe baseline result and summary export exists.
  - [ ] Evidence defines VLM Strategy A input: PRPD image + safe metadata + time-series prediction/probabilities + safe numeric summary.
  - [ ] Evidence defines Strategy B as optional: waveform/spectrogram/PRPD-derived images only after Strategy A works.
  - [ ] Evidence defines VLM output evaluation: label match, JSON parse success, hallucination check, metadata/TS evidence usage.

  **QA Scenarios**:
  ```text
  Scenario: VLM strategy guardrail check
    Tool: bash
    Steps: python - <<'PY'
from pathlib import Path
text = Path('docs/VLM_STRATEGY.md').read_text(encoding='utf-8')
for key in ['원본 CSV 전체를 VLM 프롬프트에 넣지 않는다', 'Qwen2.5-VL-3B', 'JSON']:
    print(key, key in text)
PY
    Expected: each key guardrail prints True
    Evidence: .omo/evidence/task-6-vlm-guardrails.txt

  Scenario: Time-series output dependency check
    Tool: bash
    Steps: test -f results/experiments.csv && echo 'VLM_READY_INPUT_EXISTS' || echo 'VLM_READY=false: missing results/experiments.csv'
    Expected: if experiment output is missing, readiness is false and reason is recorded
    Evidence: .omo/evidence/task-6-readiness-status.txt
  ```

  **Commit**: NO | Message: `docs(research): define vlm readiness gate` | Files: `.omo/evidence/task-6-*`

## Final Verification Wave (MANDATORY - after ALL implementation tasks)
> ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
- [ ] F1. Plan Compliance Audit
- [ ] F2. Evidence File Audit
- [ ] F3. Real Manual QA
- [ ] F4. Scope Fidelity Check

## Commit Strategy
No commit by default. If the user requests execution and commits later, use atomic Conventional Commits and include this footer:

```text
Plan: .omo/plans/pd-model-research-evaluation.md
```

## Success Criteria
- The dataset audit states the true input shape, label distribution, quality status, and leakage risks.
- The model recommendation answers which additional time-series/non-VLM models to try before VLM.
- The current experiment plan evaluation states what to keep, change, defer, and avoid.
- The VLM readiness gate is binary and evidence-based.
