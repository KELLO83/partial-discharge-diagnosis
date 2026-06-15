# PRD: Partial-Discharge Time-Series and Small VLM Diagnosis Project

## 1. Project Overview

This project uses the AI-Hub industrial electrical fire prevention partial-discharge dataset to build diagnosis models for partial-discharge states.

The core direction is not single-image vision classification. The project focuses on:

1. Classification from partial-discharge time-series data.
2. A small VLM diagnosis model that combines PRPD images, metadata, and time-series information.

Because image-classification experience is assumed, ResNet/EfficientNet-style single-image classifiers are excluded from the core scope. The focus is time-series modeling and multimodal VLM alignment.

## 2. Problem Definition

Partial discharge in industrial power equipment is an important precursor to electrical fire and insulation failure. This project uses PRPD images, partial-discharge time-series signals, and equipment/environment metadata to diagnose the current equipment state.

The basic classification target has five classes:

| Label ID | Label name |
| --- | --- |
| 0 | normal |
| 1 | noise |
| 2 | surface_discharge |
| 3 | corona_discharge |
| 4 | void_discharge |

The final VLM should output either natural-language diagnosis or structured JSON diagnosis that a field engineer can understand, not only a class index.

## 3. Goals

### 3.1 Primary Goals

- Understand the `.PNG`, `.CSV`, and `.JSON` matching structure.
- Generate `manifest.csv` from JSON labels.
- Load partial-discharge time-series CSV files and analyze basic statistics and sequence structure.
- Train time-series baseline models.
- Measure time-series classification performance.

### 3.2 Secondary Goals

- Build a VLM instruction dataset from PRPD images, JSON metadata, and time-series summary features.
- Fine-tune a small VLM using LoRA or QLoRA.
- Train the VLM to diagnose partial-discharge types in natural language or JSON.

### 3.3 Extended Goals

- Convert time-series CSV files into graph images and use them with PRPD images as multi-image VLM input.
- Compare Strategy A and Strategy B.
- Evaluate both classification accuracy and generated diagnosis quality.

## 4. Scope

In scope:

- Data-structure analysis
- Manifest generation
- Time-series preprocessing
- Time-series classification model development
- Time-series feature extraction
- VLM instruction dataset generation
- Small VLM LoRA/QLoRA fine-tuning
- Natural-language diagnosis generation
- Structured JSON diagnosis generation

Out of scope:

- Building single-image vision models such as ResNet or EfficientNet as the main outcome
- Treating PRPD image-only classification as the primary result
- Time-series forecasting or future-waveform prediction
- Placing full raw CSV data directly in VLM prompts
- Training large VLMs or 70B-scale models
- Building a production deployment system for industrial sites

## 5. User Scenarios

### Scenario 1: Time-Series Classification

The user provides a partial-discharge time-series CSV file. The model analyzes the signal and classifies it as one of `normal`, `noise`, `surface_discharge`, `corona_discharge`, or `void_discharge`.

Expected output:

```json
{
  "label_id": 3,
  "label_name": "corona_discharge",
  "confidence": 0.87
}
```

### Scenario 2: VLM Natural-Language Diagnosis

The user provides a PRPD image, equipment metadata, environment metadata, and time-series summary features. The VLM combines them and outputs a diagnosis in text.

Expected output:

```text
The diagnosis is corona discharge. The PRPD image shows discharge patterns concentrated in specific phase regions, and the time-series signal contains repeated peaks. Insulation inspection and continued monitoring are recommended.
```

### Scenario 3: VLM Structured Diagnosis

The VLM outputs JSON for easier system integration.

Expected output:

```json
{
  "diagnosis": "corona_discharge",
  "label_id": 3,
  "risk_level": "caution",
  "reason": "The PRPD pattern and time-series summary features are consistent with corona discharge.",
  "recommended_action": "Inspect high-voltage insulation areas and monitor whether discharge signals increase."
}
```

## 6. Development Strategy

Project sequence:

```text
Stage 1: Inspect data structure and generate the manifest
Stage 2: Build time-series-only models
Stage 3: Extract time-series summary features
Stage 4: Build a Strategy A VLM baseline
Stage 5: Extend to Strategy B multi-image VLM experiments
Stage 6: Compare Strategy A and Strategy B
```

## 7. Time-Series Model Strategy

Input:

```text
partial-discharge time-series CSV
```

Output:

```text
partial-discharge class label 0 through 4
```

This track is classification, not forecasting.

```text
Not forecasting:
past time-series -> future time-series prediction

Classification:
full partial-discharge time-series -> 5-class discharge type
```

Forecasting-oriented foundation models such as TimesFM, Chronos, and Lag-Llama are excluded. Prioritize time-series classification models and time-series foundation models that support downstream classification.

Current CSV shape:

```text
sample shape = (20, 7680)
```

Interpretation:

```text
20 rows = 20 measurement segments or pseudo-channels
7680 columns = time points for each segment
```

Do not assume the `20` axis is physical sensor channels. Treat it as a practical pseudo-channel or segment dimension until dataset documentation confirms otherwise.

Common model input shapes:

```text
Conv/TCN family: (batch, pseudo_channels, time) = (B, 20, 7680)
RNN/Hugging Face PatchTST/some Transformer family: (batch, time, pseudo_channels) = (B, 7680, 20)
```

### Core Experiments

| Group | Model | Purpose |
| --- | --- | --- |
| Non-Transformer | GRU | Basic RNN baseline |
| Non-Transformer | InceptionTime | Strong CNN/Inception baseline for time-series classification |
| Transformer / Modern SOTA | PatchTST | Patch-based Transformer classifier |
| Transformer / Modern SOTA | TimesNet | Converts 1D time-series into 2D temporal variation |
| Foundation / Pretrained | MOMENT | Fine-tune a pretrained time-series foundation model |

Core order:

```text
1. GRU Classifier
2. InceptionTime Classifier
3. PatchTST Classifier
4. TimesNet Classifier
5. MOMENT Fine-tuning
```

### Extended Experiments

Run these only after the Core pipeline is stable.

| Group | Model | Purpose |
| --- | --- | --- |
| Non-Transformer | TCN | Dilated causal convolution baseline |
| Non-Transformer | ResNet1D | 1D residual convolution baseline |
| Non-Transformer / Modern CNN | ModernTCN | Modern convolution-block classifier |
| Transformer / Modern SOTA | iTransformer | Inverted attention over variable/channel axis |
| Transformer / Modern SOTA | TimeMixer | Multi-resolution decomposition and mixing |
| Foundation / Pretrained | UniTS | Unified model that supports classification among other tasks |
| Foundation / Pretrained | GPT4TS / One-Fits-All | Reuses GPT-2-style pretrained language-model blocks for time-series |
| Representation Learning | TS2Vec | Self-supervised representation learning plus downstream classifier |

CPU-only baselines such as MiniROCKET, MultiROCKET, HYDRA, feature baselines, and `sktime` classifiers are optional comparisons and must use dedicated runners rather than the GPU `train.py` path.

### Model Notes

Non-Transformer models establish strong baselines. GRU is the first simple baseline; InceptionTime, TCN, ResNet1D, and ModernTCN test convolutional approaches for local pulse and waveform patterns.

Transformer / Modern SOTA models test whether patching, temporal variation modeling, inverted attention, or mixing architectures help classify long partial-discharge signals.

Foundation / Pretrained models test whether pretrained time-series representations can transfer to partial-discharge downstream classification.

### Final Time-Series Strategy

Implement Core experiments first:

```text
GRU -> InceptionTime -> PatchTST -> TimesNet -> MOMENT
```

Then selectively add Extended experiments:

```text
iTransformer -> TCN -> TimeMixer -> UniTS -> GPT4TS -> TS2Vec -> ResNet1D
```

Optional CPU-only classical/feature baselines should be compared after GPU results are organized.

## 8. VLM Model Strategy

Input:

```text
PRPD image
+ equipment metadata
+ environment metadata
+ time-series summary features
```

Output:

```text
natural-language diagnosis or structured JSON diagnosis
```

Current local baseline:

- SmolVLM2-2.2B-Instruct with QLoRA

Future candidate models:

- Qwen3-VL-2B-Instruct
- Qwen2.5-VL-3B-Instruct
- Qwen3-VL-4B-Instruct
- PaliGemma / PaliGemma 2
- Small LLaVA-family VLMs

Use 2B to 3B models because the target GPU is an RTX 4060 Laptop with 8GB VRAM.

## 9. Difference Between Classifiers and VLM Training

The main difference is not just the model family. The training objective changes.

Traditional image/time-series classifiers predict one class index. A VLM receives image, metadata, and time-series context, then autoregressively predicts the next text tokens for a target answer.

| Item | Traditional classifier | Small VLM |
| --- | --- | --- |
| Input | Image or time-series | PRPD image + metadata + time-series summary |
| Output | Class probability vector | Natural-language or JSON text |
| Example output | `[0.1, 0.0, 0.8, 0.1, 0.0]` | `The diagnosis is surface discharge.` |
| Objective | Predict target class index | Predict next tokens in the target answer |
| Loss | Multi-class cross entropy | Autoregressive token-level cross entropy |
| Evaluation | Accuracy, F1 | Label match, JSON parse rate, diagnosis quality |

Classifier structure:

```text
input data
  -> encoder
  -> classification head
  -> 5-class probabilities
  -> cross entropy loss
```

VLM structure:

```text
PRPD image + text context
  -> vision encoder + LLM
  -> next-token prediction
  -> token-level cross entropy loss
```

This reframes partial-discharge diagnosis from plain classification into explainable text generation.

## 10. How to Feed Time-Series Information to the VLM

VLMs primarily consume images and text. Full raw CSV prompts are inappropriate. Convert time-series data into VLM-friendly representations.

Two strategies are compared:

## 11. Strategy A: Time-Series Summary Text + PRPD Image

This is the recommended first implementation.

Extract key features from the CSV and include them in the text prompt. Use the PRPD image as the image input. Add equipment metadata and time-series summaries as text.

Structure:

```text
PRPD image
  -> vision encoder

time-series summary features + equipment metadata + environment metadata
  -> text prompt

vision output + text prompt
  -> small VLM
  -> natural-language or JSON diagnosis
```

Example prompt:

```text
Equipment name: 22.9kV switchgear.
Insulator type: gas insulation.
Temperature: 25 C.
Humidity: 60%.
Time-series summary: RMS=0.221, max=1.83, min=-1.76, dominant_frequency=60.0Hz.
Time-series model prediction: corona_discharge, confidence=0.87.

Using the attached PRPD image and the above information, diagnose the current partial-discharge state.
```

Advantages:

- Lower implementation complexity.
- Small VLMs can often use numeric text features effectively.
- Time-series model results connect naturally to the VLM.
- Dataset generation is simple.
- Classification accuracy and diagnosis quality can both be evaluated.

Limitations:

- Fine waveform details are compressed.
- VLM performance depends on feature-engineering quality.

## 12. Strategy B: Time-Series Graph Image + PRPD Image

Convert the time-series CSV into a graph image and pass it with the PRPD image as multi-image input.

Structure:

```text
PRPD image
  -> vision encoder

time-series waveform image
  -> vision encoder

equipment metadata + environment metadata
  -> text prompt

visual features from both images + text prompt
  -> small VLM
  -> natural-language or JSON diagnosis
```

Advantages:

- Preserves raw waveform shape visually.
- Uses multi-image VLM capabilities.
- Reduces manual feature-engineering requirements.

Limitations:

- The graph style, axis range, resolution, and normalization must be fixed.
- Two images increase GPU memory usage.
- Model and training code must properly support multi-image input.

## 13. Strategy Comparison

| Item | Strategy A: summary text | Strategy B: graph image |
| --- | --- | --- |
| Implementation complexity | Low | Medium |
| GPU memory | Lower | Higher |
| Raw time-series preservation | Lower | Higher |
| Feature-engineering dependency | Higher | Lower |
| Multi-image VLM dependency | Lower | Higher |
| Recommended order | First experiment | Second-stage extension |

## 14. VLM Training Data Format

Strategy A example:

```json
{
  "image": "path/to/prpd.png",
  "messages": [
    {
      "role": "user",
      "content": "Equipment name: 22.9kV switchgear. Insulator type: gas insulation. Temperature: 25 C. Humidity: 60%. Time-series summary: RMS=0.221, max=1.83, dominant_frequency=60.0Hz. Diagnose the partial-discharge state from the PRPD image and metadata."
    },
    {
      "role": "assistant",
      "content": "The diagnosis is corona discharge. The PRPD image and operating conditions indicate a corona-discharge pattern. Inspect high-voltage insulation and continue monitoring."
    }
  ]
}
```

Structured output example:

```json
{
  "diagnosis": "corona_discharge",
  "label_id": 3,
  "risk_level": "caution",
  "reason": "The PRPD pattern and time-series summary features are consistent with corona discharge.",
  "recommended_action": "Inspect high-voltage insulation areas and monitor whether discharge signals increase."
}
```

## 15. Training Method

Full VLM fine-tuning is expensive, so prioritize PEFT.

Priority:

1. Apply LoRA to LLM layers.
2. Train the projection layer or apply LoRA to it.
3. If memory allows, apply LoRA to selected vision-encoder attention layers.

Memory-constrained setup:

```text
freeze vision encoder
+ LoRA on language/projection layers
```

Less constrained setup:

```text
partial vision-encoder LoRA
+ train projection layer
+ LoRA on language layers
```

## 16. Evaluation Metrics

Time-series model evaluation:

- Accuracy
- F1-score
- Confusion matrix
- Per-class precision/recall

VLM evaluation:

- Accuracy after extracting labels from output
- JSON parse success rate
- `label_id` match rate
- Manual diagnosis quality review
- Hallucination checks
- Metadata-use checks

## 17. Initial Development Order

1. Download AI-Hub sample data.
2. Inspect real folder names and JSON examples.
3. Generate `manifest.csv` from JSON files.
4. Run `python ml/timeseries/scripts/validate_dataset.py --fail-on-invalid`.
5. Generate a fixed split manifest with `python ml/timeseries/scripts/make_splits.py --manifest data/manifest.csv --output data/manifest_random_split_seed42.csv`.
6. Run initial EDA with `python ml/timeseries/scripts/run_eda.py`.
7. Check label distribution, metadata distribution, CSV shapes, signal statistics, phase-bin pulse distribution, and leakage-risk columns.
8. Exclude paths, label text, defect information, and `max_discharge_value` from default feature baselines.
9. Validate feature baselines and CPU classical baselines with smoke runs.
10. Run GPU neural/foundation time-series smoke tests.
11. Train GPU neural/foundation models with a shared metric/result schema.
12. Extract time-series summary features.
13. Build a Strategy A VLM instruction dataset.
14. Fine-tune a small VLM with LoRA or QLoRA.
15. Generate time-series graph images for Strategy B.
16. Run multi-image VLM experiments.
17. Compare Strategy A and Strategy B.

## 18. Success Criteria

Primary success:

- Manifest generation works on the real data structure.
- Time-series CSV loading and preprocessing work.
- A time-series baseline model can train.
- Validation accuracy can be measured.

Secondary success:

- VLM instruction dataset generation works.
- Small VLM LoRA training works.
- The VLM can use PRPD images and time-series summaries to output diagnosis.
- Label ID or label name can be extracted reliably from output.

Extended success:

- Time-series graph images can be generated.
- VLM experiments can consume PRPD image and time-series graph image together.
- Strategy A and Strategy B results can be compared.

## 19. Design Notes

- Do not assume the VLM vision encoder understands PRPD images perfectly out of the box.
- A pretrained vision encoder can still extract generic visual features such as points, lines, density, distribution, and symmetry.
- Lightweight LoRA/PEFT adaptation is appropriate for the PRPD domain.
- Do not place full raw CSV data in VLM prompts.
- Compress time-series data through a separate model or feature extractor before passing it to the VLM.
- The core contribution is time-series modeling plus VLM-based explainable diagnosis, not single-image classification.
