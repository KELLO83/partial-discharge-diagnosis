# Time-Series Model Candidates

Official implementation sources and wrapper policies are defined in `docs/MODEL_IMPLEMENTATION_SOURCES.md`. This document explains experiment candidates conceptually. Actual implementations should import official repositories, Hugging Face models, or validated libraries whenever practical.

The current time-series task is classification, not forecasting.

```text
input: partial-discharge CSV time-series (20, 7680)
output: 5-class partial-discharge type
```

Labels:

| Label ID | Label name |
| --- | --- |
| 0 | normal |
| 1 | noise |
| 2 | surface_discharge |
| 3 | corona_discharge |
| 4 | void_discharge |

## Model Groups

Experiment models are grouped as:

```text
1. Non-Transformer models
2. Transformer / Modern SOTA models
3. Foundation / Pretrained models
```

Start with Core models and expand to Extended models later. The default training policy is CUDA GPU-based. `train.py` must train exactly one model per run. `core`, `extended`, `all`, and `cpu_only` are descriptive group names and are not valid `--model` values. CPU-only classical baselines are listed as Extended candidates but use separate runners.

## Core Experiments

| Group | Model | Main purpose |
| --- | --- | --- |
| Non-Transformer | GRU | RNN baseline |
| Non-Transformer | InceptionTime | Strong CNN-based time-series baseline |
| Transformer / Modern SOTA | PatchTST | Patch-based Transformer |
| Transformer / Modern SOTA | TimesNet | Converts 1D time-series into 2D temporal variation |
| Foundation / Pretrained | MOMENT | Fine-tuning a pretrained time-series foundation model |

## Extended Experiments

| Group | Model | Main purpose |
| --- | --- | --- |
| Non-Transformer | TCN | Dilated causal convolution baseline |
| Non-Transformer | ResNet1D | Residual 1D CNN baseline |
| Non-Transformer / Modern CNN | ModernTCN | Modern convolution-block CNN/TCN comparison |
| Transformer / Modern SOTA | iTransformer | Attention over variable/channel axis |
| Transformer / Modern SOTA | TimeMixer | Modern multi-resolution mixing model |
| Foundation / Pretrained | UniTS | Unified multi-task time-series model |
| Foundation / Pretrained | GPT4TS / One-Fits-All | Transfers GPT-2 pretrained LM blocks to time-series |
| Representation Learning | TS2Vec | Self-supervised time-series representation learning |

Extended models that may be expensive:

| Model | Expected cost | Reason |
| --- | --- | --- |
| ModernTCN | medium to high | Modern convolution blocks over long sequences |
| iTransformer | medium to high | Attention-based structure sensitive to input length and channel settings |
| TimeMixer | high | Multi-resolution mixing and long-sequence processing |
| UniTS | very high | Heavy unified/foundation-style official implementation |
| GPT4TS / One-Fits-All | very high | GPT-2-style pretrained LM transfer |
| TS2Vec | very high | Self-supervised representation training plus downstream classifier |

`TCN` and `ResNet1D` are relatively cheaper within Extended models, but still require smoke and subset runs before the full 30k dataset.

## Optional CPU-Only Extended Baselines

| Group | Model | Main purpose |
| --- | --- | --- |
| Classical Baseline | MiniROCKET | Strong random-convolution feature baseline with RidgeClassifier |
| Classical Baseline | MultiROCKET | Extended MiniROCKET-style convolution feature baseline |
| sktime Feature-based | SummaryClassifier | Very fast summary-feature baseline |
| sktime Feature-based | Catch22Classifier | Fast baseline using 22 validated features |
| sktime Feature-based | RandomIntervalClassifier | Random-interval feature baseline; subset only if expensive |
| sktime Feature-based | TSFreshClassifier | Automated feature extraction; subset only |
| sktime Feature-based | FreshPRINCE | TSFresh-family ensemble; subset only |
| Classical Baseline | ROCKET | Official `sktime` `RocketClassifier` comparison |
| Classical Ensemble | Arsenal | Official `sktime` ROCKET ensemble; subset only because it can be slow |
| Classical Baseline | HYDRA | Dictionary + convolution classical TSC baseline |
| Feature Baseline | Logistic / SVM / RandomForest | Fast interpretable baseline from statistical, amplitude, and FFT features |
| Feature Foundation | TabPFN | Tabular foundation classifier on extracted features |

The feature baseline is a tabular classifier, not a raw time-series model:

```text
HFCT CSV time-series
-> amplitude / pulse / cycle / phase-bin / FFT / numeric PRPD histogram features
-> Logistic / Linear SVM / RandomForest / TabPFN
-> 5-class classification
```

The current implementation uses CSV signals by default. Metadata is attached only when `--include-metadata` is enabled, and only through a safe numeric whitelist. Do not use class-bearing values such as file paths, sample IDs, label text, or defect details as features.

## Input Shape

Each CSV has:

```text
raw CSV shape = (20, 7680)
```

Do not assume the `20` axis is physical sensor channels. Because JSON `recording_time_length` is 20 and sensor type is usually `HFCT` or `UHF`, interpret it as:

```text
20 rows = 20 measurement segments or pseudo-channels
7680 columns = time points for each segment
```

Model input uses the `20` axis as `pseudo-channel` or `segment dimension`.

Common input formats:

```text
Conv/TCN family:
(batch, pseudo_channels, time) = (B, 20, 7680)

RNN/Hugging Face PatchTST/some Transformer models:
(batch, time, pseudo_channels) = (B, 7680, 20)
```

The DataLoader or wrapper must transpose per model.

## 1. GRU

Group:

```text
Non-Transformer / RNN baseline
```

GRU is a simplified RNN family model with reset and update gates.

Project role:

```text
The first deep-learning time-series classification baseline.
```

Advantages:

- Simpler than LSTM.
- Fewer parameters and faster training.
- Good first baseline.

Cautions:

- Sequential processing over length `7680` can be slow.
- Long-range dependency modeling may be limited.

Expected input:

```text
(B, 7680, 20)
```

## 2. InceptionTime

Group:

```text
Non-Transformer / strong CNN baseline
```

InceptionTime adapts the Inception idea to time-series classification by applying multiple 1D convolution filter sizes in parallel.

Project role:

```text
Stronger non-Transformer baseline than GRU.
```

Advantages:

- Strong known baseline for time-series classification.
- Effective for local patterns, peaks, and short-term variation.
- Easier to parallelize than RNNs.

Cautions:

- Weaker than Transformers for global dependency modeling.
- Memory use depends on kernel sizes and depth.

Expected input:

```text
(B, 7680, 20)
```

## 3. PatchTST

Group:

```text
Transformer / Modern SOTA
```

PatchTST splits long time-series into patches and feeds them as tokens into a Transformer, similar to how ViT patches images.

Project role:

```text
Transformer baseline suited for long partial-discharge signals.
```

Advantages:

- Reduces long sequence length through patching.
- Strong experiment candidate among Transformer models.
- Channel-independent strategies are natural.

Cautions:

- Patch length and stride matter.
- A classification head may need wrapper-specific handling.

Expected input:

```text
(B, 20, 7680)
```

## 4. TimesNet

Group:

```text
Transformer / Modern SOTA
```

TimesNet converts 1D time-series into 2D temporal variation and uses 2D convolution-style structures to learn periodic and repeated patterns.

Project role:

```text
SOTA-style model for repeated peaks, periodicity, and phase patterns in partial-discharge signals.
```

Advantages:

- Strong for periodic time-series patterns.
- The 2D representation can expose richer structure.
- Commonly used in THU Time-Series-Library tasks.

Cautions:

- Input shape and task settings can be implementation-sensitive.
- Classification configuration must be explicit.

Expected input:

```text
(B, 7680, 20)
```

## 5. MOMENT

Group:

```text
Foundation / Pretrained
```

MOMENT is a time-series foundation model pretrained with masked patch reconstruction and adapted through downstream heads.

Project role:

```text
Core foundation-model experiment that transfers pretrained representation to partial-discharge classification.
```

Advantages:

- Supports classification tasks.
- Uses a pretrained backbone.
- Enables head-only training, partial fine-tuning, and full fine-tuning comparisons.

Cautions:

- Patch length, sequence length, and channel settings must match model expectations.
- Full fine-tuning uses more GPU memory.

Expected input:

```text
(B, 20, 7680)
or the shape required by MOMENT processor/config
```

## 6. TCN

TCN uses dilated causal convolutions to build a long receptive field without sequential RNN computation.

Project role:

```text
Convolutional baseline compared against GRU.
```

Expected input:

```text
(B, 20, 7680)
```

## 7. ResNet1D

ResNet1D applies residual blocks to 1D convolutions.

Project role:

```text
Simple and stable 1D CNN baseline.
```

Expected input:

```text
(B, 20, 7680)
```

## 8. MiniROCKET

MiniROCKET extracts random convolution features and trains a simple classifier such as RidgeClassifier.

Project role:

```text
Strong classical baseline for checking whether deep learning is necessary.
```

Cautions:

- Not an end-to-end neural model.
- Usually CPU-based through `sktime/sklearn`.
- Excluded from default GPU `train.py` training.

Expected input:

```text
(samples, channels, time), depending on the implementation
```

## 9. iTransformer

iTransformer uses inverted attention over variables or channels instead of the usual time axis.

Project role:

```text
Transformer experiment for relationships among the 20 segments or pseudo-channels.
```

Expected input:

```text
(B, 7680, 20)
```

## 10. TimeMixer

TimeMixer uses multi-resolution decomposition and mixing, closer to MLP/mixing approaches than attention.

Project role:

```text
Modern SOTA comparison outside standard attention-based Transformers.
```

Expected input:

```text
(B, 7680, 20) or (B, 20, 7680), depending on implementation
```

## 11. UniTS

UniTS is a unified model for multiple time-series tasks such as forecasting, classification, imputation, and anomaly detection.

Project role:

```text
Unified foundation-style model comparison for classification.
```

Expected input:

```text
(B, 7680, 20)
or the shape required by UniTS dataloader/config
```

## 12. GPT4TS / One-Fits-All

GPT4TS / One-Fits-All reuses GPT-2-style pretrained language-model Transformer blocks for time-series analysis, adapting with input/output projections and normalization layers.

Project role:

```text
Experiment in transferring pretrained LLM-style representations to time-series classification.
```

Expected input:

```text
patch/token embeddings passed into a GPT-style backbone
```

## 13. TS2Vec

TS2Vec learns time-series representations through self-supervision and attaches a downstream classifier.

Project role:

```text
Representation-learning experiment with reduced direct label dependence.
```

Expected input:

```text
(B, 7680, 20)
or implementation-specific shape
```

## Recommended Experiment Order

Start with Core experiments:

```text
1. GRU
2. InceptionTime
3. PatchTST
4. TimesNet
5. MOMENT
```

Then extend:

```text
1. iTransformer
2. TCN
3. ModernTCN
4. TimeMixer
5. UniTS
6. GPT4TS / One-Fits-All
7. TS2Vec
8. ResNet1D
9. MiniROCKET / MultiROCKET / sktime feature-based / ROCKET / HYDRA (optional CPU-only classical baseline)
10. Feature baseline / TabPFN (optional CPU-only feature baseline)
```

Use `ml/scripts/run_sktime_classifier.py` or a dedicated runner for classical TSC models provided by `sktime`. Prioritize `Catch22Classifier` and `SummaryClassifier` as fast baselines. Do not run `RandomIntervalClassifier`, `TSFreshClassifier`, `FreshPRINCE`, or `Arsenal` on the full dataset immediately.

## Final Goal

The time-series track should compare model performance and connect the best time-series model results to the VLM stage.

Information that can be passed to the VLM:

```text
time-series model predicted label
time-series model confidence
class probability
hidden embedding
statistical features
phase-bin features
pulse/cycle features
numeric PRPD histogram summary
```

VLM input example:

```text
PRPD image
+ JSON metadata
+ time-series model prediction
+ time-series summary features
```
