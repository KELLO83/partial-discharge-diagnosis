# Official Model Implementation Sources

This document defines when the project should implement a model directly and when it should wrap an official repository or package for partial-discharge CSV time-series classification experiments.

Principles:

- Baselines that are stable PyTorch primitives, such as GRU, may be implemented directly.
- Paper models should be imported from official repositories, Hugging Face, or widely used validated libraries whenever practical.
- If an official implementation is forecasting-oriented, implement only a classification adapter or head in this repository.
- Do not reimplement full paper-model bodies inside this repository.

## Source Table

| Model | Preferred source | Official / validated | Classification fit | Project usage |
|---|---|---:|---:|---|
| GRU | PyTorch `torch.nn.GRU` | official primitive | high | Keep direct baseline implementation |
| TCN | `locuslab/TCN` or `tsai` TCN | official / validated library | high | Use current `tsai` TCN wrapper |
| InceptionTime | `hfawaz/InceptionTime` or `tsai` InceptionTime | official / validated library | high | Use current `tsai` InceptionTime wrapper |
| ResNet1D | `tsai` ResNet family | validated library | high | Use current `tsai` ResNet wrapper |
| MiniROCKET | `sktime` `MiniRocketMultivariate` | validated library | high | Extended candidate; keep separate from GPU `train.py` as CPU-only classical baseline |
| MultiROCKET | `sktime` `MultiRocketMultivariate` | validated library | high | Use CPU-only optional runner |
| SummaryClassifier | `sktime` `SummaryClassifier` | validated library | high | Fast feature-based TSC baseline |
| Catch22Classifier | `sktime` `Catch22Classifier` | validated library | high | Fast baseline based on 22 validated features |
| RandomIntervalClassifier | `sktime` `RandomIntervalClassifier` | validated library | high | Random-interval baseline; requires `--allow-expensive` and subset |
| TSFreshClassifier | `sktime` `TSFreshClassifier` | validated library | medium to high | Many features and can be slow; requires `--allow-expensive` and subset |
| FreshPRINCE | `sktime` `FreshPRINCE` | validated library | medium to high | TSFresh-family ensemble; requires `--allow-expensive` and subset |
| ROCKET | `sktime` `RocketClassifier` | validated library | high | Use CPU-only optional runner |
| Arsenal | `sktime` `Arsenal` | validated library | high | Can be very slow; requires `--allow-expensive` and subset |
| HYDRA | `aeon` HYDRA classifier | validated library | high | Use CPU-only optional runner |
| Feature baseline | scikit-learn Logistic/LinearSVM/RandomForest | validated library | medium | Extract amplitude/pulse/cycle/phase-bin/FFT/numeric PRPD features and use as fast baselines |
| TabPFN | `tabpfn` `TabPFNClassifier` | official / validated library | medium | Optional tabular foundation baseline on extracted features, not raw time-series |
| ModernTCN | `luodhhh/ModernTCN` | official | high | Use official classification implementation wrapper |
| PatchTST | Hugging Face `PatchTSTForClassification`, official `yuqinie98/PatchTST` | official / HF | high | Keep current Hugging Face wrapper direction |
| TimesNet | `thuml/Time-Series-Library` | official | high | Use current TSLib official wrapper |
| iTransformer | `thuml/iTransformer`, `thuml/Time-Series-Library` | official | medium | Official implementation is forecasting-oriented; review classification adapter |
| TimeMixer | `kwuking/TimeMixer`, `thuml/Time-Series-Library`, PyPOTS/NeuralForecast implementations | official / validated library | medium to high | Prefer official wrapper after checking classification support |
| MOMENT | `momentfm`, `moment-timeseries-foundation-model/moment` | official package/repo | high | Use official classification pipeline |
| UniTS | `mims-harvard/UniTS` | official | high | Build wrapper around official repository |
| GPT4TS / One-Fits-All | `DAMO-DI-ML/One_Fits_All` | official | medium | Connect external GPT-2-adapter implementation; review classification head requirements |
| TS2Vec | `yuezhihan/ts2vec` | official | high | Use representation encoder plus classifier head |

## Official / Semi-Official Links

- GRU: PyTorch `torch.nn.GRU`
- TCN official: https://github.com/locuslab/TCN
- TCN tsai: https://timeseriesai.github.io/tsai/models.tcn.html
- InceptionTime official: https://github.com/hfawaz/InceptionTime
- tsai library: https://github.com/timeseriesAI/tsai
- MiniROCKET sktime: https://www.sktime.net/en/v0.20.0/api_reference/auto_generated/sktime.transformations.panel.rocket.MiniRocket.html
- MultiROCKET sktime: https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.transformations.panel.rocket.MultiRocketMultivariate.html
- sktime ROCKET/Arsenal classifiers: https://www.sktime.net/en/stable/api_reference/classification.html
- sktime feature-based classifiers: https://www.sktime.net/en/stable/api_reference/classification.html
- aeon convolution classifiers: https://www.aeon-toolkit.org/en/stable/api_reference/classification.html
- TabPFN: https://github.com/PriorLabs/TabPFN
- ModernTCN official: https://github.com/luodhhh/ModernTCN
- PatchTST official: https://github.com/yuqinie98/PatchTST
- PatchTST Hugging Face: https://huggingface.co/docs/transformers/model_doc/patchtst
- Time-Series-Library: https://github.com/thuml/Time-Series-Library
- TimesNet official implementation source: https://github.com/thuml/Time-Series-Library
- iTransformer official: https://github.com/thuml/iTransformer
- TimeMixer official: https://github.com/kwuking/TimeMixer
- MOMENT official: https://github.com/moment-timeseries-foundation-model/moment
- UniTS official: https://github.com/mims-harvard/UniTS
- GPT4TS / One-Fits-All official: https://github.com/DAMO-DI-ML/One_Fits_All
- TS2Vec official: https://github.com/yuezhihan/ts2vec

## Current Code-State Interpretation

Except for GRU, `ml/timeseries/src/models` should point toward official or external wrappers.

Keep direct implementation:

- `gru.py`: baseline based on PyTorch `torch.nn.GRU`

Official/external wrappers:

- `patchtst.py`: prefer Hugging Face `PatchTSTForClassification`
- `tcn.py`: `tsai` TCN or locuslab official TCN adapter
- `inception_time.py`: `tsai` or `hfawaz/InceptionTime` adapter
- `resnet1d.py`: `tsai` ResNet adapter
- `minirocket.py`: connect through `sktime` after installation as an sklearn-style pipeline
- `run_sktime_classifier.py`: separate CPU runner for `sktime` feature-based classifiers, `RocketClassifier`, and `Arsenal`
- `moderntcn.py`: connect `luodhhh/ModernTCN` classification implementation
- `moment.py`: connect `MOMENTPipeline` after `momentfm` installation
- `units.py`: connect the `mims-harvard/UniTS` repository
- `gpt4ts.py`: connect the `DAMO-DI-ML/One_Fits_All` repository
- `ts2vec.py`: connect the `yuezhihan/ts2vec` repository
- `timesnet.py`: connect the `thuml/Time-Series-Library` repository
- `itransformer.py`: connect `thuml/iTransformer` or `thuml/Time-Series-Library`
- `timemixer.py`: connect `kwuking/TimeMixer` or `thuml/Time-Series-Library`

If an official dependency is not installed or cloned, raising a clear `ImportError` is expected behavior. Do not run experiments with ad hoc fallback implementations pretending to be paper models.

## Application Priority

1. `PatchTST`: keep and clean up the existing Hugging Face connection and config.
2. `MOMENT`: check whether `momentfm` can be installed, then connect the classification pipeline.
3. `ModernTCN`: keep the official classification-repo wrapper.
4. `MiniROCKET`, `MultiROCKET`, `SummaryClassifier`, `Catch22Classifier`, `RandomIntervalClassifier`, `TSFreshClassifier`, `FreshPRINCE`, `ROCKET`, `Arsenal`, `HYDRA`: run as CPU-only optional baselines through dedicated runners when needed.
5. `Feature baseline`, `TabPFN`: run as optional comparisons after extracting amplitude/pulse/cycle/phase-bin/FFT/numeric PRPD features.
6. `TimesNet`: connect `thuml/Time-Series-Library` through a vendor/clone or submodule path.
7. `iTransformer`: review adapter options based on official repo or TSLib.
8. `TimeMixer`: check official classification support and write an adapter.
9. `TS2Vec`: train official encoder and attach a linear classifier head.
10. `UniTS`, `GPT4TS`: treat as lower-priority foundation extended models because dependencies and input formats are heavier.
