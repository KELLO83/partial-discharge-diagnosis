# Time-Series Model Families

## Source

- Local plan: `docs/TIMESERIES_MODELS.md`
- Implementation source policy: `docs/MODEL_IMPLEMENTATION_SOURCES.md`

## Relevance

Time-series models are the primary evidence source for composite diagnosis.

## Candidate Groups

| Group | Models |
|---|---|
| Baseline neural | GRU, TCN, InceptionTime, ResNet1D |
| Transformer/SOTA | PatchTST, iTransformer, TimesNet, TimeMixer |
| Foundation/representation | MOMENT, UniTS, GPT4TS, TS2Vec |
| CPU baselines | MiniROCKET, MultiROCKET, HYDRA, feature baselines |

## Current Service Requirement

The selected model must eventually implement:

```text
TimeSeriesInferenceAdapter.run(input) -> TimeSeriesResult
```

The output must include label, confidence, all five class probabilities, and
summary features.
