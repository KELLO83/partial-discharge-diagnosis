# PRPD Similarity Retrieval Evaluation Report

## Full Index

- Index: `prpd_similarity_retrieval/case_feature_index.npz`
- Cases: `30,010`
- Query label usage: hidden
- Top-k: `3`
- Batch size: `256`
- Elapsed: `577.185s`

## Overall Result

| Retriever | Evaluated | Top-1 label match | Top-3 label match |
|---|---:|---:|---:|
| Feature retrieval | 30,010 | 1.000000 | 1.000000 |
| Prototype encoder | 30,010 | 1.000000 | 1.000000 |
| Learned projection | 30,010 | 0.998834 | 0.999367 |
| Metadata baseline | 30,010 | 0.233322 | 0.233322 |
| Delta | 30,010 | +0.766678 | +0.766678 |

## Prototype Encoder

- Index: `prpd_similarity_retrieval/case_embedding_index.prototype.npz`
- Encoder: deterministic random projection + label-centroid calibration
- Embedding dimension: `202`
- Evaluation elapsed: `20.929s`
- Top-1 label match: `1.000000`
- Top-3 label match: `1.000000`

This prototype validates the embedding index/search/evaluation path. It is not yet a production neural encoder because it uses supervised label centroids fitted on the available index. The next evaluation should use hard splits or human-reviewed neighbor relevance.

## Learned Projection Encoder

- Index: `prpd_similarity_retrieval/case_embedding_index.learned.npz`
- Encoder: feature standardization + PCA projection + label centroid affinity
- Encoder version: `supervised_projection_encoder_v1`
- Embedding dimension: `94`
- Top-1 label match: `0.998834`
- Top-3 label match: `0.999367`

The learned projection index stores the encoder state with the embedding matrix, so backend runtime can transform a new inspection case into the same embedding space. This is still not a final CNN/TS2Vec model; it is the operational embedding retrieval path used as the next baseline.

## Hard Split Sanity Check

- Command: `python -m prpd_similarity_retrieval.cli evaluate-hard-split --index prpd_similarity_retrieval\case_feature_index.npz --limit 50 --top-k 3 --batch-size 32 --include-prototype`
- Split field: `equipment_name`
- Holdout: `25.8kV GIS`
- Train candidates: `25,010`
- Holdout queries: `50`
- Query label usage: hidden

| Retriever | Evaluated | Top-1 label match | Top-3 label match |
|---|---:|---:|---:|
| Feature retrieval | 50 | 0.320000 | 0.320000 |
| Prototype encoder | 50 | 0.060000 | 0.320000 |
| Metadata baseline | 50 | 0.000000 | 0.000000 |

This is a limited sanity check, not the final score. It confirms that the evaluation harness can remove the holdout equipment group from candidate search and exposes a more realistic gap than leave-one-out evaluation.

## Equipment Hard Split Sample Report

- Report file: `prpd_similarity_retrieval/hard_split_report.sample.md`
- Command: `python -m prpd_similarity_retrieval.cli evaluate-hard-split-report --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --limit-per-holdout 30 --top-k 3 --batch-size 32 --include-prototype --format markdown --output prpd_similarity_retrieval\hard_split_report.sample.md`
- Scope: all `equipment_name` holdout groups, limited to `30` queries per holdout
- Query label usage: hidden

| Holdout | Query | Train | Feature top-1 | Feature top-3 | Metadata top-1 | Metadata top-3 | Prototype top-1 | Prototype top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 25.8kV GIS | 30 | 25,010 | 0.300000 | 0.300000 | 0.000000 | 0.000000 | 0.033333 | 0.300000 |
| ACSR-OC | 30 | 26,675 | 0.633333 | 0.633333 | 0.000000 | 0.000000 | 0.633333 | 0.633333 |
| CNCV-W | 30 | 26,675 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| TFR-CV | 30 | 26,675 | 0.666667 | 0.666667 | 0.000000 | 0.000000 | 0.666667 | 0.733333 |
| 계기용 변압기 | 30 | 26,675 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| 단상 유입변압기 | 30 | 26,675 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.200000 | 0.200000 |
| 전력용 유입변압기 | 30 | 26,675 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| 22.9kV 배전반 | 30 | 27,510 | 0.833333 | 0.900000 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| 7.2kV 배전반 | 30 | 27,510 | 0.533333 | 0.533333 | 0.000000 | 0.000000 | 0.900000 | 0.933333 |

The equipment hard split shows that the current handcrafted feature baseline is not uniformly reliable across held-out equipment groups. `CNCV-W` and `단상 유입변압기` need full-report verification and nearest-neighbor inspection before treating the retrieval score as operationally meaningful.

## Full Equipment Hard Split

- Report file: `prpd_similarity_retrieval/hard_split_report.full.feature.md`
- Command: `python -m prpd_similarity_retrieval.cli evaluate-hard-split-report --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000 --format markdown --output prpd_similarity_retrieval\hard_split_report.full.feature.md`
- Scope: all `equipment_name` holdout groups
- Query label usage: hidden
- Prototype encoder: not included in this full run

| Holdout | Query | Train | Feature top-1 | Feature top-3 | Metadata top-1 | Metadata top-3 |
|---|---:|---:|---:|---:|---:|---:|
| 25.8kV GIS | 5,000 | 25,010 | 0.316200 | 0.333200 | 0.200000 | 0.200000 |
| ACSR-OC | 3,335 | 26,675 | 0.303448 | 0.342729 | 0.200000 | 0.200000 |
| CNCV-W | 3,335 | 26,675 | 0.294153 | 0.361019 | 0.200000 | 0.200000 |
| TFR-CV | 3,335 | 26,675 | 0.459670 | 0.476162 | 0.200000 | 0.200000 |
| 계기용 변압기 | 3,335 | 26,675 | 0.287856 | 0.342429 | 0.200000 | 0.200000 |
| 단상 유입변압기 | 3,335 | 26,675 | 0.190105 | 0.197301 | 0.200000 | 0.200000 |
| 전력용 유입변압기 | 3,335 | 26,675 | 0.507346 | 0.542129 | 0.200000 | 0.200000 |
| 22.9kV 배전반 | 2,500 | 27,510 | 0.974000 | 0.979200 | 0.400000 | 0.400000 |
| 7.2kV 배전반 | 2,500 | 27,510 | 0.767600 | 0.773200 | 0.400000 | 0.400000 |

The full report confirms that the strongest issue is not only `CNCV-W`; `단상 유입변압기` is below the metadata baseline under full holdout evaluation. The 배전반 groups are much easier for the current handcrafted feature space.

## Full Equipment Prototype Hard Split

- Report file: `prpd_similarity_retrieval/hard_split_report.full.prototype.md`
- Command: `python -m prpd_similarity_retrieval.cli evaluate-prototype-hard-split-report --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000 --format markdown --output prpd_similarity_retrieval\hard_split_report.full.prototype.md`
- Scope: all `equipment_name` holdout groups
- Query label usage: hidden
- Encoder: `prototype_centroid_encoder_v1`

| Holdout | Query | Train | Prototype top-1 | Prototype top-3 |
|---|---:|---:|---:|---:|
| 25.8kV GIS | 5,000 | 25,010 | 0.275200 | 0.332200 |
| ACSR-OC | 3,335 | 26,675 | 0.279160 | 0.394003 |
| CNCV-W | 3,335 | 26,675 | 0.327136 | 0.392504 |
| TFR-CV | 3,335 | 26,675 | 0.561319 | 0.586807 |
| 계기용 변압기 | 3,335 | 26,675 | 0.418291 | 0.435382 |
| 단상 유입변압기 | 3,335 | 26,675 | 0.350225 | 0.360420 |
| 전력용 유입변압기 | 3,335 | 26,675 | 0.600000 | 0.614393 |
| 22.9kV 배전반 | 2,500 | 27,510 | 0.988000 | 0.994800 |
| 7.2kV 배전반 | 2,500 | 27,510 | 0.722000 | 0.773200 |

Prototype hard split improves several weak feature groups, especially `단상 유입변압기`, `CNCV-W`, and `계기용 변압기`, but it does not solve every domain gap. `25.8kV GIS` remains near the feature baseline and `ACSR-OC` top-1 is slightly lower than the handcrafted feature baseline. This confirms that the prototype encoder is useful for evaluation plumbing but still not a substitute for a trained PRPD/time-series embedding model.

## Full Equipment Learned Projection Hard Split

- Report file: `prpd_similarity_retrieval/hard_split_report.full.learned.md`
- Command: `python -m prpd_similarity_retrieval.cli evaluate-learned-hard-split-report --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000 --format markdown --output prpd_similarity_retrieval\hard_split_report.full.learned.md`
- Scope: all `equipment_name` holdout groups
- Query label usage: hidden
- Encoder: `supervised_projection_encoder_v1`

| Holdout | Query | Train | Learned top-1 | Learned top-3 |
|---|---:|---:|---:|---:|
| 25.8kV GIS | 5,000 | 25,010 | 0.244000 | 0.262400 |
| ACSR-OC | 3,335 | 26,675 | 0.476162 | 0.519340 |
| CNCV-W | 3,335 | 26,675 | 0.320540 | 0.353523 |
| TFR-CV | 3,335 | 26,675 | 0.589805 | 0.614093 |
| 계기용 변압기 | 3,335 | 26,675 | 0.538831 | 0.594003 |
| 단상 유입변압기 | 3,335 | 26,675 | 0.482459 | 0.498351 |
| 전력용 유입변압기 | 3,335 | 26,675 | 0.730435 | 0.750825 |
| 22.9kV 배전반 | 2,500 | 27,510 | 0.885200 | 0.908800 |
| 7.2kV 배전반 | 2,500 | 27,510 | 0.883200 | 0.913600 |

Learned projection improves the weak feature/prototype groups `ACSR-OC`, `TFR-CV`, `계기용 변압기`, `단상 유입변압기`, `전력용 유입변압기`, and `7.2kV 배전반`. It is worse for `25.8kV GIS`, `CNCV-W`, and `22.9kV 배전반`, so production scoring should keep fallback/ensemble logic rather than assuming one embedding is uniformly better.

## Hard Split Failure Samples

- `CNCV-W`: `prpd_similarity_retrieval/hard_split_failures.cncv_w.sample.md`
- `단상 유입변압기`: `prpd_similarity_retrieval/hard_split_failures.single_oil_transformer.sample.md`
- Command pattern: `python -m prpd_similarity_retrieval.cli sample-hard-split-failures --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --holdout-value <value> --top-k 3 --max-failures 5 --format markdown --output <path>`

Observed failure pattern:

| Holdout | Query pattern | Retrieved pattern |
|---|---|---|
| CNCV-W | 노이즈 query | 정상/표면 방전/코로나 방전 ACSR-OC cases |
| 단상 유입변압기 | 노이즈 query | 전력용 유입변압기 정상 cases |

The sampler confirms that the current feature space is often matching equipment/domain texture more strongly than the intended noise-vs-discharge distinction for these held-out groups. The next review artifact should place query PRPD/time-series and retrieved PRPD/time-series side by side.

## Visual Review Artifacts

- `CNCV-W`: `prpd_similarity_retrieval/hard_split_review.cncv_w.sample.html`
- `단상 유입변압기`: `prpd_similarity_retrieval/hard_split_review.single_oil_transformer.sample.html`
- Command pattern: `python -m prpd_similarity_retrieval.cli sample-hard-split-failures --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --holdout-value <value> --top-k 3 --max-failures 5 --format html --output <path>`

Each HTML review page shows the holdout query beside its top-3 retrieved cases with:

- PRPD image
- downsampled time-series waveform SVG
- label and similarity score
- key equipment/sensor metadata
- human relevance controls: `유사`, `애매`, `비유사`
- review note field and browser-side CSV/JSON export

Generated HTML was checked for PRPD image tags and inline waveform polylines; no missing-media placeholders were emitted for the generated sample pages. Independent Playwright rendering confirmed each generated page has `5` failure sections, `20/20` loaded images, `20` waveform polylines, and `15` human review controls.

## Human Review Metrics

Human review exports can be evaluated with:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-human-reviews `
  --input path\to\hard_split_human_reviews.csv `
  --top-k 3 `
  --breakdown-field query_equipment_name `
  --format markdown `
  --output prpd_similarity_retrieval\human_review_metrics.md
```

Supported input formats:

- CSV from the HTML review export
- JSON from the HTML review export

Computed metrics:

- `accepted_neighbor_rate`: reviewed neighbor rows marked as accepted, default `similar`
- `human_relevance_at_k`: reviewed query ratio with at least one accepted neighbor within top-k
- `accepted_or_uncertain_at_k`: reviewed query ratio with at least one `similar` or `uncertain` neighbor within top-k
- arbitrary field breakdowns such as `query_equipment_name`, `neighbor_equipment_name`, `query_label_name`

No real reviewer export is committed yet. The evaluator was smoke-tested with a temporary CSV containing `similar`, `uncertain`, and `not_similar` rows.

## Label Breakdown

| Label | Evaluated | Feature top-1 | Feature top-3 | Metadata top-1 | Metadata top-3 |
|---|---:|---:|---:|---:|---:|
| 노이즈 | 6,002 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 보이드 방전 | 6,002 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 정상 | 6,002 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 코로나 방전 | 6,002 | 1.000000 | 1.000000 | 0.166611 | 0.166611 |
| 표면 방전 | 6,002 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Equipment Breakdown

| Equipment | Evaluated | Feature top-1 | Feature top-3 | Metadata top-1 | Metadata top-3 |
|---|---:|---:|---:|---:|---:|
| 25.8kV GIS | 5,000 | 1.000000 | 1.000000 | 0.200000 | 0.200000 |
| ACSR-OC | 3,335 | 1.000000 | 1.000000 | 0.200000 | 0.200000 |
| CNCV-W | 3,335 | 1.000000 | 1.000000 | 0.200000 | 0.200000 |
| TFR-CV | 3,335 | 1.000000 | 1.000000 | 0.200000 | 0.200000 |
| 계기용 변압기 | 3,335 | 1.000000 | 1.000000 | 0.200000 | 0.200000 |
| 단상 유입변압기 | 3,335 | 1.000000 | 1.000000 | 0.200000 | 0.200000 |
| 전력용 유입변압기 | 3,335 | 1.000000 | 1.000000 | 0.200000 | 0.200000 |
| 22.9kV 배전반 | 2,500 | 1.000000 | 1.000000 | 0.400000 | 0.400000 |
| 7.2kV 배전반 | 2,500 | 1.000000 | 1.000000 | 0.400000 | 0.400000 |

## Sensor Breakdown

| Sensor | Evaluated | Feature top-1 | Feature top-3 | Metadata top-1 | Metadata top-3 |
|---|---:|---:|---:|---:|---:|
| HFCT | 29,010 | 1.000000 | 1.000000 | 0.206894 | 0.206894 |
| UHF | 1,000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Interpretation

The feature retrieval baseline separates labels perfectly on this dataset under label-hidden retrieval, while the metadata baseline mostly returns cases sharing equipment or sensor metadata rather than PRPD/time-series pattern similarity.

The perfect feature score is useful as a baseline sanity check, but it also means the next phase should test harder splits and human-reviewed near-neighbor relevance so the evaluation does not overstate real-world generalization.
