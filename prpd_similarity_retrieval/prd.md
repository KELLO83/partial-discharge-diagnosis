# PRPD/Time-Series Similarity Retrieval PRD

## 1. Background

현재 유사 사례 기능은 `data/manifest.csv`의 과거 사례를 대상으로 라벨, 설비 메타데이터, 일부 시계열 지표를 가중합해 정렬한다. 이 방식은 빠르게 동작하지만, 사용자가 기대하는 "현재 점검의 PRPD 패턴/파형과 실제로 비슷했던 사례"를 직접 비교하지는 않는다.

현재 구현 위치:

- `service/backend/app/similar_cases.py`
- `service/backend/app/tools.py`
- `service/backend/app/agent_runtime.py`
- `service/frontend/src/App.tsx`

## 2. Problem

현재 유사도는 다음 한계가 있다.

- PRPD 이미지 또는 PRPD 패턴 자체의 형태 유사도를 계산하지 않는다.
- 시계열 파형은 `abs_p99` 같은 일부 proxy feature만 사용한다.
- `mock_dataset_case_retriever`, `pre_embedding_mock` 이름을 쓰지만 실제 embedding 검색은 아니다.
- 메타데이터가 유사하면 파형/패턴이 달라도 상위 사례로 노출될 수 있다.
- 사용자는 검색/필터가 아니라 현재 점검 결과와 유사한 과거 사례 자동 추천을 기대한다.

## 3. Goals

- 현재 점검 입력의 PRPD 패턴과 시계열 파형을 기반으로 과거 사례 top-k를 자동 추천한다.
- 각 추천 사례에 유사도 점수와 근거를 함께 제공한다.
- 기존 `similar_case_tool` 인터페이스를 최대한 유지해 프론트엔드와 워크플로우 변경 폭을 줄인다.
- 초기 버전은 학습 없는 도메인 feature 검색으로 구축하고, 이후 embedding 기반 검색으로 확장한다.

## 4. Non-Goals

- 일반 텍스트 embedding 또는 범용 CLIP만으로 최종 유사도 검색을 구성하지 않는다.
- 첫 버전에서 대규모 vector database 운영을 필수로 두지 않는다.
- 진단 모델 자체의 최종 판정 로직을 이번 범위에서 교체하지 않는다.
- 검색/필터 UI를 사용자 주요 흐름으로 되살리지 않는다.

## 5. Users

주 사용자는 현장 점검자와 검토 담당자다. 이들은 현재 점검 결과가 과거 어떤 사례와 비슷한지, 그리고 왜 비슷하다고 판단했는지를 보고 싶어 한다.

## 6. Functional Requirements

1. 진단 실행 후 `similar_case_tool`은 현재 점검 입력을 기준으로 과거 사례 top-k를 반환해야 한다.
2. 반환 사례는 `sample_id`, `label_name`, `similarity`, `reason`, `image_url`, 핵심 메타데이터를 포함해야 한다.
3. 유사도는 PRPD 패턴, 시계열 파형, 메타데이터, 모델 라벨 근거를 조합해야 한다.
4. 결과는 `trace`의 `similar_case_tool.summary.cases`에 기록되어야 한다.
5. 프론트엔드는 수동 검색 UI 없이 "현재 점검 유사 사례" 패널에 자동 결과를 보여줘야 한다.
6. 유사도 근거는 사용자가 이해할 수 있는 문장으로 제공되어야 한다.

## 7. Non-Functional Requirements

- 초기 top-k 조회는 일반적인 데이터셋 규모에서 1초 이내를 목표로 한다.
- feature/embedding 생성은 오프라인 배치로 수행 가능해야 한다.
- 신규 알고리즘은 기존 mock retriever와 병행 검증할 수 있어야 한다.
- 재현성을 위해 feature schema와 모델 버전을 저장해야 한다.
- 입력 파일 누락 시에도 graceful fallback을 제공해야 한다.

## 8. Proposed Approach

### Phase 1: Domain Feature Similarity

학습 없이 PRPD/시계열 도메인 feature를 추출해 cosine similarity 또는 weighted distance로 검색한다.

PRPD feature 후보:

- phase bin density
- discharge magnitude histogram
- phase localization score
- entropy
- quadrant ratio
- peak concentration
- pattern compactness

Time-series feature 후보:

- RMS
- standard deviation
- abs_p99
- pulse rate
- spectral energy
- peak density
- crest factor

Metadata feature 후보:

- sensor type
- insulator type
- clearance distance
- rated voltage
- equipment type

초기 점수 예시:

```text
similarity =
  0.45 * prpd_pattern_similarity
+ 0.35 * time_series_similarity
+ 0.10 * metadata_similarity
+ 0.10 * model_label_similarity
```

### Phase 2: Domain Embedding

Phase 1 결과를 baseline으로 삼고, 도메인 전용 encoder를 학습한다.

- PRPD image encoder: CNN 또는 small ViT 기반 supervised contrastive learning
- Time-series encoder: TS2Vec류 self-supervised representation 또는 기존 시계열 모델의 penultimate layer
- Multi-modal fusion: PRPD embedding, time-series embedding, metadata embedding을 late fusion

학습 목표:

- 같은 방전 유형과 유사 계측 조건의 사례는 가깝게 둔다.
- 다른 방전 유형 또는 명확히 다른 PRPD/파형 패턴은 멀게 둔다.
- 라벨만 같고 패턴이 다른 경우를 무조건 가깝게 만들지 않는다.

### Phase 3: Retrieval Service

사례별 feature/embedding을 오프라인으로 생성하고 검색 시 로드한다.

저장 후보:

- `data/case_features.parquet`
- `data/case_embeddings.npz`
- `data/case_embedding_manifest.json`

검색 방식:

- 초기: in-memory numpy top-k
- 확장: FAISS 또는 pgvector

## 9. Backend Changes

신규 모듈 후보:

- `service/backend/app/case_features.py`
- `service/backend/app/case_embedding_store.py`
- `service/backend/app/similarity_scoring.py`

기존 변경:

- `similar_cases.py`의 `_case_score`를 feature 기반 scoring으로 대체
- `DatasetCase`에 feature/embedding path 또는 derived feature 추가
- `SimilarCase.reason` 생성 로직을 상세 근거 중심으로 개선

신규 스크립트 후보:

- `service/scripts/build_case_features.py`
- `service/scripts/build_case_embeddings.py`
- `service/scripts/evaluate_case_retrieval.py`

## 10. Frontend Changes

현재 방향을 유지한다.

- 검색/필터 UI는 제공하지 않는다.
- `현재 점검 유사 사례` 패널은 trace의 자동 추천 결과만 표시한다.
- 카드에는 유사도, 방전 유형, 설비/센서/절연 정보, 유사 근거를 표시한다.
- 향후 근거를 더 세분화하면 카드에 `PRPD 유사`, `파형 유사`, `메타데이터 일치` 같은 badge를 추가한다.

## 11. Evaluation

오프라인 평가 지표:

- top-1 label match rate
- top-3 label match rate
- same equipment/sensor match rate
- human-reviewed relevance score
- false-neighbor rate

검증 데이터:

- 동일 라벨 내 다른 패턴 사례
- 다른 라벨이지만 메타데이터가 같은 사례
- 메타데이터 부족 사례
- PRPD만 있는 사례
- 시계열만 있는 사례
- PRPD와 시계열이 모두 있는 hybrid 사례

Acceptance 기준:

- 기존 metadata-weighted baseline보다 top-3 relevance가 높아야 한다.
- 동일 라벨만으로 무조건 상위 노출되는 문제를 줄여야 한다.
- 유사도 근거가 사람이 납득 가능한 수준이어야 한다.

## 12. Risks

- 데이터셋 규모가 작으면 embedding 학습이 과적합될 수 있다.
- PRPD 이미지가 시각화 방식에 따라 달라지면 image encoder가 표시 스타일을 학습할 수 있다.
- 시계열 샘플링 조건이 일정하지 않으면 직접 비교가 불안정할 수 있다.
- 라벨 품질이 낮으면 supervised contrastive 학습이 오히려 나쁜 이웃을 만들 수 있다.

Mitigation:

- Phase 1 feature baseline을 먼저 만든다.
- 라벨 기반 평가와 사람 검토 평가를 분리한다.
- augmentation과 normalization 규칙을 명확히 고정한다.
- 모델 버전과 feature schema를 trace에 기록한다.

## 13. Milestones

### M1: Feature Baseline

- PRPD/시계열 feature extractor 구현
- feature cache 생성 스크립트 구현
- 기존 `similar_case_tool`을 feature similarity 기반으로 교체
- top-k 결과와 reason 표시 검증

### M2: Evaluation Harness

- retrieval 평가 스크립트 구현
- baseline과 feature similarity 비교 리포트 생성
- 실패 사례 샘플링

### M3: Embedding Prototype

- PRPD encoder prototype
- time-series encoder prototype
- multi-modal late fusion score 구현
- M1 baseline 대비 성능 비교

### M4: Productionization

- embedding/index versioning
- index rebuild command
- runtime fallback
- trace metadata 기록

## 14. Open Questions

- PRPD 원본은 이미지뿐인지, phase/magnitude raw matrix도 확보 가능한지?
- 시계열 CSV의 sampling rate와 길이는 모든 사례에서 일정한지?
- 현재 데이터셋 라벨은 사람이 검수한 ground truth인지, 모델 생성 라벨인지?
- 유사 사례의 목적은 진단 보조인지, 최종 판정 근거 가중치에 포함되는 것인지?
- 운영 환경에서 FAISS 같은 native dependency를 허용할 수 있는지?

## 15. Implementation Status

M1 baseline 구축을 `prpd_similarity_retrieval/` 폴더 안에서 시작했다.

현재 산출물:

- `features.py`: PRPD 이미지와 시계열 CSV feature 추출
- `feature_cache.py`: 전체 빌드 재시작을 위한 JSONL feature cache
- `retrieval.py`: feature index 저장/로드와 top-k 유사 사례 검색
- `compact_index.py`: 전체 데이터셋용 `.npz` 압축 행렬 index 저장/로드와 vectorized top-k 검색
- `batch_evaluation.py`: full index 평가용 batch evaluator
- `hard_split_evaluation.py`: 장비/센서/전압 등 group holdout hard split evaluator
- `backend_adapter.py`: backend `SimilarCaseRetrievalAdapter` 호환 adapter
- `models.py`: feature/search result 데이터 구조
- `cli.py`: `.npz`/`.json` index 생성, 재시작 가능한 cache build, sample query, 외부 파일 query, leave-one-out 평가, hard split 평가, metadata baseline 비교 CLI
- `prototype_encoder.py`: PRPD image/time-series prototype embedding encoder와 train-only transform state
- `learned_encoder.py`: CNN/TS2Vec 출력으로 교체 가능한 supervised projection embedding 실험 골격
- `review_artifact.py`: hard split 실패 사례 HTML review artifact 생성
- `human_review.py`: HTML review export CSV/JSON의 human relevance metric 계산
- `tests/test_compact_index.py`: compact index가 기존 리스트 기반 검색과 같은 순위/점수를 내는지 검증
- `tests/test_feature_cache.py`, `tests/test_cli_cache_build.py`: cache 저장/로드와 CLI cache hit 동작 검증
- `tests/test_hard_split_evaluation.py`: holdout group을 candidate에서 제외하는지 검증
- `tests/test_learned_encoder.py`: learned projection embedding index 저장/검색/평가 검증
- `tests/test_review_artifact.py`: HTML review가 query/neighbor와 시계열 SVG를 렌더링하는지 검증
- `tests/test_human_review.py`: CSV/JSON review export metric 계산 검증
- `README.md`: 실행 방법과 다음 작업 정리
- `hard_split_report.sample.md`: 장비 holdout 9개 제한 hard split 리포트
- `hard_split_report.full.feature.md`: 장비 holdout 9개 전체 feature/metadata hard split 리포트
- `hard_split_report.full.prototype.md`: 장비 holdout 9개 전체 prototype encoder hard split 리포트
- `hard_split_failures.cncv_w.sample.md`, `hard_split_failures.single_oil_transformer.sample.md`: 낮은 성능 holdout의 top-k 실패 사례 샘플
- `hard_split_review.cncv_w.sample.html`, `hard_split_review.single_oil_transformer.sample.html`: 실패 query와 retrieved case의 PRPD/시계열 side-by-side review

현재 CLI 검증:

```powershell
python -m compileall prpd_similarity_retrieval
python -m pytest prpd_similarity_retrieval\tests -q
python -m prpd_similarity_retrieval.cli build-index --limit 100 --workers 4 --progress-every 25 --cache prpd_similarity_retrieval\case_feature_cache.sample.jsonl --output prpd_similarity_retrieval\case_feature_index.sample.npz
python -m prpd_similarity_retrieval.cli build-index --per-label-limit 10 --workers 4 --cache prpd_similarity_retrieval\case_feature_cache.stratified.jsonl --output prpd_similarity_retrieval\case_feature_index.stratified.npz
python -m prpd_similarity_retrieval.cli build-index --workers 4 --progress-every 1000 --cache prpd_similarity_retrieval\case_feature_cache.jsonl --output prpd_similarity_retrieval\case_feature_index.npz
python -m prpd_similarity_retrieval.cli query-sample --index prpd_similarity_retrieval\case_feature_index.sample.npz --sample-id "노이즈_고체_ACSR-OC_230910_195222_HFCT_1000" --top-k 3
python -m prpd_similarity_retrieval.cli evaluate-index --index prpd_similarity_retrieval\case_feature_index.sample.npz --top-k 3
python -m prpd_similarity_retrieval.cli compare-baseline --index prpd_similarity_retrieval\case_feature_index.stratified.npz --top-k 3
python -m prpd_similarity_retrieval.cli compare-baseline --index prpd_similarity_retrieval\case_feature_index.npz --limit 50 --top-k 3
python -m prpd_similarity_retrieval.cli compare-baseline --index prpd_similarity_retrieval\case_feature_index.npz --limit 500 --top-k 3
python -m prpd_similarity_retrieval.cli compare-baseline --index prpd_similarity_retrieval\case_feature_index.npz --top-k 3 --batch-size 256 --progress-every 5000
python -m prpd_similarity_retrieval.cli build-prototype-index --feature-index prpd_similarity_retrieval\case_feature_index.npz --output prpd_similarity_retrieval\case_embedding_index.prototype.npz
python -m prpd_similarity_retrieval.cli evaluate-prototype-index --index prpd_similarity_retrieval\case_embedding_index.prototype.npz --top-k 3 --batch-size 256
python -m prpd_similarity_retrieval.cli evaluate-hard-split --index prpd_similarity_retrieval\case_feature_index.npz --limit 50 --top-k 3 --batch-size 32 --include-prototype
python -m prpd_similarity_retrieval.cli evaluate-hard-split-report --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --limit-per-holdout 30 --top-k 3 --batch-size 32 --include-prototype --format markdown --output prpd_similarity_retrieval\hard_split_report.sample.md
python -m prpd_similarity_retrieval.cli evaluate-hard-split-report --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000 --format markdown --output prpd_similarity_retrieval\hard_split_report.full.feature.md
python -m prpd_similarity_retrieval.cli evaluate-prototype-hard-split-report --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000 --format markdown --output prpd_similarity_retrieval\hard_split_report.full.prototype.md
python -m prpd_similarity_retrieval.cli sample-hard-split-failures --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --holdout-value CNCV-W --top-k 3 --max-failures 5 --format markdown --output prpd_similarity_retrieval\hard_split_failures.cncv_w.sample.md
python -m prpd_similarity_retrieval.cli sample-hard-split-failures --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --holdout-value "단상 유입변압기" --top-k 3 --max-failures 5 --format markdown --output prpd_similarity_retrieval\hard_split_failures.single_oil_transformer.sample.md
python -m prpd_similarity_retrieval.cli sample-hard-split-failures --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --holdout-value CNCV-W --top-k 3 --max-failures 5 --format html --output prpd_similarity_retrieval\hard_split_review.cncv_w.sample.html
python -m prpd_similarity_retrieval.cli sample-hard-split-failures --index prpd_similarity_retrieval\case_feature_index.npz --split-field equipment_name --holdout-value "단상 유입변압기" --top-k 3 --max-failures 5 --format html --output prpd_similarity_retrieval\hard_split_review.single_oil_transformer.sample.html
python -m prpd_similarity_retrieval.cli evaluate-human-reviews --input path\to\hard_split_human_reviews.csv --top-k 3 --breakdown-field query_equipment_name --format markdown --output prpd_similarity_retrieval\human_review_metrics.md
python -m pytest service\backend\tests\test_diagnose_api.py -q
```

라벨별 10건 stratified sample 기준, query label을 숨긴 비교 결과:

- feature retrieval top-1 label match: `1.0`
- metadata baseline top-1 label match: `0.2`
- top-1 delta: `+0.8`

전체 `manifest.csv` 30,010건 index 생성 결과:

- `case_feature_index.npz`: 30,010건, PRPD image feature 30,010건, time-series feature 30,010건
- `case_feature_cache.jsonl`: 중단 후 재시작 가능한 feature cache
- full index 단일 query 정상 동작 확인
- full index `--limit 50` 평가 기준, query label 숨김:
  - feature retrieval top-1 label match: `1.0`
  - metadata baseline top-1 label match: `0.0`
  - top-1 delta: `+1.0`
- full index `--limit 500` 평가 기준, query label 숨김:
  - feature retrieval top-1 label match: `1.0`
  - metadata baseline top-1 label match: `0.0`
  - top-1 delta: `+1.0`
  - elapsed: 약 `66.2s`
- full index 전체 30,010건 평가 기준, query label 숨김:
  - feature retrieval top-1 label match: `1.0`
  - feature retrieval top-3 label match: `1.0`
  - metadata baseline top-1 label match: `0.233322`
  - metadata baseline top-3 label match: `0.233322`
  - top-1/top-3 delta: `+0.766678`
  - elapsed: 약 `500.6s`
- full index label/equipment/sensor breakdown 리포트:
  - `evaluation_report.md`
  - elapsed: 약 `577.2s`
  - feature retrieval은 모든 label/equipment/sensor breakdown에서 top-1/top-3 `1.0`
  - metadata baseline은 전체 top-1/top-3 `0.233322`
- prototype encoder index:
  - `case_embedding_index.prototype.npz`
  - encoder: deterministic random projection + label-centroid calibration
  - 전체 30,010건 평가 기준 top-1/top-3 label match: `1.0`
  - elapsed: 약 `20.9s`
  - 이 수치는 encoder 검색 파이프라인 검증용이며, label-supervised centroid를 쓰므로 real-world generalization 평가는 hard split이 필요하다.
- hard split 제한 검증:
  - command: `evaluate-hard-split --limit 50 --top-k 3 --batch-size 32 --include-prototype`
  - split field: `equipment_name`
  - holdout group: `25.8kV GIS`
  - train/query: `25,010` / `50`
  - feature retrieval top-1/top-3 label match: `0.32` / `0.32`
  - prototype encoder top-1/top-3 label match: `0.06` / `0.32`
  - metadata baseline top-1/top-3 label match: `0.0` / `0.0`
  - 이 평가는 holdout group을 candidate에서 제외하므로 leave-one-out보다 새로운 장비/조건 일반화에 더 가깝다.
- 장비 holdout 제한 리포트:
  - `hard_split_report.sample.md`
  - split field: `equipment_name`
  - holdout groups: `9`
  - per-holdout query limit: `30`
  - `CNCV-W`, `단상 유입변압기`는 제한 리포트에서 feature top-1/top-3 `0.0`으로 나와 실패 사례 샘플링 우선순위가 높다.
- 장비 holdout 전체 feature/metadata 리포트:
  - `hard_split_report.full.feature.md`
  - split field: `equipment_name`
  - holdout groups: `9`
  - per-holdout query limit: none
  - weakest feature top-3: `단상 유입변압기` `0.197301`, `25.8kV GIS` `0.333200`, `계기용 변압기` `0.342429`
  - strongest feature top-3: `22.9kV 배전반` `0.979200`, `7.2kV 배전반` `0.773200`, `전력용 유입변압기` `0.542129`
  - prototype encoder는 별도 prototype-only full run으로 평가한다.
- 장비 holdout 전체 prototype encoder 리포트:
  - `hard_split_report.full.prototype.md`
  - split field: `equipment_name`
  - holdout groups: `9`
  - per-holdout query limit: none
  - weakest prototype top-3: `25.8kV GIS` `0.332200`, `단상 유입변압기` `0.360420`, `CNCV-W` `0.392504`
  - strongest prototype top-3: `22.9kV 배전반` `0.994800`, `7.2kV 배전반` `0.773200`, `전력용 유입변압기` `0.614393`
  - prototype은 `단상 유입변압기`, `CNCV-W`, `계기용 변압기`에서 feature baseline보다 개선됐지만 `25.8kV GIS`, `ACSR-OC`는 개선되지 않았다.
- hard split 실패 사례 샘플:
  - `hard_split_failures.cncv_w.sample.md`
  - `hard_split_failures.single_oil_transformer.sample.md`
  - `CNCV-W` 노이즈 query가 정상/표면 방전/코로나 방전 ACSR-OC 사례로 잘못 끌리는 패턴 확인
  - `단상 유입변압기` 노이즈 query가 전력용 유입변압기 정상 사례로 잘못 끌리는 패턴 확인
- hard split visual review:
  - `hard_split_review.cncv_w.sample.html`
  - `hard_split_review.single_oil_transformer.sample.html`
  - 각 실패 query와 top-3 neighbor의 PRPD 이미지, 시계열 SVG, label, similarity score를 나란히 표시
  - neighbor별 `유사/애매/비유사` 판정, review note, CSV/JSON export 지원
  - export에 query/neighbor 장비명과 센서 타입 포함
  - 생성 HTML에서 이미지 태그와 waveform polyline이 존재하고 missing-media placeholder가 없는 것을 확인
  - Playwright 렌더링 확인: 각 HTML `5` failure sections, `20/20` loaded images, `20` waveforms, `15` review controls
- human review metric evaluator:
  - command: `evaluate-human-reviews`
  - CSV/JSON input 지원
  - metrics: `accepted_neighbor_rate`, `human_relevance_at_k`, `accepted_or_uncertain_at_k`
  - `query_equipment_name`, `neighbor_equipment_name`, `query_label_name` 등 임의 field breakdown 지원
  - 임시 CSV smoke test에서 expected metric 출력 확인
- learned projection encoder 실험 골격:
  - `learned_encoder.py`
  - train split에서 feature standardization, PCA projection, label centroid affinity를 fit한다.
  - 목표는 실제 CNN/TS2Vec encoder 출력이 준비되기 전, embedding index/search/evaluation 경로를 고정하는 것이다.
  - 저장/로드, sample search, leave-one-out 평가, hard split report 함수의 최소 동작을 테스트로 검증했다.
  - 아직 full dataset learned report와 운영 CLI 명령은 생성하지 않았다.

현재 backend 연결:

- `SimilarCaseToolInput`이 원본 `image_path`, `timeseries_path`를 전달한다.
- `service/backend/app/tools.py`의 `similar_case_adapter`가 `FeatureSimilarityCaseRetrievalAdapter`를 사용한다.
- feature index가 있으면 PRPD/시계열 feature 검색을 사용하고, 없으면 기존 metadata-weighted retriever로 fallback한다.
- backend index 탐색 순서는 `.npz` 운영 index, `.json` 호환 index, sample index 순서다.
- index build는 `--cache`를 지정하면 이미 추출된 feature를 재사용해 중단 후 재시작할 수 있다.

다음 구현 단계:

1. 실제 reviewer export 파일을 모아 `human_review_metrics.md`를 생성한다.
2. `learned_encoder.py`를 CLI/full hard split report로 노출할지, 바로 CNN/TS2Vec encoder 출력으로 교체할지 결정한다.
3. CNN/TS2Vec 계열 encoder 학습 후보를 붙인다.
4. 현재 진단 화면의 유사 사례 섹션에 hard split 리포트 기준의 신뢰도/주의 문구를 연결한다.
