# PRPD/Time-Series Similarity Retrieval PRD

Last updated: 2026-06-15

## 1. Current Status

`prpd_similarity_retrieval/`는 현재 점검의 PRPD 이미지와 시계열 CSV를 과거 데이터셋 사례와 비교해 자동으로 Top-K 유사사례를 찾는 기능이다.

현재 기준 1차 기능 개발은 완료 상태다.

- 운영 기본 retriever: `domain_feature_case_retriever`
- 운영 검색 기준: PRPD 이미지 feature + 시계열 feature
- 운영 제외 기준: 메타데이터와 라벨은 유사사례 ranking 점수에 사용하지 않음
- 기본 Top-K: backend adapter 기준 `5`
- 기본 점수 가중치: PRPD `0.55`, 시계열 `0.45`
- 프론트 노출: 진단 실행 후 `유사 사례` 화면과 리포트/근거 영역에서 자동 표시

모델 개발 관점에서는 아직 최종 embedding 모델이 아니다. 현재 운영 경로는 빠르고 가성비 좋은 도메인 feature baseline이며, `prototype_encoder.py`, `learned_encoder.py`는 향후 CNN/TS2Vec 등 실제 neural encoder를 붙이기 전 검색/평가 파이프라인을 검증하기 위한 실험 경로다.

## 2. Product Goal

사용자가 진단을 실행했을 때 "검색"을 직접 하는 것이 아니라, 현재 점검 상태와 PRPD/파형 형태가 비슷했던 과거 사례를 자동으로 보여주는 것이 목표다.

사용자가 보고 싶은 정보:

- 현재 점검과 비슷했던 과거 사례는 무엇인지
- PRPD와 시계열 중 어느 쪽이 더 닮았는지
- 유사도 점수와 추천 이유가 무엇인지
- 과거 사례의 PRPD 이미지와 시계열 원본을 현재 점검과 비교할 수 있는지

## 3. Completed User-Facing Features

### 3.1 Automatic Similar Case Recommendation

진단 실행 시 backend workflow가 `similar_case_tool`을 실행하고, 결과를 trace에 저장한다.

저장 위치:

```text
trace.events[name="similar_case_tool"].summary.cases
```

프론트는 이 trace 결과만 읽어서 `현재 점검 유사 사례` 영역에 표시한다. 사용자가 별도 검색어를 입력하거나 필터를 조작하는 흐름은 현재 1차 목표에서 제외했다.

### 3.2 Top 5 Similar Case Cards

유사사례 카드는 기본적으로 최대 5개를 보여준다.

카드에서 보여주는 항목:

- 순위
- 방전 유형
- 전체 유사도
- 추천 이유
- PRPD/파형 중 어떤 축이 더 유사한지 요약
- PRPD component score
- 시계열 component score
- PRPD 이미지 preview
- 상세 열기 버튼

예시 문구:

```text
#1 PRPD/파형 모두 유사
PRPD 유사 · 파형 보통

#2 PRPD 중심 유사
PRPD 유사 · 파형 상대 약함
```

### 3.3 Similar Case Detail Modal

카드에서 상세를 열면 현재 점검과 과거 사례를 더 자세히 비교한다.

상세 화면 기능:

- 현재 PRPD 이미지와 유사사례 PRPD 이미지 side-by-side 표시
- 현재 시계열 CSV와 유사사례 시계열 CSV waveform preview 표시
- 유사도 component bar 표시
- 방전유형, PRPD 이미지 여부, 시계열 CSV 여부, 신호 피크, 전체 유사도 비교
- 과거 사례 원본 asset은 backend의 `/dataset/cases/{sample_id}/image`, `/dataset/cases/{sample_id}/timeseries`에서 제공

### 3.4 Metadata Exclusion

데이터셋의 메타데이터는 대부분 동일하거나 반복되는 값이 많기 때문에 현재 운영 유사도에는 반영하지 않는다.

현재 ranking에 쓰는 값:

- PRPD image feature cosine similarity
- time-series feature cosine similarity

현재 ranking에 쓰지 않는 값:

- equipment_name
- equipment_rated_voltage
- equipment_rated_current
- equipment_type
- insulator_type
- insulator_name
- sensor_type
- clearance_distance
- label_id / label_name

메타데이터는 asset 표시나 상세 정보 표시를 위한 보조 정보로만 남긴다.

## 4. Runtime Behavior

### 4.1 Backend Adapter

운영 adapter:

```text
prpd_similarity_retrieval/backend_adapter.py
FeatureSimilarityCaseRetrievalAdapter
```

adapter 이름:

```text
domain_feature_case_retriever
```

동작:

1. 현재 점검 입력의 `image_path`, `timeseries_path`를 받는다.
2. `extract_case_features()`로 현재 점검 feature를 만든다.
3. feature index를 로드한다.
4. PRPD/시계열 cosine similarity로 Top 5를 검색한다.
5. backend `SimilarCase` schema로 변환한다.
6. `feature_component_prpd`, `feature_component_timeseries`를 metadata에 실어 frontend로 보낸다.

### 4.2 Index Loading Order

backend가 feature index를 찾는 순서:

1. `PRPD_CASE_FEATURE_INDEX` 환경 변수
2. `prpd_similarity_retrieval/case_feature_index.npz`
3. `prpd_similarity_retrieval/case_feature_index.json`
4. `prpd_similarity_retrieval/case_feature_index.sample.npz`
5. `prpd_similarity_retrieval/case_feature_index.sample.json`

index가 없으면 메타데이터 fallback 검색을 수행하지 않고 빈 유사사례 결과를 반환한다.

```text
retriever_name = prpd_timeseries_case_retriever_unavailable
retriever_version = no_feature_index
cases = []
```

이렇게 한 이유는 메타데이터가 대부분 유사해서 실제 PRPD/파형 유사사례로 오해될 가능성이 크기 때문이다.

### 4.3 Returned SimilarCase Payload

backend가 frontend에 넘기는 주요 값:

```text
sample_id
label_id
label_name
similarity
reason
image_url
timeseries_url
metadata.feature_component_prpd
metadata.feature_component_timeseries
metadata.retriever_mode = domain_feature_retriever
```

## 5. Feature Extraction

구현 파일:

```text
prpd_similarity_retrieval/features.py
```

### 5.1 PRPD Image Feature

입력:

```text
*.png PRPD image
```

처리:

1. 이미지를 grayscale로 변환
2. `32 x 32`로 resize
3. darkness map 생성
4. thumbnail vector 생성
5. horizontal profile `16 bins`
6. vertical profile `16 bins`
7. quadrant energy
8. active ratio, entropy, compactness, mean, std descriptor
9. 전체 vector 정규화

목적:

- PRPD 점 분포
- phase 방향 분포
- magnitude 방향 분포
- 특정 사분면 집중도
- 패턴 밀집도와 entropy

### 5.2 Time-Series Feature

입력:

```text
*.csv time-series signal
```

처리:

1. 숫자 CSV 로드
2. 평균 제거
3. 표준편차, RMS, max abs 계산
4. abs value histogram `12 bins`
5. FFT spectrum 계산
6. high-frequency ratio
7. spectral centroid
8. pulse rate
9. 전체 vector 정규화

목적:

- 파형 진폭 분포
- 피크성
- 고주파 성분
- pulse 밀도
- 파형 형태의 거친 유사성

## 6. Similarity Algorithm

구현 파일:

```text
prpd_similarity_retrieval/retrieval.py
prpd_similarity_retrieval/compact_index.py
```

운영 점수:

```text
similarity =
  0.55 * PRPD image feature cosine similarity
+ 0.45 * time-series feature cosine similarity
```

누락된 입력은 점수 계산에서 제외하고, 남은 component weight로 재정규화한다.

예:

- PRPD와 시계열이 모두 있으면 `0.55 / 0.45` 가중합
- PRPD만 있으면 PRPD 점수만 사용
- 시계열만 있으면 시계열 점수만 사용
- 둘 다 없으면 유사도 `0`

추천 이유:

- PRPD component가 높으면 `PRPD 패턴 유사`
- 시계열 component가 높으면 `시계열 파형 유사`
- 둘 다 명확히 높지 않으면 `PRPD/시계열 feature 기준 근접 사례`

## 7. Index and Storage

### 7.1 Compact NPZ Index

운영 기본 index:

```text
prpd_similarity_retrieval/case_feature_index.npz
```

현재 전체 index 상태:

```text
case count: 30,010
PRPD image feature: 30,010
time-series feature: 30,010
schema version: domain_feature_v1
```

`compact_index.py`는 전체 vector를 numpy matrix로 저장해 검색 시 vectorized cosine score를 계산한다.

저장 내용:

- schema_version
- cases_json
- image_vectors
- image_available
- image_norms
- timeseries_vectors
- timeseries_available
- timeseries_norms

### 7.2 JSON Index

호환용 사람이 읽을 수 있는 index:

```text
case_feature_index.json
case_feature_index.sample.json
```

대규모 운영에는 `.npz`가 기본이다.

### 7.3 Feature Cache

재시작 가능한 feature 추출 cache:

```text
case_feature_cache.jsonl
case_feature_cache.sample.jsonl
case_feature_cache.stratified.jsonl
```

빌드 중간에 끊겨도 이미 추출한 sample은 재사용하고 누락분만 이어서 계산한다.

## 8. CLI Features

구현 파일:

```text
prpd_similarity_retrieval/cli.py
```

### 8.1 Build

전체 또는 샘플 feature index 생성:

```powershell
python -m prpd_similarity_retrieval.cli build-index `
  --workers 4 `
  --progress-every 1000 `
  --cache prpd_similarity_retrieval\case_feature_cache.jsonl `
  --output prpd_similarity_retrieval\case_feature_index.npz
```

지원 옵션:

- `--limit`
- `--per-label-limit`
- `--workers`
- `--cache`
- `--progress-every`
- `.npz` / `.json` output

### 8.2 Query

index 안에 있는 sample 기준 검색:

```powershell
python -m prpd_similarity_retrieval.cli query-sample `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --sample-id "<sample_id>" `
  --top-k 5
```

외부 현재 점검 파일 기준 검색:

```powershell
python -m prpd_similarity_retrieval.cli query-files `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --image-path path\to\current_prpd.png `
  --timeseries-path path\to\current_signal.csv `
  --top-k 5
```

### 8.3 Baseline Evaluation

leave-one-out label match 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-index `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --top-k 3 `
  --batch-size 256
```

feature retrieval과 metadata-only baseline 비교:

```powershell
python -m prpd_similarity_retrieval.cli compare-baseline `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --top-k 3 `
  --batch-size 256
```

### 8.4 Hard Split Evaluation

장비/센서/전압 등 group holdout 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-hard-split-report `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --top-k 3 `
  --batch-size 256 `
  --format markdown `
  --output prpd_similarity_retrieval\hard_split_report.full.feature.md
```

목적:

- 같은 장비군이 candidate에 남아 있는 쉬운 leave-one-out 성능을 과신하지 않기 위함
- 새로운 장비/조건에서 유사사례가 얼마나 버티는지 확인

### 8.5 Failure Sampling and Review

hard split 실패 사례 샘플링:

```powershell
python -m prpd_similarity_retrieval.cli sample-hard-split-failures `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --holdout-value CNCV-W `
  --top-k 3 `
  --max-failures 5 `
  --format html `
  --output prpd_similarity_retrieval\hard_split_review.cncv_w.sample.html
```

HTML review 기능:

- query PRPD image 표시
- retrieved case PRPD image 표시
- query/retrieved waveform SVG 표시
- label, similarity score 표시
- `유사`, `애매`, `비유사` 판정
- reviewer note
- CSV/JSON export

### 8.6 Human Review Metrics

사람 검토 export 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-human-reviews `
  --input path\to\hard_split_human_reviews.csv `
  --top-k 3 `
  --breakdown-field query_equipment_name `
  --format markdown `
  --output prpd_similarity_retrieval\human_review_metrics.md
```

지원 metric:

- accepted_neighbor_rate
- human_relevance_at_k
- accepted_or_uncertain_at_k
- breakdown by query/neighbor metadata field

## 9. Experimental Encoder Features

운영 기본은 feature retrieval이다. 아래 두 encoder는 최종 모델이 아니라 embedding 검색 파이프라인과 평가 체계를 검증하기 위한 실험 경로다.

### 9.1 Prototype Encoder

구현 파일:

```text
prototype_encoder.py
```

역할:

- handcrafted feature vector를 deterministic random projection으로 embedding화
- label centroid calibration 적용
- embedding index 저장/검색/평가 경로 검증

명령:

```powershell
python -m prpd_similarity_retrieval.cli build-prototype-index `
  --feature-index prpd_similarity_retrieval\case_feature_index.npz `
  --output prpd_similarity_retrieval\case_embedding_index.prototype.npz
```

### 9.2 Learned Projection Encoder

구현 파일:

```text
learned_encoder.py
```

역할:

- feature standardization
- PCA projection
- label centroid affinity
- encoder state 저장/로드
- 향후 CNN/TS2Vec encoder를 붙이기 전 runtime embedding 경로 검증

명령:

```powershell
python -m prpd_similarity_retrieval.cli build-learned-index `
  --feature-index prpd_similarity_retrieval\case_feature_index.npz `
  --output prpd_similarity_retrieval\case_embedding_index.learned.npz
```

주의:

- 현재 backend 운영 경로는 learned index를 우선 사용하지 않는다.
- learned/prototype 결과는 모델 개발 실험 및 비교 기준으로 유지한다.
- 실제 neural encoder가 들어오면 동일 hard split 평가로 feature baseline과 비교해야 한다.

## 10. Evaluation Results Summary

상세 리포트:

```text
evaluation_report.md
hard_split_report.full.feature.md
hard_split_report.full.prototype.md
hard_split_report.full.learned.md
```

현재 주요 결과:

- full index: 30,010건
- feature retrieval leave-one-out top-1/top-3 label match: `1.000000 / 1.000000`
- metadata baseline leave-one-out top-1/top-3 label match: `0.233322 / 0.233322`
- prototype leave-one-out top-1/top-3 label match: `1.000000 / 1.000000`
- learned projection leave-one-out top-1/top-3 label match: `0.998834 / 0.999367`

해석:

- leave-one-out 성능은 같은 데이터셋 안의 가까운 사례를 찾는 쉬운 조건이라 운영 품질의 최종 근거로 보기는 어렵다.
- 더 중요한 평가는 equipment hard split이다.

equipment hard split feature 결과:

- 강한 그룹: `22.9kV 배전반`, `7.2kV 배전반`, `전력용 유입변압기`
- 약한 그룹: `단상 유입변압기`, `25.8kV GIS`, `계기용 변압기`

known failure:

- `CNCV-W` noise query가 ACSR-OC 정상/표면/코로나 사례로 끌리는 경우
- `단상 유입변압기` noise query가 전력용 유입변압기 정상 사례로 끌리는 경우

## 11. Backend and Frontend Integration

### 11.1 Backend

연결 위치:

```text
service/backend/app/tools.py
service/backend/app/application/agent_runtime.py
service/backend/app/application/workflow.py
prpd_similarity_retrieval/backend_adapter.py
```

완료된 연결:

- `SimilarCaseToolInput`이 현재 점검 원본 `image_path`, `timeseries_path`를 전달
- `FeatureSimilarityCaseRetrievalAdapter`가 feature index 검색 수행
- trace에 `similar_case_tool` 이벤트 저장
- dataset case asset endpoint로 PRPD/CSV 제공
- backend diagnose API test에서 `domain_feature_case_retriever` 확인

### 11.2 Frontend

연결 위치:

```text
service/frontend/src/App.tsx
service/frontend/src/styles.css
```

완료된 화면 기능:

- 현재 점검 유사 사례 자동 표시
- 검색/필터 UI 제거
- PRPD/시계열 component score 표시
- Top 순위 이유 한 줄 표시
- PRPD/파형 중 어느 쪽이 더 닮았는지 요약
- 상세 모달에서 PRPD 이미지 비교
- 상세 모달에서 waveform preview 비교
- 메타데이터 중심 비교 제거

## 12. Module Inventory

| File | Role |
|---|---|
| `features.py` | manifest 로드, PRPD 이미지 feature 추출, 시계열 CSV feature 추출 |
| `models.py` | `CaseRecord`, `CaseFeatures`, `SearchResult` 데이터 구조 |
| `retrieval.py` | list 기반 feature 검색, metadata baseline 검색, JSON index 저장/로드 |
| `compact_index.py` | `.npz` 압축 index 저장/로드, vectorized top-k 검색 |
| `feature_cache.py` | JSONL feature cache 저장/로드 |
| `backend_adapter.py` | backend `SimilarCaseRetrievalAdapter` 구현 |
| `cli.py` | build/query/evaluate/review CLI |
| `batch_evaluation.py` | compact index batch 평가 |
| `hard_split_evaluation.py` | group holdout hard split 평가와 report 생성 |
| `review_artifact.py` | 실패 사례 HTML review artifact 생성 |
| `human_review.py` | 사람 검토 CSV/JSON metric 계산 |
| `prototype_encoder.py` | prototype embedding index/search/evaluation |
| `learned_encoder.py` | supervised projection embedding index/search/evaluation |
| `evaluation_report.md` | 주요 평가 결과 요약 |
| `HANDOFF.md` | 다음 작업자를 위한 빠른 인수인계 |
| `README.md` | 실행 명령과 사용법 |

## 13. Test Coverage

테스트 위치:

```text
prpd_similarity_retrieval/tests/
service/backend/tests/test_diagnose_api.py
```

주요 검증:

- compact index 저장/로드
- compact index와 list 검색 결과 일치
- feature cache round-trip
- CLI cache build hit/miss
- batch evaluation
- hard split에서 holdout group candidate 제외
- prototype index 저장/검색/평가
- learned projection index 저장/검색/평가
- HTML review artifact 생성
- human review metric 계산
- backend diagnose API trace에 similar case 결과 포함

최근 확인된 명령:

```powershell
python -m pytest prpd_similarity_retrieval\tests\test_compact_index.py prpd_similarity_retrieval\tests\test_learned_encoder.py service\backend\tests\test_diagnose_api.py -q
npm run build
```

## 14. Current Limitations

- 현재 운영 알고리즘은 handcrafted feature baseline이다.
- PRPD 이미지가 시각화 스타일에 민감할 수 있다.
- 시계열 sampling rate/길이가 다르면 feature 비교 안정성이 떨어질 수 있다.
- leave-one-out label match는 높지만 실제 일반화 성능은 hard split 기준으로 봐야 한다.
- metadata를 ranking에서 제거했기 때문에 PRPD/시계열 입력이 누락된 경우 유사사례 품질은 제한된다.
- learned/prototype은 실험 경로이며 최종 모델로 간주하지 않는다.

## 15. Next Plan

사람 검증을 제외하고 다음에 할 수 있는 작업:

1. PRPD 이미지/시계열 feature threshold를 실제 화면 결과 기준으로 보정한다.
2. hard split 약한 장비군의 실패 사례를 기준으로 feature 개선 후보를 정리한다.
3. 실제 PRPD image encoder 후보를 붙인다.
4. 실제 time-series encoder 후보를 붙인다.
5. neural encoder 결과를 `equipment_name` hard split으로 feature baseline과 비교한다.
6. feature/prototype/learned/neural 중 장비군별 fallback 또는 ensemble 기준을 설계한다.
7. index rebuild와 배포 절차를 별도 운영 문서로 정리한다.

## 16. Acceptance Criteria

1차 기능 완료 기준:

- 진단 후 유사사례 Top 5가 자동 표시된다.
- 유사도 ranking은 PRPD/시계열만 사용한다.
- metadata-only 유사사례는 노출하지 않는다.
- 카드에서 순위 이유와 PRPD/파형 축별 요약이 보인다.
- 상세에서 PRPD 이미지와 waveform을 비교할 수 있다.
- backend API test와 frontend build가 통과한다.

현재 1차 기준은 충족된 상태다.
