# PRPD Similarity Retrieval

이 폴더는 현재 점검의 PRPD 이미지와 시계열 CSV를 과거 사례와 비교하는 도메인 feature 기반 유사사례 검색 기능이다.

현재 기능 범위와 완료 상태는 [prd.md](prd.md)를 먼저 본다. 다음 작업 인수인계는 [HANDOFF.md](HANDOFF.md)를 본다.

## 현재 구현 범위

- `data/manifest.csv`에서 사례 목록 로드
- PRPD 이미지 feature 추출
- 시계열 CSV feature 추출
- 메타데이터/라벨 baseline 점수 계산
- feature index 생성: 운영용 `.npz` 압축 행렬, 호환용 `.json`
- index 내부 sample 기준 top-k 검색
- 외부 이미지/CSV 기준 top-k 검색
- 장비/센서/전압 그룹 holdout hard split 평가
- prototype embedding index와 prototype-only hard split 평가
- learned projection embedding index 생성, 검색, 평가
- learned projection 전체 장비 holdout hard split 평가
- backend `similar_case_tool` 연결용 adapter 제공
- frontend 현재 점검 유사 사례 카드/상세 모달에 PRPD/시계열 자동 추천 결과 표시

현재 운영 기본 경로는 `domain_feature_case_retriever`다. PRPD image feature와 time-series feature만 ranking에 사용하며, 메타데이터/라벨은 운영 유사도 점수에서 제외한다. `prototype_encoder.py`와 `learned_encoder.py`는 최종 CNN/TS2Vec 모델이 아니라, 실제 neural encoder 출력으로 교체하기 전 embedding index/search/evaluation 경로를 고정한 실험 구현이다.

## 점수 구성

```text
similarity =
  0.55 * PRPD image feature cosine similarity
+ 0.45 * time-series feature cosine similarity
```

누락된 입력은 점수 계산에서 제외하고, 남은 component weight로 재정규화한다.

## 사용법

샘플 index 생성:

```powershell
python -m prpd_similarity_retrieval.cli build-index `
  --limit 100 `
  --workers 4 `
  --progress-every 25 `
  --cache prpd_similarity_retrieval\case_feature_cache.sample.jsonl `
  --output prpd_similarity_retrieval\case_feature_index.sample.npz
```

라벨별 균형 샘플 index 생성:

```powershell
python -m prpd_similarity_retrieval.cli build-index `
  --per-label-limit 10 `
  --workers 4 `
  --cache prpd_similarity_retrieval\case_feature_cache.stratified.jsonl `
  --output prpd_similarity_retrieval\case_feature_index.stratified.npz
```

전체 index 생성:

```powershell
python -m prpd_similarity_retrieval.cli build-index `
  --workers 4 `
  --progress-every 1000 `
  --cache prpd_similarity_retrieval\case_feature_cache.jsonl `
  --output prpd_similarity_retrieval\case_feature_index.npz
```

출력 확장자가 `.npz`면 vector matrix를 압축 저장하고, `.json`이면 사람이 읽기 쉬운 호환 포맷으로 저장한다. 전체 데이터셋은 `.npz`를 기본으로 사용한다.
`--workers`는 PRPD 이미지와 시계열 feature 추출을 병렬화한다. 전체 데이터셋 빌드에서는 4부터 시작하고, 장비 부하를 보면서 조정한다.
`--progress-every`는 진행률 JSON을 stderr로 출력하므로 최종 stdout 결과 JSON은 유지된다.
`--cache`는 feature 추출 결과를 JSONL로 누적 저장한다. 빌드가 중간에 끊겨도 다음 실행에서 이미 추출한 sample은 재사용하고 누락분만 이어서 계산한다.

backend에서 특정 index 파일을 강제하려면 환경 변수를 지정한다.

```powershell
$env:PRPD_CASE_FEATURE_INDEX = "C:\Users\Kello\partial-discharge-diagnosis\prpd_similarity_retrieval\case_feature_index.npz"
```

backend는 다음 feature index 순서로 검색한다.

1. `prpd_similarity_retrieval\case_feature_index.npz`
2. `prpd_similarity_retrieval\case_feature_index.json`
3. `prpd_similarity_retrieval\case_feature_index.sample.npz`
4. `prpd_similarity_retrieval\case_feature_index.sample.json`
5. index가 없으면 빈 유사사례 결과 반환

index에 있는 sample 기준 유사 사례 검색:

```powershell
python -m prpd_similarity_retrieval.cli query-sample `
  --index prpd_similarity_retrieval\case_feature_index.sample.npz `
  --sample-id "노이즈_고체_ACSR-OC_230910_195222_HFCT_1000" `
  --top-k 5
```

외부 점검 파일 기준 유사 사례 검색:

```powershell
python -m prpd_similarity_retrieval.cli query-files `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --image-path path\to\current_prpd.png `
  --timeseries-path path\to\current_signal.csv `
  --metadata-json path\to\metadata.json `
  --top-k 5
```

index 기준 leave-one-out 라벨 매칭 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-index `
  --index prpd_similarity_retrieval\case_feature_index.sample.npz `
  --top-k 3
```

feature retrieval과 metadata-only baseline 비교:

```powershell
python -m prpd_similarity_retrieval.cli compare-baseline `
  --index prpd_similarity_retrieval\case_feature_index.stratified.npz `
  --top-k 3
```

평가 명령은 기본적으로 query label을 숨긴다. 모델 판정 라벨까지 반영한 운영 조건을 보고 싶으면 `--use-query-label`을 추가한다.

장비 그룹 holdout hard split 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-hard-split `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --holdout-value "25.8kV GIS" `
  --top-k 3 `
  --include-prototype
```

`--holdout-value`를 생략하면 가장 큰 non-global 그룹을 holdout query로 자동 선택한다. Hard split은 holdout 그룹을 candidate에서 제외하므로, leave-one-out보다 새로운 장비/조건 일반화에 가까운 검증이다.

장비 그룹별 hard split 리포트 생성:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-hard-split-report `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --limit-per-holdout 30 `
  --top-k 3 `
  --include-prototype `
  --format markdown `
  --output prpd_similarity_retrieval\hard_split_report.sample.md
```

`--limit-per-holdout`를 빼면 각 holdout 그룹 전체를 평가한다. 전체 평가는 더 오래 걸리므로, 먼저 제한 리포트로 실패 장비군을 빠르게 찾고 전체 리포트를 돌린다.

전체 장비 holdout 리포트 생성:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-hard-split-report `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --top-k 3 `
  --batch-size 256 `
  --progress-every 1000 `
  --format markdown `
  --output prpd_similarity_retrieval\hard_split_report.full.feature.md
```

전체 장비 holdout prototype-only 리포트 생성:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-prototype-hard-split-report `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --top-k 3 `
  --batch-size 256 `
  --progress-every 1000 `
  --format markdown `
  --output prpd_similarity_retrieval\hard_split_report.full.prototype.md
```

hard split 실패 사례 샘플링:

```powershell
python -m prpd_similarity_retrieval.cli sample-hard-split-failures `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --holdout-value CNCV-W `
  --top-k 3 `
  --max-failures 5 `
  --format markdown `
  --output prpd_similarity_retrieval\hard_split_failures.cncv_w.sample.md
```

이 명령은 holdout query의 top-k 결과 안에 같은 label이 없는 사례만 뽑는다. 현재 검색이 어떤 방전 유형으로 잘못 끌리는지 직접 확인할 때 사용한다.

PRPD 이미지와 시계열을 같이 보는 HTML review 생성:

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

HTML review에서는 각 retrieved case마다 `유사`, `애매`, `비유사`를 선택하고 메모를 남길 수 있다. 상단 `CSV`/`JSON` 버튼은 현재 판정값을 브라우저에서 내려받는 용도다.

export된 human review CSV/JSON 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-human-reviews `
  --input path\to\hard_split_human_reviews.csv `
  --top-k 3 `
  --breakdown-field query_equipment_name `
  --format markdown `
  --output prpd_similarity_retrieval\human_review_metrics.md
```

기본 accepted relevance는 `similar`다. `uncertain`도 정답으로 포함하려면 `--accepted-value uncertain`을 추가한다.

Prototype encoder index 생성:

```powershell
python -m prpd_similarity_retrieval.cli build-prototype-index `
  --feature-index prpd_similarity_retrieval\case_feature_index.npz `
  --output prpd_similarity_retrieval\case_embedding_index.prototype.npz
```

Prototype encoder sample 검색:

```powershell
python -m prpd_similarity_retrieval.cli query-prototype-sample `
  --index prpd_similarity_retrieval\case_embedding_index.prototype.npz `
  --sample-id "노이즈_고체_ACSR-OC_230910_195222_HFCT_1000" `
  --top-k 3
```

Prototype encoder 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-prototype-index `
  --index prpd_similarity_retrieval\case_embedding_index.prototype.npz `
  --top-k 3 `
  --batch-size 256
```

Learned projection encoder index 생성:

```powershell
python -m prpd_similarity_retrieval.cli build-learned-index `
  --feature-index prpd_similarity_retrieval\case_feature_index.npz `
  --output prpd_similarity_retrieval\case_embedding_index.learned.npz
```

Learned projection sample 검색:

```powershell
python -m prpd_similarity_retrieval.cli query-learned-sample `
  --index prpd_similarity_retrieval\case_embedding_index.learned.npz `
  --sample-id "노이즈_고체_ACSR-OC_230910_195222_HFCT_1000" `
  --top-k 3
```

Learned projection 전체 leave-one-out 평가:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-learned-index `
  --index prpd_similarity_retrieval\case_embedding_index.learned.npz `
  --top-k 3 `
  --batch-size 256
```

Learned projection 전체 장비 holdout 리포트 생성:

```powershell
python -m prpd_similarity_retrieval.cli evaluate-learned-hard-split-report `
  --index prpd_similarity_retrieval\case_feature_index.npz `
  --split-field equipment_name `
  --top-k 3 `
  --batch-size 256 `
  --progress-every 1000 `
  --format markdown `
  --output prpd_similarity_retrieval\hard_split_report.full.learned.md
```

## 파일 구조

- `prd.md`: 전체 구축 PRD
- `hard_split_report.sample.md`: 장비 holdout별 제한 hard split 리포트
- `hard_split_report.full.feature.md`: 장비 holdout별 전체 feature/metadata hard split 리포트
- `hard_split_report.full.prototype.md`: 장비 holdout별 전체 prototype encoder hard split 리포트
- `hard_split_report.full.learned.md`: 장비 holdout별 전체 learned projection hard split 리포트
- `hard_split_failures.*.sample.md`: 낮은 성능 holdout의 top-k 실패 사례 샘플
- `hard_split_review.*.sample.html`: 실패 query와 retrieved case의 PRPD/시계열 side-by-side review
- `features.py`: PRPD/시계열/메타데이터 feature 추출
- `feature_cache.py`: 재시작 가능한 JSONL feature cache
- `retrieval.py`: feature index 저장/로드와 top-k 검색
- `compact_index.py`: 전체 데이터셋용 `.npz` 압축 행렬 index 저장/검색
- `hard_split_evaluation.py`: group holdout hard split 평가
- `backend_adapter.py`: backend `SimilarCaseRetrievalAdapter` 호환 adapter
- `prototype_encoder.py`: PRPD image/time-series prototype embedding encoder
- `learned_encoder.py`: PCA 기반 supervised projection embedding index/search/evaluation 모듈
- `models.py`: 검색 입력/출력 데이터 구조
- `cli.py`: index 생성, query 실행, 평가 비교 CLI
- `tests/`: index/search/evaluation/review artifact 테스트

## 현재 로컬 검증 상태

- 전체 `manifest.csv` 기준 `case_feature_index.npz` 생성 완료: `30,010`건
- full index feature coverage: PRPD image `30,010`건, time-series `30,010`건
- full index 파일 크기: 약 `34 MB`
- 재시작 cache 파일 크기: 약 `400 MB`
- full index 단일 sample query 검증 완료
- full index `--limit 50` 기준 metadata baseline 비교 완료
- full index `--limit 500` 기준 metadata baseline 비교 완료: 약 `66.2s`
- full index 전체 `30,010`건 metadata baseline 비교 완료: 약 `500.6s`
  - feature retrieval top-1/top-3 label match: `1.0` / `1.0`
  - metadata baseline top-1/top-3 label match: `0.233322` / `0.233322`
- full index 전체 `30,010`건 label/equipment/sensor breakdown 리포트 완료: [evaluation_report.md](evaluation_report.md)
- prototype encoder index 생성 완료: `case_embedding_index.prototype.npz`
- prototype encoder 전체 `30,010`건 평가 완료: 약 `20.9s`
  - prototype top-1/top-3 label match: `1.0` / `1.0`
- learned projection index 생성 완료: `case_embedding_index.learned.npz`
- learned projection 전체 `30,010`건 평가 완료:
  - learned top-1/top-3 label match: `0.998834` / `0.999367`
- hard split 제한 검증 완료:
  - command: `evaluate-hard-split --limit 50 --top-k 3 --include-prototype`
  - split field: `equipment_name`
  - holdout: `25.8kV GIS`
  - train/query: `25,010` / `50`
  - feature retrieval top-1/top-3 label match: `0.32` / `0.32`
  - prototype encoder top-1/top-3 label match: `0.06` / `0.32`
  - metadata baseline top-1/top-3 label match: `0.0` / `0.0`
- 장비 9개 holdout 제한 리포트 생성 완료: [hard_split_report.sample.md](hard_split_report.sample.md)
  - command: `evaluate-hard-split-report --split-field equipment_name --limit-per-holdout 30 --top-k 3 --include-prototype`
  - 일부 장비군은 feature/prototype 모두 `0.0`으로 나와, full report와 실패 사례 샘플링이 필요하다.
- 장비 9개 holdout 전체 feature/metadata 리포트 생성 완료: [hard_split_report.full.feature.md](hard_split_report.full.feature.md)
  - command: `evaluate-hard-split-report --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000`
  - lowest feature top-3: `단상 유입변압기` `0.197301`, `25.8kV GIS` `0.333200`, `계기용 변압기` `0.342429`
  - strongest feature top-3: `22.9kV 배전반` `0.979200`, `7.2kV 배전반` `0.773200`, `전력용 유입변압기` `0.542129`
  - metadata baseline은 일반 장비군 `0.200000`, 배전반 `0.400000`으로 고정되어 feature baseline의 장비군별 편차가 명확하다.
- 장비 9개 holdout 전체 prototype encoder 리포트 생성 완료: [hard_split_report.full.prototype.md](hard_split_report.full.prototype.md)
  - command: `evaluate-prototype-hard-split-report --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000`
  - weakest prototype top-3: `25.8kV GIS` `0.332200`, `단상 유입변압기` `0.360420`, `CNCV-W` `0.392504`
  - strongest prototype top-3: `22.9kV 배전반` `0.994800`, `7.2kV 배전반` `0.773200`, `전력용 유입변압기` `0.614393`
  - prototype은 `단상 유입변압기`, `CNCV-W`, `계기용 변압기`에서 feature baseline보다 개선됐지만 `25.8kV GIS`, `ACSR-OC`는 개선되지 않았다.
- 장비 9개 holdout 전체 learned projection 리포트 생성 완료: [hard_split_report.full.learned.md](hard_split_report.full.learned.md)
  - command: `evaluate-learned-hard-split-report --split-field equipment_name --top-k 3 --batch-size 256 --progress-every 1000`
  - weakest learned top-3: `25.8kV GIS` `0.262400`, `CNCV-W` `0.353523`, `단상 유입변압기` `0.498351`
  - strongest learned top-3: `7.2kV 배전반` `0.913600`, `22.9kV 배전반` `0.908800`, `전력용 유입변압기` `0.750825`
  - learned projection은 `ACSR-OC`, `TFR-CV`, `계기용 변압기`, `단상 유입변압기`, `전력용 유입변압기`, `7.2kV 배전반`에서 개선됐지만 `25.8kV GIS`, `CNCV-W`, `22.9kV 배전반`은 feature/prototype 대비 낮다.
- hard split 실패 사례 샘플 생성 완료:
  - [hard_split_failures.cncv_w.sample.md](hard_split_failures.cncv_w.sample.md)
  - [hard_split_failures.single_oil_transformer.sample.md](hard_split_failures.single_oil_transformer.sample.md)
  - `CNCV-W` 노이즈 query가 정상/표면 방전/코로나 방전 ACSR-OC 사례로 잘못 끌리는 패턴 확인
  - `단상 유입변압기` 노이즈 query가 전력용 유입변압기 정상 사례로 잘못 끌리는 패턴 확인
- hard split visual review HTML 생성 완료:
  - [hard_split_review.cncv_w.sample.html](hard_split_review.cncv_w.sample.html)
  - [hard_split_review.single_oil_transformer.sample.html](hard_split_review.single_oil_transformer.sample.html)
  - 각 실패 query와 top-3 neighbor의 PRPD 이미지, 시계열 SVG, label, similarity score를 나란히 표시
  - neighbor별 `유사/애매/비유사` 판정과 메모 입력, CSV/JSON export 지원
  - export에는 query/neighbor 장비명과 센서 타입을 포함하므로 장비군별 human metric 계산 가능
  - Playwright 렌더링 확인: 각 HTML `5` failures, `20/20` images loaded, `20` waveforms, `15` review controls
- human review metrics evaluator 구현 완료:
  - command: `evaluate-human-reviews`
  - metrics: `accepted_neighbor_rate`, `human_relevance_at_k`, `accepted_or_uncertain_at_k`
  - CSV/JSON input, markdown/json output, arbitrary field breakdown 지원
- 운영 runtime 연결 완료:
  - `backend_adapter.py`: `domain_feature_case_retriever`로 PRPD/시계열 feature 검색 수행
  - `cli.py`: learned index build/query/evaluate/full hard split report 명령은 실험 경로로 유지
  - frontend `현재 점검 유사 사례` 카드/상세 모달에 도메인 feature 기반 추천 결과 표시
  - 카드에 Top 순위 이유와 PRPD/파형 축별 요약 표시
- backend `test_diagnose_api.py` 포함 테스트 통과

## 다음 작업

1. 실제 reviewer export 파일을 모아 `human_review_metrics.md`를 생성한다.
2. CNN/TS2Vec 계열 encoder 학습 후보를 붙이고 learned projection baseline과 같은 hard split으로 비교한다.
