# PRPD Similarity Retrieval Handoff

## 현재 결론

현재 폴더 안에는 PRPD 이미지와 시계열 CSV를 이용해 과거 점검 사례를 자동으로 찾는 유사 사례 검색 기능이 구축되어 있다.

운영적으로 바로 사용하는 경로는 다음이다.

- feature retrieval: 수작업 PRPD/시계열 feature 기반 검색

실험/평가용 경로는 다음 두 가지다.

- prototype encoder: feature vector를 embedding index로 변환한 검색/평가 경로
- learned projection encoder: feature standardization/PCA/label centroid 기반 embedding 검색

백엔드는 `domain_feature_case_retriever`를 사용한다. PRPD/시계열 feature만 ranking에 반영하고, 메타데이터/라벨은 운영 유사도 점수에서 제외한다. learned projection은 최종 CNN/TS2Vec 모델은 아니며, 다음 neural encoder를 붙이기 전 embedding 검색/평가 경로를 검증하는 baseline이다.

## 완료된 산출물

- `case_feature_index.npz`: 전체 30,010건 feature index
- `case_embedding_index.prototype.npz`: 전체 30,010건 prototype embedding index
- `case_embedding_index.learned.npz`: 전체 30,010건 learned projection embedding index
- `hard_split_report.full.feature.md`: 장비 holdout feature/metadata 전체 평가
- `hard_split_report.full.prototype.md`: 장비 holdout prototype 전체 평가
- `hard_split_report.full.learned.md`: 장비 holdout learned projection 전체 평가
- `hard_split_failures.cncv_w.sample.md`: CNCV-W 실패 사례 샘플
- `hard_split_failures.single_oil_transformer.sample.md`: 단상 유입변압기 실패 사례 샘플
- `hard_split_review.cncv_w.sample.html`: CNCV-W HTML 검토 화면
- `hard_split_review.single_oil_transformer.sample.html`: 단상 유입변압기 HTML 검토 화면
- `evaluation_report.md`: 평가 결과 요약
- `prd.md`: 전체 설계/구축 방향

대용량 index/cache 파일은 `.gitignore`로 제외되어 있다.

## 핵심 평가 결과

Leave-one-out 평가는 feature/prototype 모두 top-1/top-3 `1.0`으로 나왔지만, 같은 데이터셋 안의 매우 가까운 사례를 찾는 구조라 실제 일반화 성능으로 보기 어렵다.

더 중요한 평가는 장비 holdout hard split이다.

Feature/metadata hard split:

- 약한 그룹: `단상 유입변압기` top-3 `0.197301`, `25.8kV GIS` top-3 `0.333200`, `계기용 변압기` top-3 `0.342429`
- 강한 그룹: `22.9kV 배전반` top-3 `0.979200`, `7.2kV 배전반` top-3 `0.773200`, `전력용 유입변압기` top-3 `0.542129`

Prototype hard split:

- 약한 그룹: `25.8kV GIS` top-3 `0.332200`, `단상 유입변압기` top-3 `0.360420`, `CNCV-W` top-3 `0.392504`
- 강한 그룹: `22.9kV 배전반` top-3 `0.994800`, `7.2kV 배전반` top-3 `0.773200`, `전력용 유입변압기` top-3 `0.614393`

Prototype은 `단상 유입변압기`, `CNCV-W`, `계기용 변압기`에서 feature baseline보다 개선됐지만, `25.8kV GIS`와 `ACSR-OC`는 아직 충분히 개선되지 않았다.

Learned projection hard split:

- 약한 그룹: `25.8kV GIS` top-3 `0.262400`, `CNCV-W` top-3 `0.353523`, `단상 유입변압기` top-3 `0.498351`
- 강한 그룹: `7.2kV 배전반` top-3 `0.913600`, `22.9kV 배전반` top-3 `0.908800`, `전력용 유입변압기` top-3 `0.750825`

Learned projection은 `ACSR-OC`, `TFR-CV`, `계기용 변압기`, `단상 유입변압기`, `전력용 유입변압기`, `7.2kV 배전반`에서 개선됐지만, `25.8kV GIS`, `CNCV-W`, `22.9kV 배전반`은 feature/prototype 대비 낮다. 따라서 다음 neural encoder도 반드시 hard split과 fallback/ensemble 기준으로 봐야 한다.

## 현재 구현 상태

구현 완료:

- feature 추출: `features.py`
- compact index: `compact_index.py`
- 검색/평가 CLI: `cli.py`
- batch evaluation: `batch_evaluation.py`
- hard split evaluation: `hard_split_evaluation.py`
- prototype encoder: `prototype_encoder.py`
- learned projection encoder: `learned_encoder.py`
- HTML review artifact: `review_artifact.py`
- human review metric evaluator: `human_review.py`
- backend adapter: `backend_adapter.py`
- frontend 현재 점검 유사 사례 카드/상세 모달의 자동 추천 결과 표시

## 최근 검증

마지막 검증 명령:

```powershell
python -m pytest prpd_similarity_retrieval\tests service\backend\tests\test_diagnose_api.py -q
python -m prpd_similarity_retrieval.cli build-learned-index --feature-index prpd_similarity_retrieval\case_feature_index.npz --output prpd_similarity_retrieval\case_embedding_index.learned.npz
python -m prpd_similarity_retrieval.cli evaluate-learned-index --index prpd_similarity_retrieval\case_embedding_index.learned.npz --top-k 3 --batch-size 256
python -m compileall prpd_similarity_retrieval
```

마지막 확인 결과:

- `16 passed` for focused learned/hard split tests
- `15 passed` for backend diagnose API tests
- `compileall` 통과
- `__pycache__` 정리 완료

## 다음 작업

1. HTML review에서 실제 reviewer export CSV/JSON을 받아 `human_review_metrics.md`를 생성한다.
2. CNN 기반 PRPD image encoder와 TS2Vec 계열 time-series encoder를 붙인다.
3. 새 embedding도 반드시 `equipment_name` hard split으로 feature/prototype/learned projection baseline과 비교한다.
4. 현재 진단 화면의 유사 사례 섹션에는 검색창이 아니라 “현재 점검과 비슷했던 과거 사례” 자동 추천 형태를 유지한다.
