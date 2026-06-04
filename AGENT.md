# AGENT: AI 코딩 에이전트 행동 강령

## 1. 목적

본 문서는 AI-Hub 산업 설비 전기 화재 사고 예방 부분방전 데이터를 이용해 시계열 분류 모델과 소형 VLM 멀티모달 진단 모델을 개발하는 AI 코딩 에이전트의 개발 규칙을 정의한다.

이 프로젝트의 1차 목표는 CSV 시계열 데이터만 사용해 부분방전 5종 상태를 분류하는 실험 파이프라인을 만드는 것이다. VLM은 2차 목표이며, PRPD 이미지, JSON 메타데이터, 시계열 요약 정보를 함께 사용해 자연어 진단 또는 구조화된 진단 결과를 생성한다.

AI는 코드를 작성하기 전에 다음 문서를 우선순위대로 확인해야 한다.

1. `PRD.md`: 루트 기준 프로젝트 목표, 시계열/VLM 실험 범위, 실행 규칙 요약
2. `docs/PRD.md`: 상세 프로젝트 목표, 범위, 실험 단계, 성공 기준
3. `docs/DATASET_EXPLAIN.md`: 통합 데이터 구조, CSV/PNG/JSON 형태, label, manifest 규칙
4. `docs/TIMESERIES_MODELS.md`: 실험 후보 시계열 모델, Core/Extended 구분, 모델별 입력 관점
5. `docs/VLM_STRATEGY.md`: VLM 모델 후보, 입력/출력 설계, QLoRA 학습 전략
6. `AGENT.md`: 코드 스타일, 실행 환경, 리소스 사용, 검증 규칙


## 데이터 구조설명
`docs/DATASET_EXPLAIN.md` 참조

## 프로젝트 범위

현재 구현 우선순위는 다음과 같다.

1. Train 데이터 구조 확인 및 `manifest.csv` 기반 데이터 매핑
2. 초기 EDA 실행 및 데이터/라벨/누수 위험 확인
3. CSV 시계열 분류 데이터셋/DataLoader 구현
4. GRU baseline 학습 및 평가
5. Non-Transformer, Transformer/SOTA, Foundation/Pretrained 시계열 모델 비교
6. 시계열 실험 결과를 바탕으로 VLM 입력용 요약 feature 또는 진단 문맥 생성
7. 소형 VLM(Qwen 계열 등)로 PRPD 이미지 + 메타데이터 + 시계열 요약을 결합한 진단 모델 개발

Forecasting은 현재 프로젝트 범위에서 제외한다. 이 프로젝트의 시계열 작업은 미래 값 예측이 아니라 `정상`, `노이즈`, `표면 방전`, `코로나 방전`, `보이드 방전` 5-class classification이다.

## 데이터 해석 규칙

Train 데이터는 `Train/manifest.csv`를 기준으로 CSV, PNG, JSON을 연결한다.

현재 로컬 `Train/manifest.csv`에서 확인한 데이터 형태:

- 전체 샘플 수는 `30,010`개다.
- label 분포는 `0~4` 각 `6,002`개로 균형이다.
- 하나의 샘플은 `timeseries_path` CSV, `image_path` PNG, `json_path` JSON으로 연결된다.
- CSV 시계열 배열 형태는 `(20, 7680)`이다.
- manifest 주요 컬럼은 `sample_id`, `split`, `json_path`, `image_path`, `timeseries_path`, `label_id`, `label_name`, 설비/환경 메타데이터, `max_discharge_value` 등이다.

규칙:

- CSV 원천 데이터는 header 없는 정수형 시계열 배열로 읽는다.
- 현재 확인된 CSV 형태는 `(20, 7680)`이다.
- `20`은 물리 센서 채널로 단정하지 않고 measurement segment 또는 pseudo-channel 차원으로 취급한다.
- `7680`은 시간축 time point로 취급한다.
- RNN/Transformer 입력에서는 필요에 따라 `(time, pseudo_channel) = (7680, 20)`으로 transpose한다.
- Conv/TCN/Patch 계열 입력에서는 `(pseudo_channel, time) = (20, 7680)` 형태를 기본 후보로 둔다.
- JSON의 `PD_type`은 정답 label로 사용한다.
- 코드에서는 가능하면 manifest의 `label_id`를 사용하고, manifest 재생성 시에는 JSON의 `label.PD_type` 기준으로 label을 만든다.
- VLM prompt에는 label leakage가 되는 `PD_type`, label명이 포함된 경로, 파일명 패턴을 넣지 않는다.
- feature baseline에서도 `sample_id`, `timeseries_path`, `image_path`, `json_path`, `label_name`, `defect_details`처럼 클래스명이나 정답 정보가 섞일 수 있는 문자열 컬럼을 feature로 사용하지 않는다.
- JSON 메타데이터는 VLM 단계에서 LLM 쪽 텍스트 문맥으로 사용할 수 있다.

## 코드 디렉터리 경계

코드는 목적에 따라 디렉터리를 분리한다.

```text
scripts/
  데이터 점검, manifest 생성, 일회성 유틸리티 스크립트

ml/
  시계열 데이터셋, 전처리, 모델 wrapper, 학습, 평가, leaderboard

vlm/
  VLM 학습 데이터 변환, prompt 템플릿, LoRA/SFT 학습, 추론

results/
  실험 결과 CSV/JSON, metric, model artifact, 로그 요약

docs/
  프로젝트 요구사항, 데이터 설명, 모델 설명 문서
```

규칙:

- 시계열 모델 학습 코드는 `ml/` 아래에 작성한다.
- VLM 데이터 변환 및 학습 코드는 `vlm/` 아래에 작성한다.
- 데이터 점검/manifest 생성처럼 반복 실행 가능한 유틸리티는 `scripts/` 아래에 작성한다.
- 실험 결과와 모델 산출물은 `results/` 아래에 저장한다.
- 원천 데이터는 `Train/` 아래 구조를 유지하고, 코드가 원천 데이터를 덮어쓰지 않는다.
- 외부 pip/오픈소스 구현이 있는 Transformer/Foundation 시계열 모델은 프로젝트 내부에 모델 본체를 재구현하지 않고, 모델별 wrapper 파일에서 import해서 사용한다.
- 모델 wrapper는 모델별 파일로 분리한다. 예: `gru.py`, `tcn.py`, `patchtst.py`, `moment.py`.
- GRU를 제외한 유명 논문 모델은 공식 repo, Hugging Face, PyTorch 기본 모듈, 또는 검증 라이브러리 구현을 우선 사용한다.
- 공식 dependency가 설치되어 있지 않은 모델은 임의 fallback 구현으로 조용히 대체하지 않고, 설치/clone 방법을 안내하는 `ImportError`를 발생시킨다.
- 공식 repo clone 연결은 환경변수로 처리한다. 예: `TSLIB_REPO`, `ITRANSFORMER_REPO`, `TIMEMIXER_REPO`, `MODERNTCN_REPO`, `UNITS_REPO`, `ONE_FITS_ALL_REPO`, `TS2VEC_REPO`.
- VLM 관련 코드나 데이터셋을 작성하기 전에는 `docs/VLM_STRATEGY.md`를 먼저 확인한다.
- VLM은 PRPD 이미지 단독 분류 모델이 아니라, PRPD 이미지 + JSON 메타데이터 + 시계열 요약 정보를 결합한 진단 리포트 생성 모델로 구현한다.
- VLM 학습에서 원본 CSV 전체를 프롬프트에 넣지 않는다. 시계열 모델 예측값, confidence, class probability, 통계 feature처럼 압축된 정보를 텍스트 문맥으로 제공한다.
- 초기 VLM 후보는 Qwen2.5-VL-3B-Instruct를 우선하고, smoke 검증에는 Qwen2-VL-2B-Instruct를 사용할 수 있다.
- VLM 초기 학습은 QLoRA SFT를 우선하며, vision encoder는 먼저 freeze하고 LLM/projector 계층 위주로 LoRA를 적용한다.
- VLM 출력은 자연어보다 구조화된 JSON 진단 결과를 우선한다.

## 모델 실험 원칙

시계열 모델 실험은 다음 그룹으로 구분한다.

- Non-Transformer: GRU, TCN, InceptionTime, ResNet1D, ModernTCN
- Transformer / Modern SOTA: PatchTST, iTransformer, TimesNet, TimeMixer
- Foundation / Pretrained: MOMENT, UniTS, GPT4TS / One-Fits-All
- Representation Learning: TS2Vec
- CPU-only optional baseline: MiniROCKET, MultiROCKET, sktime feature-based classifiers, ROCKET, Arsenal, HYDRA, feature baseline, TabPFN

원칙:

- 첫 baseline은 GRU로 시작한다.
- 처음부터 모든 모델을 구현하지 않고, Core 모델을 먼저 완성한 뒤 Extended 모델로 확장한다.
- Extended 중 `TimeMixer`, `UniTS`, `GPT4TS`, `TS2Vec`은 훈련 시간이 길어질 수 있으므로 full 30k 실행 전에 반드시 small subset smoke와 중간 subset을 먼저 실행한다.
- `iTransformer`, `ModernTCN`도 긴 시퀀스 비용이 있으므로 처음에는 `seq_len` 축소 또는 `--sample-size` 제한을 사용한다.
- 모든 모델은 동일한 split, 동일한 label mapping, 동일한 metric으로 비교한다.
- classification metric은 accuracy, macro F1, weighted F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, 실제 방전이 정상으로 예측된 수를 기본으로 한다.
- 모델 훈련 전 `python ml/scripts/validate_dataset.py --fail-on-invalid`로 manifest 경로, label, CSV shape, NaN/inf, 상수 신호를 검증한다.
- manifest에 `split=train`과 `split=valid`가 있으면 모든 runner는 해당 split을 우선 사용한다. 모델 비교용 run에서는 같은 split manifest를 재사용한다.
- feature baseline의 기본 metadata whitelist에는 `max_discharge_value`를 포함하지 않는다. 해당 값은 leakage 가능성이 있으므로 별도 ablation에서만 사용한다.
- forecasting 전용 모델(TimesFM, Chronos, Lag-Llama 등)은 현재 범위에서 제외한다.
- foundation model은 가능한 경우 pretrained checkpoint를 downstream classification head 또는 fine-tuning 방식으로 사용한다.
- from-scratch 학습과 pretrained fine-tuning 결과는 leaderboard에서 구분 기록한다.
- `sktime`이 공식/검증 classifier를 제공하는 classical TSC 모델은 직접 구현하지 않고 `sktime` runner를 사용한다. `RandomInterval`, `TSFresh`, `FreshPRINCE`, `Arsenal`은 `--allow-expensive`와 small subset 없이는 실행하지 않는다.
- `train.py`는 GPU neural/foundation model 전용 단일 학습 CLI다. `MiniROCKET`, `MultiROCKET`, `HYDRA`, `feature_*`, `sktime_*` 같은 CPU-only baseline은 `train.py`에서 직접 학습하지 않고, `ml/scripts/run_*.py` 전용 runner를 사용한다.
- CPU-only baseline도 한 번 실행할 때 하나의 모델만 실행한다. 각 runner는 인자 없이도 안전한 smoke 기본값을 갖는다. 예: `python ml/scripts/run_feature_baseline.py --model logistic`, `python ml/scripts/run_minirocket.py`.
- `train.py --list-models`에 표시되는 `cpu_only` 항목은 발견/안내용 목록이며, 해당 이름을 `--model`로 넣으면 전용 runner 명령을 로그로 안내하고 종료해야 한다.
- 현재 Train 데이터와 manifest 기준으로 초기 EDA를 한 번 실행해 label 분포, metadata 분포, CSV shape, signal 통계, phase-bin pulse 분포, leakage 위험 컬럼을 확인한다. 매 모델 훈련 전에 반복 실행할 필요는 없고, 데이터 구조, manifest 생성 방식, label mapping, feature 설계, split 정책이 바뀐 경우에만 다시 실행한다.

## Python 실행 환경 정책

모델 훈련 환경은 현재 프로젝트 루트의 `.venv`를 기본으로 사용한다.

### ML 모델 훈련 환경

GPU 기반 시계열/VLM 모델 훈련과 대규모 전처리는 다음 가상환경을 사용한다.

```text
.venv
```

목적:

- 일반 CPython 3.14, GIL 활성 버전 사용
- CUDA 지원 PyTorch 기반 GPU 학습
- 대용량 CSV 로딩, manifest 검증, feature 추출, DataLoader 실행
- pandas/numpy/scikit-learn 등 CPU multi-thread 작업용 환경

주의:

- 라이브러리가 Python 3.14를 지원하지 않는 경우, 해당 모델은 호환 가능한 환경에서 별도 실행하고 사유를 기록한다.
- `.venv`에서 pip install 또는 wheel 로딩이 실패하는 패키지는 무리하게 우회하지 않고, 해당 패키지가 공식 지원하는 별도 Python 환경에서 실행한 뒤 실행 환경 차이를 실험 로그에 기록한다.
- `.venv`에서 호환 wheel이 없다는 이유로 패키지를 직접 소스 빌드하지 않는다. 특히 PyTorch, CUDA, 시계열 foundation model 관련 패키지는 사용자가 명시적으로 지시하지 않는 한 소스 빌드를 시도하지 않는다.
- 모델별 실행 환경은 실험 로그에 기록한다.

전통 ML/CPU baseline 예외:

- 현재 프로젝트의 주 실험은 딥러닝 시계열 분류이므로 LightGBM/CatBoost류 tabular baseline은 기본 실험 범위에 포함하지 않는다.
- MiniROCKET, MultiROCKET, HYDRA, feature baseline, TabPFN, shapelet, sklearn classifier처럼 CPU 기반 전통 시계열 baseline을 추가하는 경우에도 기본은 `.venv`를 사용한다.
- 해당 패키지가 Python 3.14를 지원하지 않으면 별도 호환 환경에서 실행하고, 실행 환경 차이를 실험 로그에 기록한다.
- CPU fallback은 내부 멀티스레딩을 사용하므로 가능한 경우 `n_jobs=14` 또는 라이브러리별 thread 옵션을 명시한다.

Neural/Transformer/Foundation 예외:

- GRU, TCN, InceptionTime, ResNet1D, ModernTCN, PatchTST, iTransformer, TimesNet, TimeMixer, MOMENT, UniTS, GPT4TS 계열은 기본적으로 GPU 학습 또는 GPU 추론을 사용한다.
- 이 모델들은 PyTorch/CUDA/공식 pretrained checkpoint 호환성이 중요하므로 `.venv`에서 설치하고 실행한다.
- `.venv`에 PyTorch, CUDA extension, time-series foundation model package를 소스 빌드해서 억지로 맞추지 않는다.
- `ml/requirements.txt`는 `.venv` 기준 dependency 목록으로 취급한다.
- GPU를 사용할 수 없는 경우에만 명시적으로 CPU fallback을 검토하고, 실행 환경과 사유를 실험 로그에 기록한다.
- MOMENT, UniTS, GPT4TS 등 pretrained checkpoint가 필요한 모델은 브라우저 로그인 프롬프트에 의존하지 않고 Hugging Face token, local checkpoint path, 또는 명시적 cache path를 사용한다.

### 서버/API/관리자 화면 실행 환경

FastAPI 백엔드, 관리자 API, 일반 서버를 추가하는 경우에도 다음 가상환경을 사용한다.

```text
.venv
```

목적:

- 일반 CPython 3.14 환경 사용
- FastAPI, SQLAlchemy, Alembic, PostgreSQL 연동, 관리자 API 실행
- ML 훈련 환경과 서비스 runtime dependency 충돌 방지

프론트엔드는 Python 가상환경이 아니라 `frontend/`의 Node.js 패키지 환경을 사용한다.

## ML 훈련 리소스 사용 정책

## ML/AI 진행률 표시 정책

ML/AI 모델 개발 코드는 장시간 실행되는 작업의 진행 상황을 사용자가 확인할 수 있게 구성한다.

적용 대상:

- 대용량 CSV 로딩 및 chunk sampling
- 전처리, encoding, split 생성
- 모델 학습 epoch/iteration loop
- batch prediction
- hyperparameter sweep
- foundation model sample 평가
- leaderboard 생성

원칙:

- 가능한 경우 `tqdm` 기반 progress bar를 사용한다.
- PyTorch Lightning, Hugging Face Trainer, 자체 training loop처럼 callback/logging이 있는 모델은 progress bar 또는 주기적 log 중 하나를 제공한다.
- CLI script는 `--no-progress` 옵션을 둘 수 있으나 기본값은 progress 표시이다.
- progress 출력은 metric/result CSV를 오염시키지 않는다.
- 긴 작업은 현재 단계, 처리 row 수, 전체 row 수 또는 chunk 수, elapsed time을 알 수 있게 한다.
- 서버/API runtime에서는 terminal progress bar 대신 structured log 또는 DB job status로 진행 상태를 노출한다.

## ML/AI 터미널 로깅 정책

ML/AI 모델 개발 코드는 중요한 실행 상태를 logger의 `info` 레벨로 터미널에 출력한다.

적용 대상:

- 데이터 파일 로딩 시작/완료
- 사용 split, 입력 shape, label mapping, metadata 사용 여부
- train/validation row 수
- sample size, seed, label별 row 수
- 모델명, 실행 환경, Python executable/version
- 주요 hyperparameter, sequence length, pseudo-channel 수
- GPU 실행 여부, CUDA device, mixed precision 사용 여부
- GPU/CPU worker 설정
- 학습 시작/완료, 학습 시간
- 예측 시작/완료, 예측 시간
- metric 계산 결과
- 결과 CSV/JSON 저장 경로
- model artifact 저장/로드 경로, 구현 시

원칙:

- 단순 `print()` 대신 표준 `logging` 모듈 또는 프로젝트 공통 logger를 사용한다.
- 기본 로그 레벨은 `INFO`로 둔다.
- 경고성 상황은 `warning`, 실패는 `error` 또는 예외로 남긴다.
- progress bar와 logger 출력이 서로 깨지지 않도록 `tqdm.write()` 또는 logging handler 설정을 사용한다.
- 실험 결과 CSV에는 metric/config를 저장하고, 터미널 log는 실행 상태를 사람이 추적하기 위한 용도로 둔다.

## ML/AI 실험 실행 단위 정책

모델 훈련 스크립트는 한 번 실행할 때 정확히 하나의 실험만 수행한다.

원칙:

- 1 script run = 1 model = 1 input config = 1 sample/full-data setting = 1 seed = 1 training job
- 하나의 CLI 실행에서 여러 모델을 순차 실행하지 않는다.
- 하나의 CLI 실행에서 여러 sample size를 순차 실행하지 않는다.
- 하나의 CLI 실행에서 여러 input config를 순차 실행하지 않는다.
- 하나의 CLI 실행에서 여러 seed 또는 hyperparameter sweep을 순차 실행하지 않는다.
- sweep, grid search, AutoML, batch leaderboard runner를 만들지 않는다.
- leaderboard 생성은 이미 완료된 `results/experiments.csv`를 읽는 후처리 작업으로만 수행한다.
- 대용량 학습은 사용자가 명시적으로 실행한 단일 명령만 수행한다.
- 실패한 실험을 자동으로 다음 실험으로 넘어가게 만들지 않는다.
- 재시도도 사용자의 명시적 명령으로만 수행한다.

이유:

- 30,010건 Train working dataset과 원본 대규모 데이터를 다룰 때 연속 실험은 GPU/CPU/메모리 점유 위험이 크다.
- 여러 실험이 한 프로세스에 섞이면 로그, 진행률, 실패 원인, 결과 CSV 해석이 어려워진다.
- Transformer/Foundation 모델처럼 장시간/고자원 작업은 사용자가 실험 단위를 명확히 통제해야 한다.

### GPU 훈련

GPU 사용 가능 모델은 가능한 한 GPU를 우선 사용한다.

기본 정책:

```text
target_gpu_memory_utilization: 0.90
```

원칙:

- 모델 훈련 시 GPU 가용 메모리의 최대 90%까지 사용하는 것을 목표로 한다.
- `train.py`에서 `--batch-size`를 생략하면 synthetic forward/backward probe로 단일 모델의 batch size를 자동 산정한다.
- `--batch-size`를 명시한 경우에는 사용자가 의도한 수동 실험으로 보고 자동 batch sizing을 끈다.
- OOM이 발생하면 batch size, embedding dimension, model depth 순서로 줄인다.
- GPU memory 사용량은 실험 결과에 기록한다.
- 다른 서비스 프로세스가 같은 GPU를 사용 중이면 90% 정책을 낮출 수 있다.

### CPU 훈련

CPU 기반 전처리와 전통 시계열 baseline도 `.venv` 환경을 우선 사용한다. PyTorch 기반 Neural/Transformer/Foundation 모델 역시 `.venv`에서 실행한다.

원칙:

- 이 프로젝트의 주 훈련은 GPU 기반이므로 LightGBM식 CPU multi-thread 훈련 최적화 코드는 기본으로 작성하지 않는다.
- Windows 로컬 실험에서는 PyTorch DataLoader `num_workers=0`을 기본값으로 둔다.
- CSV 로딩 병목이 확인된 경우에만 `num_workers`를 2, 4, 8 순서로 올려 보고, 안정성이 확인되면 최대 14까지 사용한다.
- CPU 기반 전처리, feature 추출, batch prediction 작업에서 병렬화가 필요한 경우에만 `n_jobs` 또는 worker 옵션을 명시한다.
- 시스템 부하가 과도하거나 메모리 병목이 발생하면 workers를 낮추고 사유를 기록한다.
