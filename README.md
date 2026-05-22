# partial-discharge-diagnosis

AI-Hub 산업 설비 전기 화재 사고 예방 부분방전 데이터를 이용한 시계열 분류 및 소형 VLM 멀티모달 진단 프로젝트입니다.

## Scope

- CSV 시계열 기반 부분방전 5-class classification
- PRPD 이미지, JSON 메타데이터, 시계열 요약 정보를 결합한 VLM 진단 모델
- Forecasting은 현재 범위에서 제외

## Documents

- `docs/PRD.md`: 프로젝트 요구사항과 실험 계획
- `docs/DATASET_EXPLAIN.md`: 데이터 구조와 manifest 설명
- `docs/TIMESERIES_MODELS.md`: 실험 후보 시계열 모델 정리
- `docs/VLM_STRATEGY.md`: VLM 모델 후보와 학습 전략
- `AGENT.md`: 개발/실험 실행 규칙

## Time-Series Training

공통 진입점은 루트의 `train.py`입니다. 모델별 입력 shape은 각 wrapper의 `input_layout`에 따라 dataloader가 자동으로 맞춥니다.

```bash
python train.py --list-models
python train.py --model gru --sample-size 100 --epochs 1
python train.py --model patchtst --sample-size 500 --epochs 3
python train.py --model moment --sample-size 500 --epochs 3
```

`train.py`는 한 번 실행할 때 정확히 하나의 모델만 훈련합니다. `core`, `extended`, `all`, `cpu_only` 같은 그룹명은 `--model` 값으로 지원하지 않습니다.
Core 모델은 `GRU, InceptionTime, PatchTST, TimesNet, MOMENT`입니다.
Extended GPU 모델은 `TCN, ResNet1D, iTransformer, TimeMixer, UniTS, GPT4TS, TS2Vec`입니다.
`MiniROCKET`은 Extended 성격의 optional classical baseline이지만 `sktime/sklearn` 기반 CPU-only 모델이라 GPU 학습 라인업과 분리합니다.

일반 PyTorch 모델은 `train.py`가 `ml/src/experiments/runner.py`를 호출합니다. `TS2Vec`는 학습 방식이 달라 공식 구현 기반 별도 runner로 자동 분기합니다.
이 프로젝트의 훈련 코드는 CUDA GPU 전용입니다. GPU가 없으면 CPU로 fallback하지 않고 로그를 남긴 뒤 훈련을 시작하지 않습니다.
`--batch-size`를 생략하면 synthetic forward/backward probe로 CUDA 메모리 목표치(`--target-gpu-memory-utilization`, 기본 0.90) 이내의 batch size를 자동 산정합니다. 수동 batch size가 필요할 때만 `--batch-size`를 지정합니다.
