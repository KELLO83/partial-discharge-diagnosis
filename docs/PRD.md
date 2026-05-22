# PRD: 부분방전 시계열 및 소형 VLM 진단 프로젝트

## 1. 프로젝트 개요

이 프로젝트는 AI-Hub의 `산업 설비 전기 화재 사고 예방 부분방전 데이터`를 활용하여 부분방전 상태를 진단하는 모델을 개발하는 것을 목표로 한다.

핵심 방향은 단일 비전 모델 개발이 아니라, 다음 두 가지를 연습하고 구현하는 것이다.

1. 부분방전 시계열 데이터 기반 분류 모델
2. PRPD 이미지, 메타데이터, 시계열 정보를 결합한 소형 VLM 기반 진단 모델

이미 비전 모델 개발 경험이 충분하다는 전제에서, ResNet/EfficientNet 같은 단일 이미지 분류 모델 개발은 핵심 범위에서 제외한다. 대신 시계열 모델링과 VLM 멀티모달 정렬(Alignment)에 집중한다.

## 2. 문제 정의

산업 전력 설비에서 발생하는 부분방전은 전기 화재나 절연 파괴의 주요 전조 현상이다. 이 프로젝트는 PRPD 이미지와 부분방전 시계열 신호, 설비/환경 메타데이터를 활용하여 현재 설비 상태를 진단한다.

기본 분류 대상은 다음 5개 클래스다.

| 라벨 ID | 라벨명 |
| --- | --- |
| 0 | 정상 |
| 1 | 노이즈 |
| 2 | 표면 방전 |
| 3 | 코로나 방전 |
| 4 | 보이드 방전 |

최종 VLM 모델은 단순 라벨만 출력하는 것이 아니라, 현장 엔지니어가 이해할 수 있는 자연어 진단 결과 또는 구조화된 JSON 진단 결과를 출력하는 것을 목표로 한다.

## 3. 프로젝트 목표

### 3.1 1차 목표

- AI-Hub 데이터의 `.PNG`, `.CSV`, `.JSON` 매칭 구조를 파악한다.
- JSON을 기준으로 `manifest.csv`를 생성한다.
- 부분방전 시계열 CSV를 로드하고 기본 통계 및 시퀀스 구조를 분석한다.
- 시계열 baseline 모델을 학습한다.
- 시계열 모델의 분류 성능을 측정한다.

### 3.2 2차 목표

- PRPD 이미지, JSON 메타데이터, 시계열 요약 특징을 결합한 VLM 학습 데이터셋을 생성한다.
- 소형 VLM을 LoRA 또는 QLoRA 방식으로 파인튜닝한다.
- VLM이 부분방전 유형을 자연어 또는 JSON 형식으로 진단하도록 학습한다.

### 3.3 확장 목표

- 시계열 CSV를 그래프 이미지로 변환하고, PRPD 이미지와 함께 멀티 이미지 VLM 입력으로 사용한다.
- 전략 A와 전략 B의 성능 및 진단 품질을 비교한다.
- classification accuracy와 생성형 진단 품질을 함께 평가한다.

## 4. 범위

### 포함 범위

- 데이터 구조 분석
- manifest 생성
- 시계열 전처리
- 시계열 분류 모델 개발
- 시계열 특징 추출
- VLM instruction dataset 생성
- 소형 VLM LoRA/QLoRA 파인튜닝
- 자연어 진단 결과 생성
- 구조화된 JSON 진단 결과 생성

### 제외 범위

- ResNet, EfficientNet 등 단일 비전 모델을 처음부터 개발하는 작업
- PRPD 이미지 단독 분류 모델을 메인 성과로 삼는 작업
- 시계열 forecasting 문제. 즉, 과거 구간으로 미래 파형을 예측하는 작업
- 원본 CSV 전체를 VLM 프롬프트에 그대로 넣는 방식
- 대형 VLM 또는 70B급 모델 학습
- 실제 산업 현장 배포 시스템 구축

## 5. 사용자 시나리오

### 시나리오 1: 시계열 기반 부분방전 분류

사용자는 부분방전 시계열 CSV 파일을 입력한다. 모델은 해당 신호를 분석하여 정상, 노이즈, 표면 방전, 코로나 방전, 보이드 방전 중 하나로 분류한다.

예상 출력:

```json
{
  "label_id": 3,
  "label_name": "코로나 방전",
  "confidence": 0.87
}
```

### 시나리오 2: VLM 기반 자연어 진단

사용자는 PRPD 이미지, 설비 메타데이터, 환경 메타데이터, 시계열 요약 특징을 입력한다. VLM은 이를 종합하여 진단 결과를 자연어로 출력한다.

예상 출력:

```text
진단 결과는 코로나 방전입니다. PRPD 이미지에서 특정 위상 구간의 방전 패턴이 관찰되며,
시계열 신호에서도 반복적인 피크가 나타납니다. 절연 상태 점검과 지속적인 모니터링이 필요합니다.
```

### 시나리오 3: VLM 기반 구조화 진단

VLM은 자연어 대신 시스템 연동이 쉬운 JSON 형식으로 결과를 출력한다.

예상 출력:

```json
{
  "diagnosis": "코로나 방전",
  "label_id": 3,
  "risk_level": "주의",
  "reason": "PRPD 패턴과 시계열 요약 특징이 코로나 방전 특성과 일치합니다.",
  "recommended_action": "고전압 절연 부위를 점검하고 방전 신호 증가 여부를 모니터링하세요."
}
```

## 6. 모델 개발 전략

이 프로젝트는 다음 순서로 진행한다.

```text
1단계: 데이터 구조 확인 및 manifest 생성
2단계: 시계열 단독 모델 개발
3단계: 시계열 요약 특징 추출
4단계: 전략 A 기반 VLM baseline 구축
5단계: 전략 B 기반 멀티 이미지 VLM 확장 실험
6단계: 전략 A와 전략 B 비교
```

## 7. 시계열 모델 전략

입력:

```text
부분방전 시계열 CSV
```

출력:

```text
부분방전 유형 라벨 0~4
```

이 프로젝트의 시계열 모델 트랙은 forecasting이 아니라 classification이다.

```text
Forecasting 아님:
과거 시계열 -> 미래 시계열 예측

Classification:
전체 부분방전 시계열 -> 방전 유형 5-class 분류
```

따라서 TimesFM, Chronos, Lag-Llama처럼 미래 파형 예측에 특화된 foundation model은 프로젝트 범위에서 제외한다. 대신 부분방전 유형 분류에 직접 사용할 수 있는 시계열 분류 모델과 downstream fine-tuning이 가능한 시계열 foundation model을 우선한다.

현재 Train 데이터 기준 CSV 하나의 입력 형태는 다음과 같다.

```text
sample shape = (20, 7680)
```

여기서 `20`축은 실제 센서 채널 수라고 단정하지 않는다. JSON의 `recording_time_length`가 20이고 센서 타입은 `HFCT` 또는 `UHF`로 기록되어 있으므로, 초기 해석은 다음과 같이 둔다.

```text
20 rows = 20개 측정 구간 또는 segment
7680 columns = 각 segment의 time points
```

따라서 모델 입력에서는 `20`축을 `pseudo-channel` 또는 `segment dimension`으로 취급한다. 이 결정은 모델 구현을 위한 실용적 가정이며, 추후 데이터 명세를 추가 확인하면 더 정확히 보정한다.

모델 구현에서는 일반적으로 다음 두 형태 중 하나로 변환해 사용한다.

```text
Conv/TCN 계열: (batch, pseudo_channels, time) = (B, 20, 7680)
RNN/Hugging Face PatchTST/일부 Transformer 계열: (batch, time, pseudo_channels) = (B, 7680, 20)
```

### 시계열 모델 실험 라인업

이 프로젝트에서는 VLM 개발 전에 시계열 분류 모델만 먼저 실험한다. 사용자는 시계열 프로젝트 경험이 많지 않으므로, 처음부터 너무 많은 모델을 동시에 구현하지 않는다. 먼저 핵심 모델을 통해 시계열 분류 파이프라인을 완성하고, 이후 확장 실험으로 넓혀간다.

모델은 다음 세 그룹으로 구분한다.

```text
1. Non-Transformer 모델
2. Transformer / Modern SOTA 모델
3. Foundation / Pretrained 모델
```

### Core Experiments

처음 구현할 핵심 실험 세트다. 이 5개 모델만 제대로 비교해도 RNN, CNN, Transformer, SOTA, Foundation 흐름을 모두 경험할 수 있다.

| 그룹 | 모델 | 목적 |
| --- | --- | --- |
| Non-Transformer | GRU | RNN 계열 기본 baseline |
| Non-Transformer | InceptionTime | 시계열 분류에서 강한 CNN/Inception 계열 baseline |
| Transformer / Modern SOTA | PatchTST | patch 기반 Transformer 분류 모델 |
| Transformer / Modern SOTA | TimesNet | 1D 시계열을 2D temporal variation으로 변환하는 SOTA 계열 모델 |
| Foundation / Pretrained | MOMENT | 사전학습된 시계열 foundation model을 downstream classification에 fine-tuning |

Core 실험 순서:

```text
1. GRU Classifier
2. InceptionTime Classifier
3. PatchTST Classifier
4. TimesNet Classifier
5. MOMENT Fine-tuning
```

### Extended Experiments

Core 실험이 안정적으로 끝난 뒤 추가로 실험할 후보들이다. 처음부터 모두 구현하지 않고, 시간과 GPU 여유가 있을 때 확장한다.

이 프로젝트의 기본 훈련 정책은 CUDA GPU 기반이다. 또한 `AGENT.md`의 실행 단위 정책에 따라 `train.py`는 한 번 실행할 때 정확히 하나의 모델만 훈련해야 한다. 따라서 `core`, `extended`, `all`, `cpu_only` 같은 그룹명은 `--model` 값으로 지원하지 않고, 실제 훈련 명령에는 `gru`, `patchtst`, `moment`처럼 구체적인 단일 모델명만 사용한다. `MiniROCKET`은 Extended 성격의 추가 실험 후보가 맞지만, `sktime/sklearn` 기반 CPU classical baseline이므로 GPU 학습 라인업과 분리해 optional로 둔다.

| 그룹 | 모델 | 목적 |
| --- | --- | --- |
| Non-Transformer | TCN | dilated causal convolution 기반의 RNN 대체 baseline |
| Non-Transformer | ResNet1D | 1D residual convolution 기반 baseline |
| Transformer / Modern SOTA | iTransformer | 변수/채널축을 token처럼 다루는 inverted attention 모델 |
| Transformer / Modern SOTA | TimeMixer | 다해상도 분해와 mixing 기반 최신 시계열 모델 |
| Foundation / Pretrained | UniTS | classification을 포함한 다중 시계열 task를 지원하는 unified model |
| Foundation / Pretrained | GPT4TS / One-Fits-All | GPT-2 계열 pretrained LM을 시계열 분석에 재활용하는 모델 |
| Representation Learning | TS2Vec | self-supervised 시계열 representation 학습 후 downstream classifier 적용 |

Optional CPU-only Extended baseline:

| 그룹 | 모델 | 목적 |
| --- | --- | --- |
| Classical Baseline | MiniROCKET | 딥러닝 없이 random convolution feature와 RidgeClassifier로 비교하는 강한 시계열 분류 baseline |

Extended 실험 우선순위:

```text
1. iTransformer
2. TCN
3. TimeMixer
4. UniTS
5. GPT4TS / One-Fits-All
6. TS2Vec
7. ResNet1D
8. MiniROCKET (optional CPU-only classical baseline)
```

### 모델 그룹별 설명

#### Non-Transformer 모델

Non-Transformer 모델은 시계열 분류의 기본기와 강한 전통적 baseline을 확인하기 위한 그룹이다.

GRU는 첫 baseline으로 사용한다. LSTM과 비슷한 목적을 가지지만 파라미터 수가 적고 학습이 빠르기 때문에, 초기 기준 성능을 만들기에 적합하다.

InceptionTime은 시계열 분류에서 널리 쓰이는 강한 deep learning baseline이다. 여러 크기의 convolution filter를 병렬로 사용해 다양한 시간 스케일의 패턴을 포착한다.

TCN은 RNN 대체 baseline이다. `7680` 길이의 긴 시계열을 순차적으로 처리하는 RNN보다 convolution 기반 접근이 더 안정적일 수 있다.

MiniROCKET은 딥러닝 모델은 아니지만 시계열 분류에서 매우 강력한 baseline으로 알려져 있다. 랜덤 convolution kernel로 feature를 추출하고 간단한 classifier로 분류한다. 다만 GPU에서 backpropagation으로 학습하는 neural model이 아니므로 기본 GPU 실험 라인업에서는 제외하고, CPU-only optional 비교 실험으로 둔다.

#### Transformer / Modern SOTA 모델

Transformer / Modern SOTA 모델은 최신 시계열 아키텍처가 부분방전 분류에 유효한지 확인하기 위한 그룹이다.

PatchTST는 긴 시계열을 patch 단위로 token화하기 때문에 현재 데이터처럼 길이가 긴 센서 신호에 적합하다.

TimesNet은 시계열의 다중 주기성 및 반복 패턴을 2D 구조로 변환해 학습한다. 부분방전 신호에 반복 피크나 위상성 패턴이 있을 수 있으므로 실험 가치가 높다.

iTransformer는 변수 또는 채널 축을 token처럼 다루는 구조다. 현재 CSV의 `20`개 row를 실제 센서 채널이라고 단정할 수는 없지만, segment/pseudo-channel 간 관계를 모델링하는 실험으로 사용할 수 있다.

TimeMixer는 다해상도 분해와 mixing 기반의 최신 시계열 모델이다. classification 실험 후보로 두되, Core 실험 이후 확장 단계에서 진행한다.

#### Foundation / Pretrained 모델

Foundation / Pretrained 모델은 비전 분야의 DINOv2처럼, 대규모 데이터로 사전학습된 시계열 표현을 부분방전 downstream classification에 전이하는 실험이다.

MOMENT는 가장 먼저 사용할 foundation model이다. classification task를 지원하고, pretrained backbone 위에 classification head를 붙여 fine-tuning하는 구조가 비교적 명확하다.

UniTS는 여러 시계열 task를 하나의 모델로 처리하는 unified time-series model이다. classification task를 지원하므로 MOMENT 이후 확장 실험 후보로 둔다.

GPT4TS / One-Fits-All은 GPT-2 계열 pretrained language model을 시계열 분석에 재활용하는 접근이다. 순수 시계열 foundation model은 아니지만, LLM 기반 representation transfer 실험으로 의미가 있다.

### 최종 실험 전략

초기에는 Core Experiments만 구현한다.

```text
GRU -> InceptionTime -> PatchTST -> TimesNet -> MOMENT
```

Core 실험이 끝난 뒤, 시간이 허용되면 Extended Experiments를 추가한다.

```text
iTransformer -> TCN -> TimeMixer -> UniTS -> GPT4TS -> TS2Vec -> ResNet1D
```

MiniROCKET은 Extended 후보이지만 CPU-only classical baseline이므로, GPU 실험 결과가 어느 정도 정리된 뒤 선택적으로 비교한다.

이 방식은 시계열 프로젝트 경험이 부족한 상태에서도 학습 곡선을 관리하면서, 포트폴리오에는 충분히 넓은 모델 비교를 보여줄 수 있다.

시계열 전처리에서 확인할 내용:

- CSV 컬럼 구조
- 샘플별 시퀀스 길이
- 결측치 여부
- 값의 스케일
- 정규화 필요 여부
- padding 또는 truncation 필요 여부
- train/valid/test split 유지 여부

추출할 수 있는 시계열 특징:

- 평균
- 표준편차
- 최솟값
- 최댓값
- peak-to-peak
- RMS
- FFT 기반 dominant frequency
- spectral energy
- 시계열 모델 예측 라벨
- 시계열 모델 confidence score

예시:

```json
{
  "ts_mean": 0.013,
  "ts_std": 0.219,
  "ts_max": 1.83,
  "ts_min": -1.76,
  "ts_rms": 0.221,
  "dominant_frequency": 60.0,
  "ts_model_prediction": "코로나 방전",
  "ts_model_confidence": 0.87
}
```

위 요약 특징은 시계열 분류 모델 자체의 학습 입력으로 필수는 아니다. 이후 VLM 단계에서 LLM 프롬프트에 제공할 보조 정보로 사용할 수 있다.

### 제외 또는 후순위 모델

다음 모델들은 유명한 시계열 foundation/forecasting 모델이지만, 현재 목표가 미래 파형 예측이 아니라 방전 유형 분류이므로 프로젝트 범위에서는 제외한다.

| 모델 | 제외/후순위 이유 |
| --- | --- |
| TimesFM | Google의 forecasting foundation model. 미래 시계열 예측에는 적합하지만 5-class 분류에는 직접적이지 않음 |
| Chronos | 값 양자화 기반 forecasting foundation model. 분류보다는 zero-shot forecasting 중심 |
| Lag-Llama | LLaMA 구조 기반 forecasting 모델. 분류 fine-tuning보다는 예측 태스크에 가까움 |
| Moirai | Salesforce 계열 universal forecasting foundation model. 현재 분류 트랙에는 직접적이지 않음 |
| TTM | IBM의 경량 forecasting foundation model. 미래 예측 중심이라 초기 분류 트랙에서 제외 |
| Time-MoE | MoE 기반 forecasting foundation model. zero/few-shot forecasting 중심 |
| Timer | generative pretrained time-series transformer. forecasting/generative task 중심 |
| Informer | long-term forecasting 중심 Transformer |
| Autoformer | seasonal-trend decomposition 기반 forecasting 모델 |
| FEDformer | frequency-domain forecasting 모델 |
| Pyraformer | long sequence forecasting 중심 Transformer |

이 모델들은 나중에 별도 forecasting 트랙을 만들 때만 검토한다. 현재 프로젝트에서는 forecasting 트랙을 만들지 않는다.

## 8. VLM 모델 전략

입력:

```text
PRPD 이미지
+ 설비 메타데이터
+ 환경 메타데이터
+ 시계열 요약 특징
```

출력:

```text
자연어 진단 결과 또는 구조화된 JSON 진단 결과
```

후보 모델:

- Qwen2-VL-2B-Instruct
- Qwen2-VL-7B-Instruct
- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- PaliGemma / PaliGemma 2
- LLaVA 계열 소형 VLM

초기 실험은 Qwen2-VL 또는 Qwen2.5-VL의 2B~3B급 모델을 우선 고려한다. GPU 여유가 있으면 7B급으로 확장한다.

## 9. 기존 분류 모델과 VLM의 학습 방식 차이

이 프로젝트에서 VLM을 사용하는 가장 큰 차별점은 단순히 모델 종류가 바뀌는 것이 아니다. 학습 목표와 손실 함수의 개념이 바뀐다.

기존 비전 분류 모델은 이미지를 보고 5개 클래스 중 하나의 인덱스를 맞히도록 학습한다. 반면 VLM은 이미지, 메타데이터, 시계열 요약 문맥을 입력받고, 그 다음에 생성해야 할 텍스트 토큰을 순차적으로 예측하도록 학습한다.

| 구분 | 기존 비전/시계열 분류 모델 | 소형 VLM 모델 |
| --- | --- | --- |
| 입력 | 이미지 또는 시계열 | PRPD 이미지 + 메타데이터 + 시계열 요약 정보 |
| 출력 | 클래스 확률 벡터 | 자연어 또는 JSON 형식 텍스트 |
| 출력 예시 | `[0.1, 0.0, 0.8, 0.1, 0.0]` | `진단 결과는 표면 방전입니다.` |
| 학습 목표 | 정답 클래스 인덱스 예측 | 정답 문장의 다음 토큰 예측 |
| 손실 함수 | Multi-class Cross Entropy Loss | Autoregressive Token-level Cross Entropy Loss |
| 평가 방식 | accuracy, F1-score | 라벨 일치율, JSON 파싱 성공률, 진단문 품질 |

기존 분류 모델의 학습 구조:

```text
입력 데이터
  └── Encoder
      └── Classification Head
          └── 5개 클래스 확률
              └── Cross Entropy Loss
```

VLM의 학습 구조:

```text
PRPD 이미지 + 텍스트 문맥
  └── Vision Encoder + LLM
      └── 다음 토큰 예측
          └── Token-level Cross Entropy Loss
```

예를 들어 정답 문장이 다음과 같다면:

```text
진단 결과는 코로나 방전입니다.
```

VLM은 이 문장을 한 번에 분류값으로 맞히는 것이 아니라, 다음 토큰을 순서대로 예측한다.

```text
진단 -> 결과는 -> 코로나 -> 방전입니다
```

따라서 VLM 프로젝트의 핵심은 단순한 클래스 분류가 아니라, 부분방전 진단을 설명 가능한 텍스트 생성 문제로 재정의하는 것이다. 이 점이 기존 ResNet, EfficientNet, LSTM 기반 프로젝트와 가장 큰 차별점이다.

## 10. VLM에 시계열 데이터를 넣는 방식

VLM은 이미지와 텍스트를 주 입력으로 사용하는 모델이다. 따라서 원본 CSV 전체를 VLM 프롬프트에 넣는 방식은 적절하지 않다. 시계열 CSV는 VLM이 이해하기 쉬운 형태로 변환해야 한다.

이 프로젝트에서는 두 가지 전략을 비교한다.

## 11. 전략 A: 시계열 요약 텍스트 + PRPD 이미지

가장 먼저 구현할 추천 방식이다.

시계열 CSV에서 핵심 특징을 추출한 뒤, 해당 값을 텍스트 프롬프트에 포함한다. PRPD 이미지는 VLM의 이미지 입력으로 넣고, 시계열 요약값과 설비 메타데이터는 텍스트로 함께 넣는다.

구조:

```text
PRPD 이미지
  └── Vision Encoder

시계열 요약 특징 + 설비 메타데이터 + 환경 메타데이터
  └── Text Prompt

Vision Encoder 출력 + Text Prompt
  └── Small VLM
      └── 자연어 진단 또는 JSON 진단 출력
```

예시 프롬프트:

```text
설비명: 22.9kV 배전반.
절연체 종류: 기체 절연체.
온도: 25도.
습도: 60%.
시계열 요약: RMS=0.221, max=1.83, min=-1.76, dominant_frequency=60.0Hz.
시계열 모델 예측 결과: 코로나 방전, confidence=0.87.

첨부된 PRPD 이미지와 위 정보를 종합하여 현재 부분방전 상태를 진단해줘.
```

장점:

- 구현 난이도가 낮다.
- 소형 VLM이 텍스트 수치 정보를 비교적 잘 활용할 수 있다.
- 시계열 모델 개발 결과를 VLM에 자연스럽게 연결할 수 있다.
- 학습 데이터셋 생성이 단순하다.
- 분류 정확도와 자연어 진단 품질을 함께 평가하기 좋다.

단점:

- 시계열 원신호의 세부 파형 형태는 일부 손실된다.
- feature engineering 품질에 따라 VLM 성능이 달라질 수 있다.

## 12. 전략 B: 시계열 그래프 이미지 + PRPD 이미지

두 번째 방식은 시계열 CSV를 그래프 이미지로 변환한 뒤, PRPD 이미지와 함께 VLM에 멀티 이미지 입력으로 넣는 방식이다.

구조:

```text
PRPD 이미지
  └── Vision Encoder

시계열 파형 그래프 이미지
  └── Vision Encoder

설비 메타데이터 + 환경 메타데이터
  └── Text Prompt

두 이미지의 시각 특징 + Text Prompt
  └── Small VLM
      └── 자연어 진단 또는 JSON 진단 출력
```

데이터 가공 방식:

```text
CSV 시계열 파일
  └── matplotlib 또는 seaborn으로 plot 생성
      └── timeseries_plot.png 저장
```

VLM 입력 예시:

```json
{
  "images": [
    "path/to/prpd.png",
    "path/to/timeseries_plot.png"
  ],
  "messages": [
    {
      "role": "user",
      "content": "첫 번째 이미지는 PRPD 패턴이고, 두 번째 이미지는 부분방전 시계열 파형입니다. 설비명은 22.9kV 배전반이고 온도는 25도, 습도는 60%입니다. 두 이미지를 함께 분석하여 부분방전 상태를 진단해줘."
    },
    {
      "role": "assistant",
      "content": "진단 결과는 코로나 방전입니다. PRPD 이미지에서 특정 위상 구간의 방전 분포가 관찰되며, 시계열 파형에서도 반복적인 피크 패턴이 나타납니다."
    }
  ]
}
```

장점:

- 원시 시계열의 파형 형태를 시각적으로 보존할 수 있다.
- 멀티 이미지 입력을 지원하는 VLM의 장점을 활용할 수 있다.
- 시계열 특징을 수동으로 많이 정의하지 않아도 된다.

단점:

- 시계열 그래프 생성 방식에 따라 모델이 보는 정보가 달라진다.
- 그래프 스타일, 축 범위, 해상도, 정규화 기준을 고정해야 한다.
- 이미지가 2장이 되므로 GPU 메모리 사용량이 증가한다.
- 모델과 학습 코드가 멀티 이미지 입력을 제대로 지원해야 한다.

## 13. 전략 A와 전략 B 비교

| 항목 | 전략 A: 시계열 요약 텍스트 | 전략 B: 시계열 그래프 이미지 |
| --- | --- | --- |
| 구현 난이도 | 낮음 | 중간 |
| GPU 메모리 | 상대적으로 적음 | 더 많이 필요 |
| 시계열 원형 보존 | 낮음 | 높음 |
| feature engineering 의존도 | 높음 | 낮음 |
| VLM 멀티 이미지 의존도 | 낮음 | 높음 |
| 추천 순서 | 1차 실험 | 2차 확장 실험 |

## 14. VLM 학습 데이터 형태

전략 A 기준 학습 데이터 예시:

```json
{
  "image": "path/to/prpd.png",
  "messages": [
    {
      "role": "user",
      "content": "설비명: 22.9kV 배전반. 절연체 종류: 기체 절연체. 온도: 25도. 습도: 60%. 시계열 요약: RMS=0.221, max=1.83, dominant_frequency=60.0Hz. PRPD 이미지와 메타데이터를 바탕으로 부분방전 상태를 진단해줘."
    },
    {
      "role": "assistant",
      "content": "진단 결과는 코로나 방전입니다. PRPD 이미지와 설비 운전 조건을 종합했을 때 코로나 방전 패턴이 나타납니다. 고전압 절연 부위 점검과 지속적인 모니터링이 필요합니다."
    }
  ]
}
```

전략 B 기준 학습 데이터 예시:

```json
{
  "images": [
    "path/to/prpd.png",
    "path/to/timeseries_plot.png"
  ],
  "messages": [
    {
      "role": "user",
      "content": "첫 번째 이미지는 PRPD 패턴이고, 두 번째 이미지는 부분방전 시계열 파형입니다. 설비 메타데이터와 두 이미지를 함께 분석하여 부분방전 상태를 진단해줘."
    },
    {
      "role": "assistant",
      "content": "진단 결과는 코로나 방전입니다. PRPD 이미지의 방전 분포와 시계열 파형의 반복 피크가 코로나 방전 특성과 일치합니다."
    }
  ]
}
```

구조화된 출력 예시:

```json
{
  "diagnosis": "코로나 방전",
  "label_id": 3,
  "risk_level": "주의",
  "reason": "PRPD 패턴과 시계열 요약 특징이 코로나 방전 특성과 일치합니다.",
  "recommended_action": "고전압 절연 부위를 점검하고 방전 신호 증가 여부를 모니터링하세요."
}
```

## 15. 학습 방식

VLM 전체를 full fine-tuning하는 것은 비용이 크므로 PEFT 기반 학습을 우선한다.

우선순위:

1. LLM 계층에 LoRA 적용
2. projection layer 학습 또는 LoRA 적용
3. GPU 여유가 있으면 vision encoder 일부 attention layer에 LoRA 적용

메모리 제약이 큰 경우:

```text
vision encoder freeze
+ language/projection 계층 LoRA
```

메모리 여유가 있는 경우:

```text
vision encoder 일부 LoRA
+ projection layer 학습
+ language 계층 LoRA
```

## 16. 평가 지표

시계열 모델 평가:

- Accuracy
- F1-score
- Confusion Matrix
- 클래스별 Precision/Recall

VLM 평가:

- 라벨 추출 후 Accuracy 계산
- JSON 출력 파싱 성공률
- label_id 일치율
- 진단 문장 품질 수동 평가
- hallucination 여부 확인
- 메타데이터 반영 여부 확인

## 17. 초기 개발 순서

1. AI-Hub 샘플 데이터를 다운로드한다.
2. 실제 폴더명과 JSON 예시를 확인한다.
3. JSON 파일을 기준으로 `manifest.csv`를 생성한다.
4. CSV 시계열 파일의 컬럼 구조와 길이를 확인한다.
5. 시계열 데이터 전처리 코드를 작성한다.
6. 1D-CNN 또는 GRU baseline 모델을 학습한다.
7. 시계열 요약 특징을 추출한다.
8. 전략 A 형식의 VLM instruction dataset을 만든다.
9. 소형 VLM을 LoRA 또는 QLoRA로 파인튜닝한다.
10. 전략 B를 위한 시계열 그래프 이미지를 생성한다.
11. 멀티 이미지 VLM 실험을 진행한다.
12. 전략 A와 전략 B의 성능 및 진단 품질을 비교한다.

## 18. 성공 기준

1차 성공 기준:

- 실제 데이터 구조를 기반으로 manifest 생성 완료
- 시계열 CSV 로딩 및 전처리 가능
- 시계열 baseline 모델 학습 가능
- validation accuracy 측정 가능

2차 성공 기준:

- VLM instruction dataset 생성 완료
- 소형 VLM LoRA 학습 가능
- PRPD 이미지와 시계열 요약 정보를 함께 사용해 진단 출력 가능
- 출력에서 라벨 ID 또는 라벨명을 안정적으로 추출 가능

확장 성공 기준:

- 시계열 그래프 이미지 생성 가능
- PRPD 이미지와 시계열 그래프 이미지를 동시에 입력하는 VLM 실험 가능
- 전략 A와 전략 B의 결과 비교 가능

## 19. 설계 메모

- VLM의 vision encoder가 PRPD 이미지를 처음부터 잘 이해한다고 가정하면 안 된다.
- 다만 pretrained vision encoder는 점, 선, 밀도, 분포, 대칭성 같은 기본 시각 특징을 이미 추출할 수 있다.
- PRPD 도메인에 맞게 LoRA/PEFT로 가볍게 적응시키는 전략이 적합하다.
- 원본 CSV 전체를 VLM 프롬프트에 넣지 않는다.
- 시계열 데이터는 별도 모델 또는 feature extractor로 압축한 뒤 VLM에 제공한다.
- 이 프로젝트의 핵심은 단일 비전 모델 개발이 아니라 시계열 모델링과 VLM 기반 설명 가능한 진단이다.
