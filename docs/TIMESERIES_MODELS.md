# 시계열 모델 후보 정리

이 문서는 부분방전 시계열 분류 트랙에서 실험할 모델 후보를 정리한다.

현재 프로젝트의 시계열 태스크는 forecasting이 아니라 classification이다.

```text
입력: 부분방전 CSV 시계열 (20, 7680)
출력: 5-class 부분방전 유형
```

라벨:

| 라벨 ID | 라벨명 |
| --- | --- |
| 0 | 정상 |
| 1 | 노이즈 |
| 2 | 표면 방전 |
| 3 | 코로나 방전 |
| 4 | 보이드 방전 |

## 모델 그룹

실험 모델은 다음 세 그룹으로 나눈다.

```text
1. Non-Transformer 모델
2. Transformer / Modern SOTA 모델
3. Foundation / Pretrained 모델
```

처음에는 Core 모델만 실험하고, 이후 Extended 모델로 확장한다.

## Core Experiments

| 그룹 | 모델 | 핵심 목적 |
| --- | --- | --- |
| Non-Transformer | GRU | RNN baseline |
| Non-Transformer | InceptionTime | 강한 CNN 기반 시계열 분류 baseline |
| Transformer / Modern SOTA | PatchTST | patch 기반 Transformer |
| Transformer / Modern SOTA | TimesNet | 1D 시계열을 2D temporal variation으로 변환 |
| Foundation / Pretrained | MOMENT | pretrained time-series foundation model fine-tuning |

## Extended Experiments

| 그룹 | 모델 | 핵심 목적 |
| --- | --- | --- |
| Non-Transformer | TCN | dilated causal convolution baseline |
| Non-Transformer | ResNet1D | residual 1D CNN baseline |
| Non-Transformer | MiniROCKET | 강한 classical 시계열 분류 baseline |
| Transformer / Modern SOTA | iTransformer | 변수/채널축 attention |
| Transformer / Modern SOTA | TimeMixer | 다해상도 mixing 기반 최신 시계열 모델 |
| Foundation / Pretrained | UniTS | unified multi-task time-series model |
| Foundation / Pretrained | GPT4TS / One-Fits-All | GPT-2 pretrained LM을 시계열에 전이 |
| Representation Learning | TS2Vec | self-supervised 시계열 표현 학습 |

## 입력 형태

현재 CSV 한 개는 다음 형태다.

```text
raw CSV shape = (20, 7680)
```

여기서 `20`축을 실제 센서 채널 수로 단정하지 않는다. JSON의 `recording_time_length`가 20이고 센서 타입은 보통 `HFCT` 또는 `UHF`로 기록되어 있으므로, 현재 단계에서는 다음처럼 해석한다.

```text
20 rows = 20개 측정 구간 또는 segment
7680 columns = 각 segment의 time points
```

따라서 모델 입력에서는 `20`축을 실제 physical channel이 아니라 `pseudo-channel` 또는 `segment dimension`으로 사용한다.

모델별로 요구하는 입력 형태가 다를 수 있다.

```text
Conv/TCN/Patch 계열:
(batch, pseudo_channels, time) = (B, 20, 7680)

RNN/일부 Transformer 계열:
(batch, time, pseudo_channels) = (B, 7680, 20)
```

따라서 dataloader에서 모델별 `transpose` 처리가 필요하다.

## 1. GRU

그룹:

```text
Non-Transformer / RNN baseline
```

GRU(Gated Recurrent Unit)는 LSTM을 간소화한 RNN 계열 모델이다. reset gate와 update gate를 사용해 과거 정보를 유지하거나 갱신한다.

우리 프로젝트에서의 역할:

```text
가장 기본적인 딥러닝 시계열 분류 baseline
```

장점:

- LSTM보다 구조가 단순하다.
- 파라미터 수가 적고 학습이 빠르다.
- 첫 시계열 baseline으로 구현하기 좋다.

주의점:

- 길이 `7680`의 긴 시계열을 순차 처리하므로 학습이 느릴 수 있다.
- long-range dependency를 완벽히 잡기 어렵다.

예상 입력:

```text
(B, 7680, 20)
```

## 2. InceptionTime

그룹:

```text
Non-Transformer / CNN 기반 강한 baseline
```

InceptionTime은 이미지 분야의 Inception 구조를 시계열 분류에 적용한 모델이다. 여러 크기의 1D convolution filter를 병렬로 사용해 다양한 시간 스케일의 패턴을 포착한다.

우리 프로젝트에서의 역할:

```text
GRU보다 강한 non-Transformer 시계열 분류 baseline
```

장점:

- 시계열 분류에서 널리 쓰이는 강한 baseline이다.
- local pattern, peak, short-term variation 포착에 강하다.
- RNN보다 병렬화가 쉽다.

주의점:

- 긴 전역 의존성은 Transformer 계열보다 약할 수 있다.
- kernel size와 depth에 따라 메모리 사용량이 달라진다.

예상 입력:

```text
(B, 20, 7680)
```

## 3. PatchTST

그룹:

```text
Transformer / Modern SOTA
```

PatchTST는 긴 시계열을 작은 patch 단위로 나누고, 각 patch를 token처럼 Transformer에 입력하는 모델이다. ViT가 이미지를 patch로 나누는 것과 유사한 아이디어를 시계열에 적용한다.

우리 프로젝트에서의 역할:

```text
긴 부분방전 시계열에 적합한 Transformer baseline
```

장점:

- 길이 `7680`의 긴 시계열을 patch로 줄여 효율적으로 처리할 수 있다.
- Transformer 기반 모델 중 실험 가치가 높다.
- channel-independent 전략을 적용하기 쉽다.

주의점:

- patch length, stride 설정이 중요하다.
- classification head를 별도로 구성해야 할 수 있다.

예상 입력:

```text
(B, 20, 7680)
```

## 4. TimesNet

그룹:

```text
Transformer / Modern SOTA
```

TimesNet은 1D 시계열을 2D temporal variation 형태로 변환한 뒤, 2D convolution 기반 구조로 시계열 패턴을 학습한다. 시계열의 다중 주기성과 반복 패턴을 포착하는 것이 핵심이다.

우리 프로젝트에서의 역할:

```text
부분방전 신호의 반복 피크, 주기성, 위상성 패턴을 포착하는 SOTA 계열 모델
```

장점:

- 주기적 패턴이 있는 시계열에 강하다.
- 1D 신호를 2D 구조로 변환해 더 풍부한 패턴을 볼 수 있다.
- THU Time-Series-Library에서 여러 task로 자주 사용된다.

주의점:

- 구현체에 따라 입력 shape과 task 설정이 까다로울 수 있다.
- classification 설정을 명확히 맞춰야 한다.

예상 입력:

```text
(B, 7680, 20)
```

## 5. MOMENT

그룹:

```text
Foundation / Pretrained
```

MOMENT는 대규모 시계열 데이터로 사전학습된 time-series foundation model이다. masked patch reconstruction 방식으로 사전학습된 backbone을 사용하고, downstream task에 맞는 head를 붙여 fine-tuning할 수 있다.

우리 프로젝트에서의 역할:

```text
DINOv2처럼 pretrained representation을 부분방전 분류에 전이하는 핵심 foundation model
```

장점:

- classification task에 사용할 수 있다.
- pretrained backbone을 활용할 수 있다.
- head-only training, partial fine-tuning, full fine-tuning 등 전략을 비교할 수 있다.

주의점:

- 모델이 기대하는 patch length, sequence length, channel 설정을 맞춰야 한다.
- 전체 fine-tuning은 GPU 메모리를 더 많이 사용한다.

예상 입력:

```text
(B, 20, 7680)
또는 MOMENT processor/config가 요구하는 형태
```

## 6. TCN

그룹:

```text
Non-Transformer / CNN baseline
```

TCN(Temporal Convolutional Network)은 dilated causal convolution을 사용해 긴 receptive field를 확보하는 시계열 모델이다. RNN처럼 순차적으로 계산하지 않아 병렬화가 쉽다.

우리 프로젝트에서의 역할:

```text
GRU와 비교할 convolution 기반 baseline
```

장점:

- 긴 시계열에 효율적이다.
- 병렬 계산이 가능하다.
- RNN보다 안정적으로 학습될 수 있다.

주의점:

- dilation, kernel size, layer 수에 따라 receptive field가 달라진다.
- causal 구조가 꼭 필요한 태스크는 아니므로 classification용 pooling 설계가 필요하다.

예상 입력:

```text
(B, 20, 7680)
```

## 7. ResNet1D

그룹:

```text
Non-Transformer / CNN baseline
```

ResNet1D는 residual block을 1D convolution에 적용한 모델이다. 이미지 ResNet의 skip connection 아이디어를 시계열 신호에 적용한다.

우리 프로젝트에서의 역할:

```text
단순하고 안정적인 1D CNN baseline
```

장점:

- 구현이 비교적 단순하다.
- residual connection 덕분에 깊은 CNN 학습이 안정적이다.
- 부분방전 신호의 local pattern을 포착하기 좋다.

주의점:

- Transformer 계열보다 전역 의존성 모델링은 약할 수 있다.
- downsampling 설계에 따라 정보 손실이 발생할 수 있다.

예상 입력:

```text
(B, 20, 7680)
```

## 8. MiniROCKET

그룹:

```text
Non-Transformer / classical strong baseline
```

MiniROCKET은 랜덤 convolution kernel을 사용해 시계열 feature를 빠르게 추출한 뒤, RidgeClassifier 같은 간단한 classifier로 분류하는 방법이다.

우리 프로젝트에서의 역할:

```text
딥러닝 모델이 정말 필요한지 확인하는 강력한 classical baseline
```

장점:

- 매우 빠르다.
- 시계열 분류에서 강력한 baseline으로 알려져 있다.
- 학습 비용이 낮다.

주의점:

- end-to-end deep learning 모델은 아니다.
- feature extractor와 classifier가 분리된다.
- VLM으로 연결할 embedding 활용성은 deep model보다 낮을 수 있다.

예상 입력:

```text
MiniROCKET 구현체에 맞게 (samples, channels, time) 형태로 변환
```

## 9. iTransformer

그룹:

```text
Transformer / Modern SOTA
```

iTransformer는 일반적인 Transformer와 달리 시간축이 아니라 변수 또는 채널 축을 token처럼 다루는 inverted attention 구조를 사용한다.

우리 프로젝트에서의 역할:

```text
20개 segment/pseudo-channel 간 관계를 모델링하는 Transformer 계열 실험
```

장점:

- multivariate time series에 적합하다.
- segment/pseudo-channel 간 상관관계를 모델링하기 좋다.
- 우리 데이터의 `(20, 7680)` 구조와 잘 맞을 가능성이 있다.

주의점:

- 기존 forecasting 중심 구현체를 classification에 맞게 설정해야 할 수 있다.
- 데이터 shape 변환이 중요하다.

예상 입력:

```text
(B, 7680, 20)
```

## 10. TimeMixer

그룹:

```text
Transformer / Modern SOTA
```

TimeMixer는 다해상도 분해와 mixing 구조를 활용하는 최신 시계열 모델이다. Transformer attention보다는 MLP/mixing 기반 접근에 가깝다.

우리 프로젝트에서의 역할:

```text
Transformer 외 최신 SOTA 계열 모델 비교
```

장점:

- 다해상도 패턴을 다루기 좋다.
- attention 기반 모델과 다른 inductive bias를 가진다.
- 최신 시계열 모델군과 비교하는 의미가 있다.

주의점:

- 원래 forecasting 중심 구현체인 경우 classification 설정을 맞춰야 한다.
- Core 실험 이후 확장 후보로 둔다.

예상 입력:

```text
구현체에 따라 (B, 7680, 20) 또는 (B, 20, 7680)
```

## 11. UniTS

그룹:

```text
Foundation / Pretrained
```

UniTS는 forecasting, classification, imputation, anomaly detection 등 여러 시계열 태스크를 하나의 모델 구조로 처리하려는 unified time-series model이다. 태스크별 prompt를 사용해 모델이 수행할 작업을 구분한다.

우리 프로젝트에서의 역할:

```text
classification task를 지원하는 unified foundation-style model 비교
```

장점:

- classification task를 지원한다.
- 멀티태스크 구조라 확장성이 있다.
- MOMENT 이후 foundation 계열 비교 대상으로 좋다.

주의점:

- MOMENT보다 적용 난이도가 높을 수 있다.
- task prompt, dataloader, config 설정을 맞춰야 한다.
- 새로운 데이터셋 포맷 변환 작업이 필요할 수 있다.

예상 입력:

```text
(B, 7680, 20)
또는 UniTS dataloader/config가 요구하는 형태
```

## 12. GPT4TS / One-Fits-All

그룹:

```text
Foundation / Pretrained / LLM transfer
```

GPT4TS 또는 One-Fits-All은 GPT-2 같은 pretrained language model의 Transformer block을 시계열 분석에 재활용하는 접근이다. 입력/출력 projection과 일부 normalization 계층만 학습해 시계열 task에 적응시키는 것이 핵심이다.

우리 프로젝트에서의 역할:

```text
LLM 사전학습 표현을 시계열 분류에 전이하는 실험
```

장점:

- LLM 계열 representation transfer를 경험할 수 있다.
- 학습 파라미터 수를 줄이는 실험이 가능하다.
- VLM/LLM 프로젝트 경험과 연결되는 흥미로운 시계열 실험이다.

주의점:

- 시계열 전용 foundation model은 아니다.
- 채널 독립 처리로 다변량 상관을 충분히 포착하지 못할 수 있다.
- 구현 난이도가 있다.

예상 입력:

```text
patch/token embedding 후 GPT 계열 backbone 입력
```

## 13. TS2Vec

그룹:

```text
Representation Learning
```

TS2Vec은 self-supervised 방식으로 시계열 representation을 학습한 뒤, downstream task에 classifier를 붙이는 모델이다.

우리 프로젝트에서의 역할:

```text
라벨 의존도를 줄인 representation learning 실험
```

장점:

- self-supervised 시계열 표현 학습을 경험할 수 있다.
- 사전학습 후 분류 head를 붙여 downstream classification을 수행할 수 있다.
- foundation model 전 단계의 representation learning 실험으로 좋다.

주의점:

- end-to-end supervised classifier보다 실험 단계가 늘어난다.
- representation 품질 평가 방식이 필요하다.

예상 입력:

```text
(B, 7680, 20)
또는 구현체 요구 형태
```

## 추천 실험 순서

처음에는 Core 실험만 진행한다.

```text
1. GRU
2. InceptionTime
3. PatchTST
4. TimesNet
5. MOMENT
```

이후 확장 실험은 다음 순서로 진행한다.

```text
1. iTransformer
2. TCN
3. MiniROCKET
4. TimeMixer
5. UniTS
6. GPT4TS / One-Fits-All
7. TS2Vec
8. ResNet1D
```

## 최종 목표

시계열 분류 트랙의 최종 목표는 모델별 성능을 비교하고, 가장 좋은 시계열 모델의 결과를 VLM 단계에 연결하는 것이다.

VLM에 전달할 수 있는 정보:

```text
시계열 모델 예측 라벨
시계열 모델 confidence
class probability
hidden embedding
통계 feature
```

VLM 입력 예시:

```text
PRPD 이미지
+ JSON 메타데이터
+ 시계열 모델 예측 결과
+ 시계열 요약 feature
```
