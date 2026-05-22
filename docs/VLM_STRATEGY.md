# VLM 모델 전략

이 문서는 부분방전 프로젝트의 VLM(Vision-Language Model) 개발 방향을 정리한다.

이 프로젝트에서 VLM은 PRPD 이미지만 보고 방전 유형을 맞히는 단일 비전 모델이 아니다. 목표는 다음 멀티모달 정보를 결합해 현장 엔지니어가 이해할 수 있는 진단 결과를 생성하는 것이다.

```text
PRPD 이미지
+ JSON 메타데이터
+ 시계열 모델 예측 결과
+ 시계열 요약 feature
-> 자연어 또는 JSON 진단 리포트
```

## 기본 방향

비전 단독 분류 모델은 이 프로젝트의 핵심 범위가 아니다.

제외하는 방향:

```text
PRPD 이미지 -> ResNet/EfficientNet -> 5-class classification
```

추구하는 방향:

```text
PRPD 이미지 + 설비/환경 메타데이터 + 시계열 요약
-> Small VLM
-> 구조화된 진단 JSON 또는 자연어 진단문
```

즉, VLM 단계의 핵심은 이미지 분류가 아니라 `멀티모달 진단 리포팅`이다.

## 추천 모델

### 1순위: Qwen2.5-VL-3B-Instruct

초기 메인 후보로 사용한다.

추천 이유:

- 3B급이라 로컬 단일 GPU 실험 가능성이 비교적 높다.
- 이미지와 텍스트 instruction following에 강하다.
- 한국어 프롬프트와 한국어 응답 품질이 괜찮다.
- Hugging Face, TRL, PEFT, QLoRA 생태계에서 활용하기 좋다.
- PRPD 이미지, JSON 메타데이터, 시계열 요약값을 함께 넣는 구조와 잘 맞는다.

### Smoke 후보: Qwen2-VL-2B-Instruct

VLM 데이터셋 생성, processor, SFT pipeline 검증용으로 사용한다.

장점:

- 더 가볍다.
- 빠르게 forward, LoRA smoke training을 확인하기 좋다.
- 코드 파이프라인 검증에 적합하다.

### 확장 후보: Qwen2.5-VL-7B-Instruct

3B 실험이 안정화된 뒤 비교 후보로 사용한다.

주의점:

- RTX 4060 Laptop 8GB 환경에서는 QLoRA라도 빡빡할 수 있다.
- batch size 1, gradient accumulation, gradient checkpointing, 4bit quantization이 필요할 가능성이 높다.

### 대안 후보: PaliGemma / PaliGemma 2

분류형 VLM fine-tuning에는 좋은 후보지만, 자연어 진단 리포트 생성과 한국어 instruction 대응은 Qwen-VL 계열이 더 프로젝트 목적에 잘 맞는다.

## 입력 데이터 구성

VLM 입력은 이미지와 텍스트를 함께 사용한다.

이미지 입력:

```text
PRPD PNG 이미지
```

텍스트 입력:

```text
설비 정보
- 설비명
- 절연체 종류
- 정격 전압
- 정격 전류
- 센서 타입

환경 정보
- 온도
- 습도
- 이격 거리

시계열 분석 정보
- 시계열 모델 예측 라벨
- confidence
- class probability
- RMS
- max / min
- peak-to-peak
- dominant frequency
- spectral energy
```

원본 CSV 전체를 VLM 프롬프트에 넣지 않는다. 시계열 raw signal은 별도 시계열 모델 또는 feature extractor로 압축한 뒤 텍스트 feature로 제공한다.

## 출력 형식

초기 학습은 자연어보다 JSON 출력을 우선한다. JSON은 평가와 후처리가 쉽기 때문이다.

권장 출력 예시:

```json
{
  "label_id": 1,
  "diagnosis": "노이즈",
  "risk_level": "낮음",
  "reason": "PRPD 패턴과 시계열 특징이 실제 부분방전보다는 노이즈성 신호에 가깝습니다.",
  "recommended_action": "센서 접촉 상태와 주변 전자기 간섭 여부를 점검하세요."
}
```

평가 항목:

- `label_id` 정확도
- `diagnosis` 라벨명 일치율
- JSON 파싱 성공률
- 메타데이터 반영 여부
- 시계열 분석 정보 반영 여부
- hallucination 여부
- 진단문 품질

## 학습 방식

초기 학습은 QLoRA 기반 SFT를 우선한다.

권장 초기 설정:

```text
base_model: Qwen2.5-VL-3B-Instruct
quantization: 4bit NF4
training: SFT
vision_encoder: freeze
projector: freeze 또는 일부 LoRA
LLM: LoRA
batch_size: 1
gradient_accumulation_steps: 8~16
gradient_checkpointing: enabled
```

처음부터 vision encoder 전체를 학습하지 않는다. PRPD 이미지는 VLM이 사전학습 중 직접 본 도메인 이미지가 아닐 가능성이 높지만, pretrained vision encoder는 점, 선, 밀도, 분포, 대칭성 같은 기본 시각 특징을 추출할 수 있다.

권장 순서:

```text
1. vision encoder freeze
2. language model 계층에 LoRA 적용
3. 이미지 + 메타데이터 + 시계열 요약으로 정답 JSON 생성 학습
4. 성능 부족 시 projector 또는 vision encoder 일부 LoRA 검토
```

## 데이터 규모 전략

현재 `Train/` working dataset은 30,010개 샘플이다. VLM 연습과 초기 LoRA 실험에는 충분하다.

권장 단계:

```text
1. VLM smoke: 100~500개
2. 첫 LoRA: 2,000~5,000개
3. 메인 실험: 10,000~30,000개
4. 최종 확장: 원본 30만 개 중 일부 또는 전체
```

처음부터 30만 개 전체를 사용하지 않는다. 먼저 3만 개 working dataset으로 데이터 포맷, 학습 안정성, JSON 출력 품질을 확인한다.

## 시계열 모델과의 연결

VLM 개발은 시계열 분류 모델 실험 이후 진행한다.

시계열 트랙 산출물:

```text
best model name
predicted label
confidence
class probabilities
statistical features
optional embedding
```

VLM instruction dataset에는 위 정보를 텍스트 prompt로 포함한다.

예시 prompt:

```text
설비 정보:
- 설비명: ACSR-OC
- 절연체: 고체 / XLPE
- 정격 전압: 22900V
- 센서 타입: HFCT

환경 정보:
- 온도: 19도
- 습도: 66%

시계열 모델 분석:
- 예측 라벨: 노이즈
- confidence: 0.82
- max_discharge_value: 82
- RMS: 0.221

첨부된 PRPD 이미지와 위 정보를 종합하여 현재 부분방전 상태를 JSON으로 진단하세요.
```

## 프로젝트 스토리라인

최종 포트폴리오 흐름은 다음과 같이 잡는다.

```text
1. CSV 시계열 분류 모델 개발
2. 여러 시계열 모델 비교
3. best 시계열 모델의 예측/요약값 추출
4. PRPD 이미지 + JSON 메타데이터 + 시계열 요약을 VLM instruction dataset으로 변환
5. Qwen2.5-VL-3B-Instruct QLoRA fine-tuning
6. JSON 진단 리포트 생성 및 평가
```

핵심 메시지:

```text
단순 이미지 분류 모델이 아니라, 시계열 센서 분석 결과와 설비 메타데이터를 VLM에 연결한 설명 가능한 산업 설비 부분방전 진단 시스템
```
