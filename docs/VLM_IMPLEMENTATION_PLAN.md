# VLM 구현 계획

## 목표

부분방전 프로젝트의 VLM 단계는 PRPD 이미지 단독 분류가 아니라, 다음 정보를 함께 입력받아 구조화된 진단 JSON을 생성하는 모델을 개발하는 것이다.

```text
PRPD 이미지
+ 안전한 설비/환경 메타데이터
+ 시계열 모델 예측 결과
+ 시계열 요약 feature
-> 경량 pretrained VLM
-> 진단 JSON
```

GPU는 RTX 4060 Laptop 8GB 기준이므로 대형 VLM full fine-tuning은 제외하고, 경량 pretrained VLM에 LoRA/QLoRA 방식으로 미세조정한다.

## 모델 선택

### 1순위: Qwen/Qwen3-VL-2B-Instruct

8GB GPU에서 가장 먼저 검증할 모델이다.

선택 이유:

- 2B급이라 8GB 환경에서 가장 현실적이다.
- image-text-to-text VLM으로 PRPD 이미지 + 텍스트 메타데이터 입력 구조와 맞는다.
- Qwen 계열은 instruction following과 JSON 출력 형식에 비교적 강하다.
- LoRA/QLoRA SFT 실험 대상으로 적합하다.

### 안정 대안: Qwen/Qwen2.5-VL-3B-Instruct

Qwen3-VL-2B가 로컬 환경에서 불안정하거나 품질이 부족할 때 비교할 후보이다.

주의:

- 3B급이라 8GB에서 더 빡빡하다.
- 반드시 4bit QLoRA, batch size 1, gradient accumulation, gradient checkpointing을 사용한다.

### 후순위 위험 후보: Qwen/Qwen3-VL-4B-Instruct

2B/3B smoke가 통과한 뒤에만 검토한다.

주의:

- 8GB에서는 OOM 가능성이 높다.
- 첫 구현 대상이 아니다.
- 4bit QLoRA, 낮은 LoRA rank, 이미지 해상도 제한이 필요하다.

### fallback 후보

- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`: Qwen 계열이 메모리나 설치 문제로 막힐 때 사용
- `google/paligemma2-3b-mix-224`: 이미지-텍스트 전이 후보지만 한국어 JSON 진단 목적에는 Qwen 계열을 우선
- `llava-hf/llava-onevision-qwen2-0.5b-si-hf`: 파이프라인 sanity check 전용

## 입력 데이터 설계

VLM에는 raw CSV를 직접 넣지 않는다. 입력은 이미지 1개와 텍스트 context 1개로 구성한다.

### Strategy A: 기본 입력

```text
image:
  manifest.image_path의 PRPD PNG

text:
  설비 정보
  환경 정보
  시계열 모델 분석 결과
  JSON 출력 지시문
```

### 이미지 입력

```text
Train/manifest.csv의 image_path
-> PRPD PNG
-> VLM processor의 image input
```

`image_path` 문자열 자체는 prompt text에 넣지 않는다. 파일 경로나 파일명에 라벨명이 포함될 수 있기 때문이다.

### 텍스트 입력에 포함할 안전한 메타데이터

```text
equipment_name
equipment_rated_voltage
equipment_rated_current
insulator_type / insulator_name
sensor_type
temperature
humidity
clearance_distance
```

### 텍스트 입력에 포함할 시계열 모델 정보

시계열 모델 실험이 끝난 뒤 다음 정보를 별도 CSV로 export해서 VLM dataset builder에서 join한다.

```text
sample_id
ts_model_name
ts_pred_label_id
ts_confidence
ts_prob_0
ts_prob_1
ts_prob_2
ts_prob_3
ts_prob_4
rms
std
abs_p99
pulse_rate
spectral_energy
```

시계열 모델 결과가 아직 없을 때는 smoke dataset에서 “시계열 모델 결과 없음”으로 명시하고, feature-only context만 사용한다.

### prompt에 넣으면 안 되는 항목

```text
label_id
label_name
PD_type
sample_id
image_path 문자열
timeseries_path 문자열
json_path 문자열
파일명
defect_details
defect_nums
max_discharge_value
raw CSV 값 전체
```

`label_id`는 정답 target과 평가에는 사용하지만, user prompt text에는 절대 넣지 않는다.

## 출력 형식

초기 VLM은 자연어보다 strict JSON 출력을 우선한다.

```json
{
  "label_id": 3,
  "diagnosis": "코로나방전",
  "risk_level": "주의",
  "reason": "PRPD 패턴과 시계열 요약 특징이 코로나 방전 특성과 일치합니다.",
  "recommended_action": "고전압 절연 부위를 점검하고 방전 신호 증가 여부를 모니터링하세요."
}
```

평가 항목:

- JSON parse success rate
- schema validity
- label accuracy
- macro F1
- confusion matrix
- hallucinated field count
- forbidden prompt field leakage count

## 학습 방식

### 기본 방식

```text
pretrained VLM
-> 4bit QLoRA
-> SFT
-> strict JSON diagnosis generation
```

### 8GB 기본 설정

```yaml
model_id: Qwen/Qwen3-VL-2B-Instruct
quantization: 4bit_nf4
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: all-linear
train_vision_tower: false
train_projector: false
batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
max_length: null
image_max_pixels: 512x512
flash_attention: false
```

처음에는 vision tower를 학습하지 않는다. PRPD 이미지는 pretrained vision encoder가 기본적인 점, 선, 밀도, 분포 패턴을 추출하도록 두고, language layer에 LoRA를 적용해 JSON 진단 형식을 학습시킨다.

## 구현 순서

### 1. VLM 입력 계약 정의

생성 파일:

```text
vlm/src/schema.py
vlm/src/prompts.py
vlm/tests/test_prompts.py
```

할 일:

- safe metadata whitelist 정의
- forbidden prompt field 정의
- target JSON schema 정의
- image + text message format 정의
- prompt leakage test 작성

### 2. 시계열 context export

생성 파일:

```text
vlm/scripts/export_ts_context.py
```

할 일:

- `Train/manifest.csv` 기준으로 sample별 feature/context 생성
- 시계열 모델 결과가 있으면 prediction/probability join
- raw CSV 배열과 path 문자열은 export하지 않음

### 3. VLM instruction dataset 생성

생성 파일:

```text
vlm/scripts/build_instruction_dataset.py
vlm/scripts/validate_instruction_dataset.py
vlm/tests/test_instruction_dataset.py
```

출력:

```text
results/vlm/instruction_smoke/train.jsonl
results/vlm/instruction_smoke/valid.jsonl
results/vlm/instruction_smoke/summary.json
```

JSONL row 구조:

```json
{
  "sample_id": "...",
  "split": "train",
  "images": ["Train/.../sample.png"],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "Train/.../sample.png"},
        {"type": "text", "text": "...JSON만 출력하세요..."}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"label_id\":3,\"diagnosis\":\"코로나방전\",...}"
    }
  ]
}
```

### 4. Qwen3-VL-2B inference smoke

생성 파일:

```text
vlm/scripts/run_inference.py
```

검증 명령:

```powershell
python vlm/scripts/run_inference.py `
  --dataset results/vlm/instruction_smoke/valid.jsonl `
  --index 0 `
  --model-id Qwen/Qwen3-VL-2B-Instruct `
  --load-in-4bit `
  --output results/vlm/inference_smoke.json
```

성공 기준:

- 모델 로드 성공
- 이미지 + 텍스트 입력 처리 성공
- raw output 저장
- JSON parse 성공 또는 parse error 명확히 기록
- CUDA peak memory 기록

### 5. QLoRA SFT smoke

생성 파일:

```text
vlm/scripts/train_sft.py
vlm/configs/qwen3_vl_2b_smoke.yaml
```

검증 명령:

```powershell
python vlm/scripts/train_sft.py `
  --config vlm/configs/qwen3_vl_2b_smoke.yaml `
  --max-steps 10
```

성공 기준:

- 10 step smoke training 완료
- adapter checkpoint 저장
- training summary 저장
- OOM 발생 시 8GB fallback 설정 기록

### 6. JSON 평가

생성 파일:

```text
vlm/scripts/evaluate_outputs.py
```

평가 명령:

```powershell
python vlm/scripts/evaluate_outputs.py `
  --predictions results/vlm/predictions_smoke.jsonl `
  --output results/vlm/eval_smoke.json
```

필수 metric:

```text
json_parse_success_rate
schema_validity_rate
label_accuracy
macro_f1
confusion_matrix
forbidden_field_hit_count
hallucinated_field_count
```

## 실험 단계

### Stage 0: zero-shot / few-shot

```text
sample: 20개
training: 없음
목적: processor, prompt, JSON output 가능성 확인
```

### Stage 1: LoRA smoke

```text
sample: 10~100개
model: Qwen3-VL-2B
training: LoRA SFT
목적: 8GB에서 학습 루프 동작 확인
```

### Stage 2: QLoRA small

```text
sample: 500~2,000개
model: Qwen3-VL-2B
training: 4bit QLoRA SFT
목적: JSON parse rate와 label accuracy 확인
```

### Stage 3: Qwen2.5-VL-3B 비교

```text
sample: 500~2,000개
model: Qwen2.5-VL-3B
training: 4bit QLoRA SFT
조건: 2B가 안정적으로 돌아간 뒤
```

### Stage 4: main

```text
sample: 5,000~10,000개
model: 더 안정적인 모델 선택
조건: JSON parse success와 VRAM 안정성 확인 후
```

### Stage 5: Strategy B

```text
input: PRPD PNG + waveform/spectrogram PNG
조건: Strategy A가 먼저 성공한 뒤
```

## Strategy A/B

### Strategy A: 우선 구현

```text
PRPD 이미지 1장
+ safe metadata
+ time-series summary
-> VLM
-> JSON diagnosis
```

이 전략이 기본이다.

### Strategy B: 후순위

```text
PRPD 이미지 1장
+ 시계열 waveform/spectrogram 이미지 1장
+ safe metadata
+ time-series summary
-> VLM
-> JSON diagnosis
```

주의:

- 이미지가 2장이 되면 visual token 부담이 커진다.
- 8GB에서는 OOM 위험이 커진다.
- Strategy A가 동작한 뒤에만 시도한다.

## 최종 구현 체크리스트

- [ ] `vlm/` 디렉터리 생성
- [ ] prompt/schema 구현
- [ ] forbidden field leakage test 구현
- [ ] time-series context export 구현
- [ ] instruction dataset builder 구현
- [ ] instruction dataset validator 구현
- [ ] Qwen3-VL-2B inference smoke 구현
- [ ] QLoRA SFT smoke 구현
- [ ] JSON evaluation 구현
- [ ] `docs/VLM_STRATEGY.md`의 모델 우선순위 업데이트
- [ ] `docs/VLM_DEVELOPMENT_RUNBOOK.md` 작성

## 성공 기준

- VLM 입력이 PRPD 이미지 + safe metadata + 시계열 요약으로 구성된다.
- raw CSV와 label leakage field가 prompt에 들어가지 않는다.
- Qwen3-VL-2B inference smoke가 가능하다.
- 8GB에서 LoRA/QLoRA smoke training이 가능하거나 명확한 fallback이 기록된다.
- 출력 JSON을 자동으로 parse/evaluate할 수 있다.
- Strategy A가 먼저 검증되고, Strategy B는 후순위로 남는다.
