# PRD: 부분방전 진단 서비스 Agent Workflow

## 1. 목적

이 문서는 향후 서비스 공정에서 사용할 `React + FastAPI + OpenAI Agents SDK` 기반 부분방전 진단 워크플로우를 정의한다. 이 단계는 코드 구현 전 계획이며, 모델 학습이 아니라 **추론 서비스 구성**을 대상으로 한다.

서비스 목표는 사용자가 PRPD 이미지, 시계열 CSV, 설비/환경 메타데이터를 입력하면 FastAPI가 Agents SDK 워크플로우를 실행하고, 기존 시계열 모델과 VLM 추론 모델을 도구로 호출해 최종 진단 리포트를 반환하는 것이다.

```text
React
-> FastAPI
-> Agents SDK workflow
-> Time-series inference tool
-> VLM inference tool
-> Guardrail / Reviewer
-> Final diagnosis report
```

## 2. 범위

### 포함

- React 입력 화면 설계 방향
- FastAPI 진단 API 설계 방향
- Agents SDK 기반 진단 워크플로우
- 시계열 모델 추론 tool 구성
- VLM 추론 tool 구성
- 입력/출력 guardrail
- trace/audit log 설계
- human review 분기 조건

### 제외

- 시계열 모델 학습
- VLM 모델 학습
- QLoRA 학습 코드 변경
- 실제 산업 현장 배포
- 사용자 인증/권한 관리
- 결제/운영 관리자 기능

## 3. 사용자 시나리오

### 시나리오 A: 정상 진단 요청

사용자는 React 화면에서 다음을 입력한다.

- PRPD PNG 이미지
- 부분방전 시계열 CSV
- 설비명
- 정격 전압/전류
- 절연체 정보
- 센서 타입
- 온도/습도
- 이격 거리

서비스는 다음을 반환한다.

```json
{
  "diagnosis_id": "diag_20260604_000001",
  "status": "completed",
  "final_label_id": 3,
  "diagnosis": "코로나 방전",
  "risk_level": "주의",
  "confidence": 0.87,
  "reason": "시계열 모델과 VLM 진단 결과가 코로나 방전 가능성을 지지합니다.",
  "recommended_action": "고전압 접속부와 전계 집중 부위를 점검하고 추세를 모니터링하세요.",
  "requires_human_review": false
}
```

### 시나리오 B: 입력 오류

CSV shape가 `(20, 7680)`이 아니거나 이미지가 PNG가 아니면 workflow를 시작하지 않는다.

```json
{
  "status": "rejected",
  "error_code": "INVALID_INPUT",
  "message": "timeseries_csv must have shape (20, 7680)."
}
```

### 시나리오 C: 낮은 신뢰도

시계열 모델과 VLM 결과가 불일치하거나 confidence가 낮으면 최종 진단을 확정하지 않고 review 상태로 반환한다.

```json
{
  "status": "needs_review",
  "requires_human_review": true,
  "reason": "시계열 모델과 VLM의 예측 라벨이 불일치합니다."
}
```

## 4. 전체 아키텍처

```text
frontend/
  React app
  - upload form
  - metadata form
  - diagnosis result view
  - trace view

service/
  FastAPI backend
  - upload endpoint
  - diagnosis endpoint
  - trace endpoint
  - Agents SDK workflow

ml/
  time-series inference code

vlm/
  VLM inference code
  prompt builder
  output evaluator
```

## 5. API 설계

### POST `/diagnose`

진단 workflow를 실행한다.

Request:

```text
multipart/form-data
- prpd_image: PNG
- timeseries_csv: CSV
- metadata: JSON string
```

Metadata JSON:

```json
{
  "equipment_name": "ACSR-OC",
  "equipment_rated_voltage": "22900V",
  "equipment_rated_current": "268A",
  "insulator_type": "고체",
  "insulator_name": "XLPE",
  "sensor_type": "HFCT",
  "temperature": 19,
  "humidity": 66,
  "clearance_distance": "1000mm"
}
```

Response:

```json
{
  "diagnosis_id": "diag_...",
  "status": "completed | needs_review | rejected",
  "final_label_id": 0,
  "diagnosis": "정상",
  "risk_level": "낮음",
  "confidence": 0.91,
  "reason": "...",
  "recommended_action": "...",
  "requires_human_review": false,
  "trace_id": "trace_..."
}
```

### GET `/diagnose/{diagnosis_id}`

저장된 진단 결과를 조회한다.

### GET `/diagnose/{diagnosis_id}/trace`

Agent workflow의 실행 trace와 tool 호출 결과를 조회한다.

### GET `/health`

서비스 상태, 모델 로딩 상태, GPU 사용 가능 여부를 반환한다.

## 6. Agents SDK 구성

OpenAI Agents SDK는 학습기가 아니라 진단 프로세스 관리자다. 공식 Agents SDK의 핵심 구성 요소인 Agent, tool, handoff, guardrail, tracing을 서비스 워크플로우에 적용한다.

### 6.1 Orchestrator Agent

역할:

- 전체 진단 workflow 시작
- 입력 검증 결과 확인
- 시계열 tool 호출
- VLM tool 호출
- Reviewer Agent 호출
- 최종 Report Agent 호출

지침:

```text
당신은 부분방전 진단 워크플로우 관리자입니다.
직접 라벨을 상상하지 말고, 반드시 tool 결과를 근거로 최종 판단을 구성하세요.
시계열 모델과 VLM 결과가 충돌하면 확정 진단을 내리지 말고 needs_review로 분기하세요.
```

### 6.2 Data Intake Agent

역할:

- 업로드 파일 메타 검증
- CSV shape 검증 결과 해석
- 이미지 형식 검증 결과 해석
- 입력 metadata 필수 필드 확인
- label leakage 가능성 차단

이 Agent는 모델 추론을 하지 않는다. 입력이 부적절하면 workflow를 중단한다.

### 6.3 Time-Series Inference Tool

Agent가 호출하는 deterministic tool이다.

입력:

```json
{
  "timeseries_csv_path": "uploads/diag_x/signal.csv"
}
```

출력:

```json
{
  "model_name": "patchtst",
  "label_id": 3,
  "label_name": "코로나 방전",
  "confidence": 0.87,
  "probabilities": {
    "0": 0.02,
    "1": 0.04,
    "2": 0.06,
    "3": 0.87,
    "4": 0.01
  },
  "features": {
    "rms": 30.37,
    "std": 4.96,
    "abs_p99": 39.0,
    "pulse_rate": 0.0069,
    "spectral_energy": 13982100.0
  }
}
```

주의:

- 원본 CSV 전체를 Agent/VLM prompt에 넣지 않는다.
- Agent에게는 추론 결과와 요약 feature만 제공한다.

### 6.4 VLM Inference Tool

Agent가 호출하는 deterministic tool이다.

입력:

```json
{
  "prpd_image_path": "uploads/diag_x/prpd.png",
  "safe_metadata": {
    "equipment_name": "ACSR-OC",
    "equipment_rated_voltage": "22900V",
    "sensor_type": "HFCT",
    "temperature": 19,
    "humidity": 66
  },
  "timeseries_summary": {
    "ts_pred_class": 3,
    "ts_confidence": 0.87,
    "rms": 30.37,
    "std": 4.96,
    "abs_p99": 39.0,
    "pulse_rate": 0.0069,
    "spectral_energy": 13982100.0
  }
}
```

출력:

```json
{
  "label_id": 3,
  "diagnosis": "코로나 방전",
  "risk_level": "주의",
  "reason": "PRPD 이미지와 시계열 요약 정보가 코로나 방전 패턴과 일치합니다.",
  "recommended_action": "고전압 접속부와 전계 집중 부위를 점검하세요."
}
```

후보 모델:

- Local first: `Qwen/Qwen3-VL-2B-Instruct`
- Local fallback: `Qwen/Qwen2.5-VL-3B-Instruct`
- Higher VRAM/cloud comparison: `LGAI-EXAONE/EXAONE-4.5-33B-AWQ`

### 6.5 Diagnosis Reviewer Agent

역할:

- 시계열 모델 결과와 VLM 결과 비교
- VLM JSON schema 검증
- label mismatch 확인
- confidence threshold 확인
- 과장된 권장 조치 차단
- human review 필요 여부 결정

분기 규칙:

```text
if input_validation_failed:
    status = rejected
elif ts_confidence < 0.60:
    status = needs_review
elif ts_label_id != vlm_label_id:
    status = needs_review
elif vlm_json_schema_invalid:
    status = needs_review
else:
    status = completed
```

### 6.6 Report Agent

역할:

- 최종 사용자 응답 생성
- diagnosis JSON 정리
- 현장 엔지니어용 간단 설명 생성
- human review 필요 사유 정리

Report Agent는 새로운 진단 라벨을 만들 수 없다. Reviewer Agent가 승인한 결과만 포맷팅한다.

## 7. Guardrail 설계

### Input Guardrail

workflow 시작 전 확인한다.

- 파일 확장자 검증
- 이미지 MIME 검증
- CSV shape 검증
- metadata 필수 필드 검증
- 사용자가 직접 label을 주입했는지 확인

### Tool Guardrail

각 tool 호출 전후에 확인한다.

Time-Series Tool:

- 입력 CSV 경로가 업로드 디렉터리 내부인지 확인
- 출력 label_id가 0~4 범위인지 확인
- probability 합이 1에 가까운지 확인

VLM Tool:

- prompt에 금지 필드가 들어가지 않았는지 확인
- 출력 JSON이 parse 가능한지 확인
- 필수 키가 모두 있는지 확인

### Output Guardrail

최종 응답 전 확인한다.

- `status` 값이 허용 enum인지 확인
- `final_label_id`와 `diagnosis`가 매핑되는지 확인
- `requires_human_review=true`일 때 확정적 권장 조치를 하지 않는지 확인

## 8. Trace / Audit Log

Agents SDK tracing을 사용해 다음을 남긴다.

- `diagnosis_id`
- `trace_id`
- 입력 파일 검증 결과
- Time-Series Tool 입력/출력 요약
- VLM Tool 입력/출력 요약
- Reviewer Agent 판단
- 최종 응답
- human review 분기 사유

민감 데이터 정책:

- 원본 CSV 전체를 trace에 저장하지 않는다.
- PRPD 이미지 바이너리를 trace에 저장하지 않는다.
- trace에는 경로, checksum, shape, 요약 feature만 저장한다.

## 9. 데이터 보안 및 누수 방지

프롬프트에 넣으면 안 되는 값:

- `label_id`
- `label_name`
- `PD_type`
- `sample_id`
- 파일명
- 파일 경로
- `defect_details`
- `defect_nums`
- `max_discharge_value`

서비스 입력 metadata에는 정답 라벨을 받지 않는다. 정답 라벨은 학습/평가 데이터에서만 존재하며, 서비스 추론 요청에는 포함되지 않는다.

## 10. React 화면 구성

### Diagnose Page

입력:

- PRPD 이미지 업로드
- 시계열 CSV 업로드
- 설비/환경 metadata form
- 진단 실행 버튼

출력:

- 최종 진단 라벨
- 위험도
- confidence
- 판단 근거
- 권장 조치
- human review 여부

### Trace Page

표시:

- 입력 검증 상태
- 시계열 모델 결과
- VLM 결과
- Reviewer 판단
- 최종 응답 생성 시간

## 11. FastAPI 서비스 구성

권장 모듈:

```text
service/
  PRD.md
  backend/
    app/
      main.py
      api/
        diagnose.py
        health.py
      schemas/
        request.py
        response.py
      agents/
        workflow.py
        prompts.py
        guardrails.py
      tools/
        timeseries.py
        vlm.py
      storage/
        uploads.py
        traces.py
  frontend/
    src/
      pages/
        DiagnosePage.tsx
        TracePage.tsx
```

## 12. 단계별 구현 계획

### Phase 1: API skeleton

- FastAPI `/health`
- FastAPI `/diagnose`
- request/response schema
- file upload 저장

### Phase 2: Deterministic inference tools

- Time-Series Inference Tool
- VLM Inference Tool
- tool 단위 mock/stub 지원

### Phase 3: Agents SDK workflow

- Orchestrator Agent
- Data Intake Agent
- Diagnosis Reviewer Agent
- Report Agent
- handoff 또는 manager-as-tools 구조 결정

권장 방식:

```text
초기 구현: manager-as-tools
확장 구현: handoff
```

이유:

- 초기 서비스는 진단 workflow가 고정되어 있으므로 manager가 tool을 순서대로 호출하는 구조가 더 예측 가능하다.
- handoff는 사용자 대화형 서비스나 복잡한 업무 분기가 많아졌을 때 도입한다.

### Phase 4: React UI

- upload form
- metadata form
- diagnosis result view
- trace view

### Phase 5: QA and deployment preparation

- valid sample 진단 e2e
- invalid CSV reject
- low confidence review 분기
- TS/VLM disagreement review 분기
- trace 저장 검증

## 13. Definition of Done

- React에서 PRPD 이미지, CSV, metadata를 업로드할 수 있다.
- FastAPI `/diagnose`가 요청을 받고 workflow를 실행한다.
- Time-Series Tool이 CSV를 추론하고 요약 feature를 반환한다.
- VLM Tool이 PRPD 이미지와 safe context를 입력받아 진단 JSON을 반환한다.
- Reviewer Agent가 충돌/저신뢰도/invalid JSON을 검출한다.
- 최종 응답이 `completed`, `needs_review`, `rejected` 중 하나로 반환된다.
- `/diagnose/{id}/trace`에서 workflow trace를 조회할 수 있다.

## 14. 참고 자료

- OpenAI Agents SDK GitHub: `https://github.com/openai/openai-agents-python`
- Agents SDK agents guide: `https://openai.github.io/openai-agents-python/agents/`
- Agents SDK handoffs guide: `https://openai.github.io/openai-agents-python/handoffs/`
- Agents SDK guardrails reference: `https://openai.github.io/openai-agents-python/ref/guardrail/`
- Agents SDK tracing guide: `https://openai.github.io/openai-agents-python/tracing/`
- Existing project PRD: `docs/PRD.md`
- VLM runbook: `docs/VLM_DEVELOPMENT_RUNBOOK.md`
