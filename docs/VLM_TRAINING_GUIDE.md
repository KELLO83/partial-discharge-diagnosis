# VLM Training Guide

이 문서는 현재 프로젝트에서 VLM을 다시 학습하는 절차를 정리한다.

현재 권장 VLM 프로필은 `smolvlm2_2b_qlora`이다. RTX 4060 Laptop GPU 기준으로 full fine-tuning이 아니라 4-bit QLoRA SFT를 사용한다.

## 1. 입력 구조

VLM 학습 입력은 다음 정보를 합쳐 만든다.

```text
PRPD 이미지
+ 안전 메타데이터
+ 시계열 모델 예측 context
+ 비전 모델 예측 context
-> VLM instruction dataset
-> QLoRA 학습
```

원본 데이터 목록은 `data/manifest.csv`를 사용한다.

시계열/비전 context CSV는 VLM 학습 중 매번 시계열 모델과 비전 모델을 다시 돌리지 않기 위해 미리 저장해둔 예측 결과 파일이다.

## 2. 현재 Active 모델 확인

먼저 백엔드가 어떤 checkpoint를 쓰는지 확인한다.

```powershell
python -c "from service.backend.app.models.model_artifacts import ModelAdapterSettings, ModelArtifactRegistry; s=ModelAdapterSettings.from_env(); r=ModelArtifactRegistry(s.artifact_root, s.artifact_overrides); [print(t, r.get(t).ready, r.get(t).checkpoint_path, r.get(t).error) for t in ('time_series','vision','vlm')]"
```

`time_series`, `vision`이 `True`여야 context CSV를 실제 모델 예측으로 만들 수 있다.

## 3. Context CSV 생성

시계열 모델이나 비전 모델을 새로 학습했거나 checkpoint를 바꿨다면 context CSV를 다시 만든다.

```powershell
python ml/vlm/scripts/export_ts_context.py `
  --manifest data/manifest.csv `
  --output artifacts/models/vlm/vlm_ts_context_inception_effb0_2000.csv `
  --sample-size 2000 `
  --model-artifact-root artifacts/models
```

```powershell
python ml/vlm/scripts/export_vision_context.py `
  --manifest data/manifest.csv `
  --output artifacts/models/vlm/vlm_vision_context_effb0_2000.csv `
  --sample-size 2000 `
  --model-artifact-root artifacts/models
```

이미 같은 시계열/비전 checkpoint로 만든 context CSV가 있으면 이 단계는 생략해도 된다.

## 4. VLM 학습 실행

기본 재학습 명령어:

```powershell
python ml/vlm/train.py `
  --model-profile smolvlm2_2b_qlora `
  --manifest data/manifest.csv `
  --ts-context artifacts/models/vlm/vlm_ts_context_inception_effb0_2000.csv `
  --vision-context artifacts/models/vlm/vlm_vision_context_effb0_2000.csv `
  --output-dir artifacts/models/vlm `
  --sample-size 2000 `
  --max-steps 40 `
  --save-steps 5 `
  --gpu-memory-fraction 0.9 `
  --eval-ratio 0.1 `
  --early-stop-patience 4 `
  --attn-implementation sdpa
```

빠른 smoke test만 할 때:

```powershell
python ml/vlm/train.py `
  --model-profile smolvlm2_2b_qlora `
  --manifest data/manifest.csv `
  --ts-context artifacts/models/vlm/vlm_ts_context_inception_effb0_2000.csv `
  --vision-context artifacts/models/vlm/vlm_vision_context_effb0_2000.csv `
  --output-dir artifacts/models/vlm `
  --sample-size 200 `
  --max-steps 5 `
  --save-steps 5 `
  --gpu-memory-fraction 0.9 `
  --eval-ratio 0.1 `
  --early-stop-patience 2 `
  --attn-implementation sdpa
```

## 5. 학습 결과 위치

학습이 끝나면 다음 형식의 폴더가 생성된다.

```text
artifacts/models/vlm/smolvlm2_2b_qlora/YYYYMMDD_HHMMSS/
```

주요 파일:

```text
best.pt/                  LoRA adapter checkpoint
resumet.pt/               재개용 trainer checkpoint
processor/                processor/tokenizer
instruction_dataset.jsonl VLM 학습 데이터
model_manifest.json       해당 run manifest
train_summary.json        학습 설정과 결과 요약
tensorboard/              TensorBoard 로그
```

서비스가 기본으로 읽는 최신 manifest는 아래에 발행된다.

```text
artifacts/models/vlm/model_manifest.json
```

## 6. TensorBoard 확인

학습 loss와 eval loss는 TensorBoard에서 확인한다.

```powershell
tensorboard --logdir artifacts/models/vlm/smolvlm2_2b_qlora
```

브라우저에서 표시되는 주소를 열면 된다. 보통 `http://localhost:6006`이다.

특정 run만 보고 싶으면:

```powershell
tensorboard --logdir artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/tensorboard
```

## 7. Loss 확인

학습 summary:

```powershell
Get-Content artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/train_summary.json
```

step별 train/eval loss:

```powershell
python -c "import json; p='artifacts/models/vlm/smolvlm2_2b_qlora/20260615_202950/checkpoint-20/trainer_state.json'; s=json.load(open(p,encoding='utf-8')); print([x for x in s['log_history'] if 'loss' in x or 'eval_loss' in x])"
```

run 폴더명과 checkpoint 번호는 실제 생성된 값으로 바꾼다.

## 8. 서비스에 새 VLM 지정

학습이 정상 완료되면 `.env`에서 VLM manifest와 checkpoint를 새 run으로 지정한다.

```env
MODEL_ADAPTER_MODE=checkpoint
MODEL_ARTIFACT_ROOT=artifacts/models
MODEL_VLM_MANIFEST=artifacts/models/vlm/smolvlm2_2b_qlora/YYYYMMDD_HHMMSS/model_manifest.json
MODEL_VLM_CHECKPOINT=artifacts/models/vlm/smolvlm2_2b_qlora/YYYYMMDD_HHMMSS/best.pt
```

변경 후 백엔드 서버를 재시작해야 적용된다.

## 9. Smoke Test

서비스 registry가 새 checkpoint를 읽는지 확인한다.

```powershell
python -c "from service.backend.app.models.model_artifacts import ModelAdapterSettings, ModelArtifactRegistry; s=ModelAdapterSettings.from_env(); r=ModelArtifactRegistry(s.artifact_root, s.artifact_overrides); a=r.get('vlm'); print(a.ready, a.model_name, a.checkpoint_path, a.error)"
```

간단한 adapter 생성 테스트는 학습 직후 실행한다.

```powershell
python -m pytest service/backend/tests/test_model_runtime.py ml/vlm/tests/test_instruction_dataset.py ml/vlm/tests/test_prompts.py
```

## 10. 언제 Step을 늘릴지

짧은 run에서 eval loss가 계속 내려가면 `--max-steps`를 늘린다.

권장 순서:

```text
5 step smoke
-> 20 step baseline
-> 40 step
-> 80 step
```

eval loss가 2회 이상 거의 줄지 않거나 증가하면 일단 중지하고 dataset/prompt/context 품질을 확인한다.

## 11. 주의사항

- `context CSV`는 시계열/비전 checkpoint를 바꿀 때 다시 만든다.
- VLM checkpoint만 바꿀 때는 context CSV를 다시 만들 필요가 없다.
- `MODEL_VLM_CHECKPOINT`만 바꾸지 말고 가능하면 `MODEL_VLM_MANIFEST`도 같은 run으로 맞춘다.
- AWQ/GGUF 모델은 보통 추론용이다. QLoRA 학습에는 현재 `smolvlm2_2b_qlora` 프로필을 우선 사용한다.
- EXAONE 4.0 1.2B는 VLM이 아니라 text LLM이다. 이미지 입력 VLM 학습 용도로 쓰지 않는다.
