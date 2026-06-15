# ML Training Workspace

This folder contains three model tracks:

```text
ml/timeseries  CSV time-series classifiers
ml/vision      lightweight PRPD image classifiers
ml/vlm         multimodal report generator SFT/QLoRA
```

Each track can still be trained directly:

```powershell
python ml/timeseries/train.py --model gru --sample-size 100
python ml/vision/train.py --sample-size 100 --epochs 1 --dry-run
python ml/vlm/train.py --model-profile smolvlm2_2b_qlora --sample-size 20 --dry-run
```

Config-driven orchestration:

```powershell
python ml/train_from_config.py --config ml/configs/training_smoke.yaml --plan-only
python ml/train_from_config.py --config ml/configs/training_smoke.yaml
python ml/train_from_config.py --config ml/configs/training_service_baseline.yaml --only vision
```

권장 사용:

```powershell
# 개발용 스모크(세 트랙 모두 dry-run)
python ml/train_from_config.py --config ml/configs/training_smoke.yaml

# 실제 학습(샘플 수를 줄여 빠른 검증 후 확장)
python ml/timeseries/train.py --model gru --manifest data/manifest.csv --artifact-dir artifacts/models/time_series --output results/experiments.csv --sample-size 20 --epochs 1
python ml/vision/train.py --manifest data/manifest.csv --output-dir artifacts/models/vision --sample-size 20 --epochs 1
python ml/vlm/train.py --model-profile qwen3_vl_2b_qlora --manifest data/manifest.csv --sample-size 20 --dry-run  # 최초엔 dry-run으로 점검
```

Config shape:

```yaml
version: 1
jobs:
  - name: vision_small_cnn_dry_run
    task: vision
    enabled: true
    args:
      manifest: data/manifest.csv
      output_dir: .omo/config-smoke/vision
      sample_size: 20
      dry_run: true
```

`task` must be one of `timeseries`, `vision`, or `vlm`. `args` are passed to the matching track's `train.py`; underscores become CLI dashes.
