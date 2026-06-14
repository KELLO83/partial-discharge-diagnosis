# PD VLM Track

This folder owns the multimodal report generator track.

Current pipeline:

```text
data/manifest.csv
+ PRPD PNG
+ safe metadata
+ optional time-series evidence CSV
-> instruction_dataset.jsonl
-> QLoRA/SFT dry-run or training
```

Smoke check:

```powershell
python ml/vlm/train.py --model-profile smolvlm2_2b_qlora --sample-size 20 --dry-run
```

Training:

```powershell
python ml/vlm/train.py --model-profile qwen2_5_vl_3b_qlora --sample-size 500 --max-steps 100
```

Model profiles:

```text
qwen2_5_vl_3b_qlora   default quality profile for Korean PRPD report SFT
smolvlm2_2b_qlora     low-VRAM smoke profile
qwen3_vl_2b_qlora     compatibility/experimental profile
```

Default outputs:

```text
artifacts/models/vlm/instruction_dataset.jsonl
artifacts/models/vlm/training_config.json
artifacts/models/vlm/dry_run_summary.json
```

The VLM should explain and format evidence from time-series, vision, metadata,
and RAG. It should not be the only diagnostic model.
