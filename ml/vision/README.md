# PD Vision Model Track

This folder owns the lightweight PRPD image model track.

Current baseline:

```text
data/manifest.csv image_path
-> PRPD PNG resolved under data/
-> 224x224 RGB tensor
-> SmallPrpdCnn
-> 5-class prediction, probabilities, confidence, evidence_context.csv
```

Smoke check:

```powershell
python ml/vision/train.py --sample-size 20 --epochs 1 --dry-run
```

Training:

```powershell
python ml/vision/train.py --sample-size 500 --epochs 5
```

Default outputs:

```text
artifacts/models/vision/checkpoint.pt
artifacts/models/vision/model_manifest.json
artifacts/models/vision/evidence_context.csv
artifacts/models/vision/train_summary.json
```

The image model is a first baseline, not the final physics-aware representation.
Future work should add PRPD/PRPS tensors derived from raw signal data.
