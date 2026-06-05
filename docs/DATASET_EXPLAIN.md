# Dataset Explanation

## 1. Dataset Overview

This project uses the AI-Hub industrial electrical fire prevention partial-discharge dataset.

- Dataset name: Industrial Electrical Fire Prevention Partial Discharge Dataset
- Source: AI-Hub
- Data page: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71682&topMenu=100
- Domain: industrial safety, power-equipment diagnosis, partial-discharge diagnosis
- Data type: sensor time-series CSV + PRPD PNG image + label/metadata JSON

Each sample is a 1:1 match between three files:

```text
one partial-discharge sample
├── PRPD image (*.png)
├── partial-discharge time-series data (*.csv)
└── label and metadata (*.json)
```

The primary modeling target is 5-class CSV time-series classification, not PRPD image-only vision classification. The secondary target is a small VLM diagnosis model using PRPD image, JSON metadata, and time-series summary information.

Forecasting is out of scope. Treat this as supervised time-series classification: one measurement CSV is used to classify the current partial-discharge state.

## 2. Dataset Scale

The AI-Hub page describes 300,000 final dataset samples:

| Split | Count | Ratio |
| --- | ---: | ---: |
| Training data | 239,980 | 80% |
| Validation data | 30,010 | 10% |
| Test data | 30,010 | 10% |
| Total | 300,000 | 100% |

It also describes raw collected data:

| Raw data | File format | Count |
| --- | --- | ---: |
| PRPD image data | `.BMP` | 861,150 |
| Partial-discharge time-series data | `.CSV` | 861,150 |
| Composite sensor data | `.XLSX` | 256 |
| Collection-environment data | `.XLSX` | 256 |

This project starts from the extracted local `Train/` folder with `.png`, `.csv`, and `.json` source/label files.

## 3. Current Local Train Data

Current local `Train/` counts:

| File type | Count | Location |
| --- | ---: | --- |
| `.png` | 30,010 | `Train/01_source_data` equivalent source folder |
| `.csv` | 30,010 | `Train/01_source_data` equivalent source folder |
| `.json` | 30,010 | `Train/02_label_data` equivalent label folder |

Current `Train/manifest.csv` summary:

- Total samples: `30,010`
- Labels `0` through `4` are balanced with `6,002` samples each
- CSV shape: `(20, 7680)`
- Each row connects `json_path`, `image_path`, `timeseries_path`, `label_id`, and equipment/environment metadata

The Train folder separates source data and label data. In the original AI-Hub extraction, folder names may contain non-English class and equipment names. Do not use folder-name strings as model features.

`VS_` is treated as a source-data prefix and `VL_` as a label-data prefix. Subfolders generally encode:

```text
VS_{discharge_type}_{insulator_type}_{equipment_name}
VL_{discharge_type}_{insulator_type}_{equipment_name}
```

## 4. Class Definitions

Use JSON `label.PD_type` as the target label.

| PD_type | English label | Description |
| ---: | --- | --- |
| 0 | normal | Normal state without partial discharge |
| 1 | noise | Noise-like signal rather than a discharge signal |
| 2 | surface_discharge | Discharge along an insulator surface |
| 3 | corona_discharge | Corona-like discharge near electric-field concentration |
| 4 | void_discharge | Discharge inside an internal void or cavity of insulation |

Use this fixed mapping in code:

```python
PD_LABELS_EN = {
    0: "normal",
    1: "noise",
    2: "surface_discharge",
    3: "corona_discharge",
    4: "void_discharge",
}
```

Current local Train label distribution:

| Label ID | Label name | Count |
| --- | --- | ---: |
| 0 | normal | 6,002 |
| 1 | noise | 6,002 |
| 2 | surface_discharge | 6,002 |
| 3 | corona_discharge | 6,002 |
| 4 | void_discharge | 6,002 |
| Total | - | 30,010 |

Warning: file and folder names can contain class names. Training labels must come only from JSON `label.PD_type` or manifest `label_id`. Do not use `image_path`, `timeseries_path`, folder names, or file-name strings as features.

## 5. Local File Matching Rules

In the local Train data, `.csv`, `.png`, and `.json` files share the same base filename.

Example pattern:

```text
source CSV:
Train/source/VS_noise_solid_ACSR-OC/
└── noise_solid_ACSR-OC_230910_195222_HFCT_1000.csv

source PNG:
Train/source/VS_noise_solid_ACSR-OC/
└── noise_solid_ACSR-OC_230910_195222_HFCT_1000.png

label JSON:
Train/labels/VL_noise_solid_ACSR-OC/
└── noise_solid_ACSR-OC_230910_195222_HFCT_1000.json
```

Generate the manifest from JSON files:

```text
1. Walk JSON files.
2. Use the JSON file stem as sample_id.
3. Find the PNG and CSV with the same stem in the source-data folders.
4. Extract JSON label.PD_type and metadata.
```

JSON-internal `label.image_path` and `label.timeseries_path` may be AI-Hub logical paths. Local paths are safer when matched by filename stem.

## 6. Filename Rules

Observed source filenames follow this conceptual pattern:

```text
{discharge_type}_{insulator_type}_{equipment_name}_{date}_{time}_{sensor_type}_{clearance_distance}.csv
{discharge_type}_{insulator_type}_{equipment_name}_{date}_{time}_{sensor_type}_{clearance_distance}.png
{discharge_type}_{insulator_type}_{equipment_name}_{date}_{time}_{sensor_type}_{clearance_distance}.json
```

Example:

```text
noise_solid_ACSR-OC_230910_195222_HFCT_1000.csv
```

| Part | Example | Meaning |
| --- | --- | --- |
| discharge_type | noise | label name |
| insulator_type | solid | insulator category |
| equipment_name | ACSR-OC | power-equipment name |
| date | 230910 | interpretable as 2023-09-10 |
| time | 195222 | interpretable as 19:52:22 |
| sensor_type | HFCT | sensor type |
| clearance_distance | 1000 | presumed sensor clearance distance |

Final training metadata should prefer JSON values over filename parsing.

## 7. CSV Time-Series Format

Current Train CSV samples are headerless numeric matrices.

| Item | Value |
| --- | ---: |
| Rows | 20 |
| Columns | 7,680 |
| Header | none |
| Delimiter | comma `,` |
| Values | integer sensor values |
| File size | about 460KB |

One CSV loads as:

```text
shape = (20, 7680)
```

Interpretation:

```text
20 rows = 20 measurement segments or pseudo-channels
7680 columns = time points per segment
```

Do not assume the `20` axis is physical sensor channels. JSON `recording_time_length` is 20 and sensor type is usually `HFCT` or `UHF`, so the modeling code treats this axis as a pseudo-channel or segment dimension.

One CSV file is one training sample:

```text
1 CSV file = 1 sample = 1 label
```

Splitting the 20 rows from one CSV across train/test would cause data leakage.

## 8. Model Input Shapes

Raw time-series model input:

```text
x.shape = [C, T]
C = 20
T = 7680
```

PyTorch batch input:

```text
x.shape = [B, C, T]
```

Model conventions:

| Model family | Input convention |
| --- | --- |
| 1D-CNN / ResNet1D / InceptionTime | `[B, C, T]` |
| TCN / some CNN models | `[B, C, T]` |
| GRU / some Transformer models | `[B, T, C]` |
| PatchTST / iTransformer / some official implementations | wrapper-specific `[B, T, C]` or `[B, C, T]` |
| MiniROCKET / MultiROCKET | `[N, C, T]` |
| Feature ML model | `[N, F]` |

Flattening is discouraged:

```text
20 x 7680 = 153,600-dimensional vector
```

It is too high-dimensional and can destroy phase/cycle structure.

## 9. 60Hz Phase/Cycle Interpretation

JSON metadata records `60Hz` power frequency. If one CSV row is interpreted as a 1-second segment:

```text
sampling_per_second = 7680
power_frequency = 60Hz
samples_per_cycle = 7680 / 60 = 128
```

Cycle phase:

```python
samples_per_cycle = 128
phase_degree = (sample_index % samples_per_cycle) / samples_per_cycle * 360.0
```

This phase information can be used as numeric features without creating PRPD images.

```text
32 phase bins x 16 amplitude bins = 512-dimensional numeric PRPD histogram
```

The `row = 1 second` interpretation is a project assumption based on the current samples and metadata. Keep shape validation for the full dataset.

## 10. PNG Image Format

Current PRPD PNG samples:

| Item | Value |
| --- | --- |
| Resolution | 256 x 256 |
| Color format | RGB |
| Pixel format | 24-bit RGB |

Most VLM processors resize and normalize images internally. Start from the assumption that source PRPD images are 256x256 RGB.

## 11. JSON Annotation Structure

JSON files contain `label` and `metadata` sections.

```text
json
├── label
│   ├── PD_type
│   ├── image_path
│   └── timeseries_path
└── metadata
    ├── equipment_information
    ├── environment
    ├── discharge_information
    └── discharge_evaluation_factors
```

JSON may include a UTF-8 BOM, so read with `utf-8-sig`.

```python
import json
from pathlib import Path

data = json.loads(Path("sample.json").read_text(encoding="utf-8-sig"))
```

Important fields:

| Field | Usage | Description |
| --- | --- | --- |
| `label.PD_type` | required | target label |
| `label.image_path` | reference | matching PRPD image path |
| `label.timeseries_path` | reference | matching CSV path |
| `metadata.equipment_information.insulator_type` | optional | insulator type |
| `metadata.equipment_information.insulator_name` | optional | insulator name |
| `metadata.equipment_information.equipment_name` | optional | power-equipment name |
| `metadata.equipment_information.equipment_manufacturer` | optional | manufacturer |
| `metadata.equipment_information.equipment_id` | discouraged | equipment identifier; possible split leakage |
| `metadata.equipment_information.equipment_rated_voltage` | optional | rated voltage |
| `metadata.equipment_information.equipment_rated_current` | optional | rated current |
| `metadata.environment.recording_time` | split caution | useful for group split, risky as a feature |
| `metadata.environment.recording_time_length` | reference | current value is 20 |
| `metadata.environment.power_supply_frequency` | reference | current value is 60Hz |
| `metadata.environment.sensor_type` | optional | mostly HFCT, some UHF |
| `metadata.environment.temperature` | optional | temperature |
| `metadata.environment.humidity` | optional | humidity |
| `metadata.environment.clearance_distance` | optional | sensor clearance distance |
| `metadata.discharge_information.defect_nums` | discouraged | possible label leakage |
| `metadata.discharge_information.defect_details` | discouraged | possible label leakage |
| `metadata.discharge_evaluation_factors.max_discharge_value` | caution | use only if available at inference time |

Example with English placeholder values:

```json
{
  "label": {
    "PD_type": 1,
    "image_path": "./source_data/noise/solid/ACSR-OC/noise_solid_ACSR-OC_230910_195222_HFCT_1000.png",
    "timeseries_path": "./source_data/noise/solid/ACSR-OC/noise_solid_ACSR-OC_230910_195222_HFCT_1000.csv"
  },
  "metadata": {
    "equipment_information": {
      "insulator_type": "solid",
      "insulator_name": "XLPE",
      "equipment_name": "ACSR-OC",
      "equipment_manufacturer": "manufacturer",
      "equipment_id": "-",
      "equipment_rated_voltage": "22900V",
      "equipment_rated_current": "268A"
    },
    "environment": {
      "recording_time": "230910_195222",
      "recording_time_length": 20,
      "power_supply_frequency": "60Hz",
      "sensor_type": "HFCT",
      "temperature": "19",
      "humidity": "66",
      "clearance_distance": "['1000mm']"
    },
    "discharge_information": {
      "defect_nums": "0",
      "defect_details": [["0"], ["0"]]
    },
    "discharge_evaluation_factors": {
      "max_discharge_value": 82
    }
  }
}
```

## 12. Equipment and Metadata Distribution

The dataset includes equipment across solid, liquid, and gas insulation categories.

| Insulator type | Equipment |
| --- | --- |
| Solid insulation | TFR-CV, CNCV-W, ACSR-OC |
| Liquid insulation | single-phase oil-immersed transformer, power oil-immersed transformer, instrument transformer |
| Gas insulation | 7.2kV switchgear, 22.9kV switchgear, 25.8kV GIS |

Current local Train distribution:

| Item | Distribution |
| --- | --- |
| Insulator type | solid 10,005 / liquid 10,005 / gas 10,000 |
| Sensor type | HFCT 29,010 / UHF 1,000 |

Equipment distribution:

| Equipment | Count |
| --- | ---: |
| ACSR-OC | 3,335 |
| CNCV-W | 3,335 |
| TFR-CV | 3,335 |
| instrument transformer | 3,335 |
| single-phase oil-immersed transformer | 3,335 |
| power oil-immersed transformer | 3,335 |
| 7.2kV switchgear | 2,500 |
| 22.9kV switchgear | 2,500 |
| 25.8kV GIS | 5,000 |

## 13. Recommended Manifest Format

Store all matched file information in one manifest before training.

Current main columns:

```text
sample_id
split
json_path
image_path
timeseries_path
label_id
label_name
insulator_type
insulator_name
equipment_name
equipment_manufacturer
equipment_id
equipment_rated_voltage
equipment_rated_current
recording_time
recording_time_length
sensor_type
temperature
humidity
clearance_distance
max_discharge_value
json_image_path
json_timeseries_path
```

Minimum required columns:

```text
sample_id,json_path,image_path,timeseries_path,label_id,label_name
```

## 14. Preprocessing Rules

### 14.1 Basic Loading

```python
import json
import numpy as np
import pandas as pd


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_timeseries(csv_path):
    x = pd.read_csv(csv_path, header=None).values
    return x.astype(np.float32)
```

### 14.2 Shape Validation

Validate every CSV before training:

```python
assert x.ndim == 2
assert x.shape[0] == 20
assert x.shape[1] == 7680
```

If a file has a different shape, choose one explicit policy: exclude it, pad/truncate it, resample it, or manage it in a separate split.

### 14.3 Normalization

Candidates:

```text
1. sample-wise z-score
2. channel-wise z-score
3. train-set global mean/std normalization
4. median/IQR robust scaling
```

Amplitude magnitude may itself be informative for partial discharge, so run normalization ablations:

```text
A. preserve raw amplitude
B. sample-wise z-score
C. channel-wise z-score
D. log / robust scaling
```

## 15. Feature Design for Feature-Based Classification

`feature_logistic`, `feature_svm`, `feature_random_forest`, and `feature_tabpfn` consume extracted feature vectors, not raw CSV tensors.

```text
CSV time-series
-> feature extraction
-> feature vector
-> Logistic / Linear SVM / RandomForest / TabPFN
-> 5-class prediction
```

Feature sets:

| Feature set | CSV feature columns | With metadata | Description |
| --- | ---: | ---: | --- |
| `small` | 64 | 74 | Fast baseline with global/stat, FFT, amplitude histogram, pulse, cycle, and half-cycle features |
| `medium` | 128 | 138 | Balanced baseline with phase-bin count/max |
| `full` | 182 | 192 | Includes a compact 96-bin numeric PRPD histogram |

The default is `small`.

## 16. Features That Must Not Be Used

These fields can cause label leakage and must not be used as training features:

```text
label.PD_type
label.image_path
label.timeseries_path
sample_id
file-name string
folder-name string
path token containing discharge type
label_name
metadata.discharge_information.defect_nums
metadata.discharge_information.defect_details
metadata.discharge_evaluation_factors.max_discharge_value
```

`max_discharge_value` requires caution. It may be usable if it is available at inference time as a sensor-derived value, but it may also be a post-labeling value. Exclude it from default experiments and use it only in a separate ablation.

## 17. Train / Validation / Test Split

Prefer the official AI-Hub train/validation/test split when available.

If creating a custom split, a naive random split can place the same equipment, date, and measurement conditions in both train and test, inflating generalization performance.

Recommended split criteria:

```text
1. file-level split
2. equipment-level group split
3. recording date/time group split
4. voltage / sensor-distance group split
5. insulator-type group split
```

Minimum rules:

```text
Do not split rows from the same CSV into different splits.
Do not place the same JSON/CSV/PNG sample in both train and test.
Remove filename/path-based leakage.
Use and reuse a fixed manifest split for model comparisons.
```

Current runners use manifest `split=train` and `split=valid` when both are present. If no valid split exists, they fall back to a label-stratified random split.

Fixed split command:

```powershell
python ml/scripts/make_splits.py --manifest Train/manifest.csv --output Train/manifest_random_split_seed42.csv --valid-ratio 0.2 --seed 42
```

## 18. Evaluation Metrics

Accuracy alone can hide class imbalance and dangerous failures. Record:

```text
accuracy
macro_f1
weighted_f1
balanced_accuracy
class-wise precision
class-wise recall
confusion_matrix
pd_to_normal_error_count
```

For field diagnosis, explicitly inspect recall for actual discharge classes and the number of true discharge samples predicted as normal.

## 19. Recommended Baselines

### 19.1 Feature-Only Baseline

```text
feature_logistic
feature_svm
feature_random_forest
feature_tabpfn
```

Input:

```text
stat features + pulse features + phase-bin features + FFT features + numeric PRPD histogram
```

### 19.2 Raw Time-Series Baseline

```text
GRU
InceptionTime
PatchTST
TimesNet
MOMENT
ModernTCN
MiniROCKET / MultiROCKET
```

Input:

```text
[B, 20, 7680]
or [B, 7680, 20], depending on the model wrapper
```

### 19.3 Hybrid Model

```text
raw time-series encoder embedding + hand-crafted feature embedding
-> concat
-> classifier
```

### 19.4 Late Fusion / Stacking

```text
feature model probability
raw time-series model probability
-> concat probabilities
-> meta classifier
-> final prediction
```
