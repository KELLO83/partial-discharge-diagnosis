# 데이터셋 설명

## 1. 데이터셋 개요

본 프로젝트는 AI-Hub의 `산업 설비 전기 화재 사고 예방 부분방전 데이터`를 사용한다.

- 데이터셋명: 산업 설비 전기 화재 사고 예방 부분방전 데이터
- 영문명: Industrial Electrical Fire Prevention Partial Discharge Dataset
- 출처: AI-Hub
- 데이터 페이지: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71682&topMenu=100
- 도메인: 산업 안전, 전력 설비 진단, 부분방전 진단
- 데이터 유형: 센서 시계열 CSV + PRPD 이미지 PNG + 라벨/메타데이터 JSON

하나의 샘플은 다음 세 파일이 1:1로 매칭되는 구조다.

```text
하나의 부분방전 샘플
├── PRPD 이미지 (*.png)
├── 부분방전 시계열 데이터 (*.csv)
└── 라벨 및 메타데이터 (*.json)
```

이 프로젝트의 1차 모델링 목표는 PRPD 이미지 단독 비전 분류가 아니라, CSV 시계열 기반 5-class classification이다. 2차 목표는 PRPD 이미지, JSON 메타데이터, 시계열 요약 정보를 결합한 소형 VLM 진단 모델이다.

Forecasting은 현재 범위에서 제외한다. 하나의 측정 CSV 파일을 입력받아 현재 부분방전 상태를 분류하는 supervised time-series classification 문제로 다룬다.

## 2. 데이터 규모

AI-Hub 데이터 페이지 기준 전체 원천 데이터는 300,000건이다.

| 구분 | 개수 | 비율 |
| --- | ---: | ---: |
| 학습 데이터 | 239,980 | 80% |
| 검증 데이터 | 30,010 | 10% |
| 테스트 데이터 | 30,010 | 10% |
| 전체 | 300,000 | 100% |

AI-Hub 페이지에는 원시 수집 데이터도 함께 설명되어 있다.

| 원시 데이터 | 파일 형식 | 개수 |
| --- | --- | ---: |
| PRPD 이미지 데이터 | `.BMP` | 861,150 |
| 부분방전 시계열 데이터 | `.CSV` | 861,150 |
| 복합센서 데이터 | `.XLSX` | 256 |
| 수집환경 데이터 | `.XLSX` | 256 |

현재 프로젝트에서는 우선 해제된 `Train/` 폴더의 `.png`, `.csv`, `.json` 원천/라벨 파일을 기준으로 개발한다.

## 3. 현재 로컬 Train 데이터

현재 로컬 `Train/` 데이터 기준 파일 수는 다음과 같다.

| 파일 형식 | 개수 | 위치 |
| --- | ---: | --- |
| `.png` | 30,010 | `Train/01.원천데이터` |
| `.csv` | 30,010 | `Train/01.원천데이터` |
| `.json` | 30,010 | `Train/02.라벨링데이터` |

현재 `Train/manifest.csv` 기준:

- 전체 샘플 수: `30,010`
- label `0~4` 각각 `6,002`개로 균형
- CSV shape: `(20, 7680)`
- 하나의 row는 `json_path`, `image_path`, `timeseries_path`, `label_id`, 설비/환경 메타데이터를 연결한다.

Train 폴더 구조는 원천데이터와 라벨링데이터가 분리되어 있다.

```text
Train/
├── 01.원천데이터/
│   ├── VS_노이즈_고체_ACSR-OC/
│   ├── VS_노이즈_고체_CNCV-W/
│   ├── ...
│   └── VS_표면방전_액체_전력용유입변압기/
└── 02.라벨링데이터/
    ├── VL_노이즈_고체_ACSR-OC/
    ├── VL_노이즈_고체_CNCV-W/
    ├── ...
    └── VL_표면방전_액체_전력용유입변압기/
```

`VS_`는 원천데이터, `VL_`은 라벨링데이터로 해석한다. 하위 폴더명은 대체로 다음 조합이다.

```text
VS_{방전유형}_{절연체종류}_{설비명}
VL_{방전유형}_{절연체종류}_{설비명}
```

## 4. 클래스 정의

JSON 라벨의 `label.PD_type` 값을 정답 라벨로 사용한다.

| PD_type | 한글 라벨 | 영문 라벨 | 설명 |
| ---: | --- | --- | --- |
| 0 | 정상 | normal | 부분방전이 없는 정상 상태 |
| 1 | 노이즈 | noise | 방전 신호가 아닌 잡음성 신호 |
| 2 | 표면방전 | surface_discharge | 절연체 표면을 따라 발생하는 방전 |
| 3 | 코로나방전 | corona_discharge | 전계 집중 부근에서 발생하는 코로나성 방전 |
| 4 | 보이드방전 | void_discharge | 절연체 내부 void/cavity에서 발생하는 방전 |

코드에서는 다음 매핑을 고정한다.

```python
PD_LABELS_KO = {
    0: "정상",
    1: "노이즈",
    2: "표면방전",
    3: "코로나방전",
    4: "보이드방전",
}

PD_LABELS_EN = {
    0: "normal",
    1: "noise",
    2: "surface_discharge",
    3: "corona_discharge",
    4: "void_discharge",
}
```

현재 로컬 Train 라벨 분포는 다음과 같다.

| 라벨 ID | 라벨명 | 개수 |
| --- | --- | ---: |
| 0 | 정상 | 6,002 |
| 1 | 노이즈 | 6,002 |
| 2 | 표면 방전 | 6,002 |
| 3 | 코로나 방전 | 6,002 |
| 4 | 보이드 방전 | 6,002 |
| 전체 | - | 30,010 |

주의: 파일명과 폴더명에 클래스명이 포함될 수 있다. 학습 라벨은 반드시 JSON의 `label.PD_type` 또는 이를 기반으로 생성한 manifest의 `label_id`를 사용한다. `image_path`, `timeseries_path`, 폴더명, 파일명 문자열을 feature로 사용하면 안 된다.

## 5. 로컬 파일 매칭 규칙

현재 로컬 Train 데이터에서는 `.csv`, `.png`, `.json` 파일의 base filename이 일치한다.

예시:

```text
원천 CSV:
Train/01.원천데이터/VS_노이즈_고체_ACSR-OC/
└── 노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.csv

원천 PNG:
Train/01.원천데이터/VS_노이즈_고체_ACSR-OC/
└── 노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.png

라벨 JSON:
Train/02.라벨링데이터/VL_노이즈_고체_ACSR-OC/
└── 노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.json
```

manifest 생성은 JSON 파일 기준으로 수행한다.

```text
1. JSON 파일을 순회한다.
2. JSON 파일 stem을 sample_id로 사용한다.
3. 같은 stem을 가진 PNG와 CSV를 원천데이터 폴더에서 찾는다.
4. JSON의 label.PD_type과 metadata를 함께 추출한다.
```

JSON 내부의 `label.image_path`, `label.timeseries_path`는 AI-Hub 논리 경로일 수 있다. 현재 로컬 경로는 `Train/01.원천데이터/VS_...` 형태이므로 JSON 내부 경로를 그대로 쓰기보다 파일명 기준으로 실제 경로를 매칭하는 편이 안전하다.

## 6. 파일명 규칙

현재 확인된 원천 파일명은 다음 형태를 따른다.

```text
{방전유형}_{절연체종류}_{설비명}_{측정일자}_{측정시간}_{센서종류}_{이격거리}.csv
{방전유형}_{절연체종류}_{설비명}_{측정일자}_{측정시간}_{센서종류}_{이격거리}.png
{방전유형}_{절연체종류}_{설비명}_{측정일자}_{측정시간}_{센서종류}_{이격거리}.json
```

예시:

```text
노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.csv
```

| 부분 | 값 | 의미 |
| --- | --- | --- |
| 방전유형 | 노이즈 | 라벨명 |
| 절연체종류 | 고체 | 절연체 분류 |
| 설비명 | ACSR-OC | 전력 설비명 |
| 측정일자 | 230910 | 2023-09-10 형식으로 해석 가능 |
| 측정시간 | 195222 | 19:52:22 형식으로 해석 가능 |
| 센서종류 | HFCT | 센서 |
| 이격거리 | 1000 | 센서 이격 거리로 추정 |

최종 학습용 메타데이터는 파일명 파싱보다 JSON 값을 우선한다.

## 7. CSV 시계열 데이터 형태

현재 Train 샘플 기준 CSV 파일은 헤더가 없는 숫자 행렬이다.

| 항목 | 값 |
| --- | ---: |
| 행 수 | 20 |
| 열 수 | 7,680 |
| 헤더 | 없음 |
| 구분자 | comma `,` |
| 값 형태 | 정수형 센서 값 |
| 파일 크기 | 약 460KB |

즉, 하나의 CSV는 다음 2차원 배열로 로드된다.

```text
shape = (20, 7680)
```

해석:

```text
20 rows = 20개 측정 구간 또는 segment
7680 columns = 각 구간의 time points
```

`20`축을 실제 물리 센서 채널 수로 단정하지 않는다. JSON의 `recording_time_length`가 20이고 센서 타입은 보통 `HFCT` 또는 `UHF`로 기록되어 있으므로, 모델 입력에서는 `20`축을 pseudo-channel 또는 segment dimension으로 취급한다.

CSV 파일 1개를 하나의 학습 샘플로 취급한다.

```text
1 CSV file = 1 sample = 1 label
```

동일 CSV 내부의 20개 row를 서로 다른 샘플로 쪼개 train/test에 나누면 data leakage가 발생할 수 있다.

## 8. 모델 입력 형태

Raw time-series model 기본 입력:

```text
x.shape = [C, T]
C = 20
T = 7680
```

PyTorch batch 입력:

```text
x.shape = [B, C, T]
```

모델별 입력 관례:

| Model family | Input convention |
| --- | --- |
| 1D-CNN / ResNet1D / InceptionTime | `[B, C, T]` |
| TCN / 일부 CNN 계열 | `[B, C, T]` |
| GRU / 일부 Transformer 계열 | `[B, T, C]` |
| PatchTST / iTransformer / 일부 공식 구현 | 구현체 wrapper에 맞춰 `[B, T, C]` 또는 `[B, C, T]` |
| MiniROCKET / MultiROCKET | `[N, C, T]` |
| Feature ML model | `[N, F]` |

단순 flatten 입력은 비추천한다.

```text
20 × 7680 = 153,600 dimensional vector
```

차원이 너무 크고 phase/cycle 구조를 잃기 쉽다.

## 9. 60Hz phase/cycle 기반 해석

JSON metadata에는 전원 주파수 `60Hz`가 기록되어 있다. CSV의 한 row를 1초 구간으로 가정하면 다음 계산이 가능하다.

```text
sampling_per_second = 7680
power_frequency = 60Hz
samples_per_cycle = 7680 / 60 = 128
```

따라서 한 cycle은 약 128 sample이며, sample index를 phase로 변환할 수 있다.

```python
samples_per_cycle = 128
phase_degree = (sample_index % samples_per_cycle) / samples_per_cycle * 360.0
```

이 phase 정보는 PRPD 이미지를 만들지 않고도 numeric feature로 사용할 수 있다.

```text
phase bin 32개 × amplitude bin 16개 = 512차원 PRPD numeric histogram
```

단, `row = 1초` 해석은 현재 샘플과 metadata를 기준으로 한 프로젝트 가정이다. 전체 데이터에 대해 동일한 구조인지 shape validation을 유지해야 한다.

## 10. PNG 이미지 데이터 형태

현재 확인한 PRPD PNG 샘플은 다음 형태다.

| 항목 | 값 |
| --- | --- |
| 해상도 | 256 x 256 |
| 색상 포맷 | RGB |
| Pixel format | 24-bit RGB |

VLM 입력으로 사용할 때는 대부분의 processor가 내부적으로 resize/normalize를 수행한다. 원본 PRPD 이미지는 256x256 RGB 이미지로 보고 시작한다.

## 11. JSON 어노테이션 구조

JSON 파일은 크게 `label`과 `metadata`로 구성된다.

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

JSON 파일은 UTF-8 BOM이 포함될 수 있으므로 Python에서 읽을 때는 `utf-8-sig`를 사용한다.

```python
import json
from pathlib import Path

data = json.loads(Path("sample.json").read_text(encoding="utf-8-sig"))
```

주요 필드:

| Field | 사용 여부 | 설명 |
| --- | --- | --- |
| `label.PD_type` | 필수 | 모델 정답 라벨 |
| `label.image_path` | 참조 | 매칭되는 PRPD 이미지 경로 |
| `label.timeseries_path` | 참조 | 매칭되는 CSV 시계열 경로 |
| `metadata.equipment_information.insulator_type` | 선택 | 절연체 종류 |
| `metadata.equipment_information.insulator_name` | 선택 | 절연체명 |
| `metadata.equipment_information.equipment_name` | 선택 | 전력 설비명 |
| `metadata.equipment_information.equipment_manufacturer` | 선택 | 제조사명 |
| `metadata.equipment_information.equipment_id` | 비추천 | 설비 고유 식별자. split leakage 가능성 있음 |
| `metadata.equipment_information.equipment_rated_voltage` | 선택 | 정격 전압 |
| `metadata.equipment_information.equipment_rated_current` | 선택 | 정격 전류 |
| `metadata.environment.recording_time` | split 주의 | 측정 시각. group split에는 유용하나 feature 사용은 주의 |
| `metadata.environment.recording_time_length` | 참조 | 측정 길이. 현재 20 |
| `metadata.environment.power_supply_frequency` | 참조 | 전원 주파수. 현재 60Hz |
| `metadata.environment.sensor_type` | 선택 | 센서 타입. 대부분 HFCT, 일부 UHF |
| `metadata.environment.temperature` | 선택 | 온도 |
| `metadata.environment.humidity` | 선택 | 습도 |
| `metadata.environment.clearance_distance` | 선택 | 센서 이격 거리 |
| `metadata.discharge_information.defect_nums` | 비추천 | label leakage 가능성 있음 |
| `metadata.discharge_information.defect_details` | 비추천 | label leakage 가능성 있음 |
| `metadata.discharge_evaluation_factors.max_discharge_value` | 주의 | 실제 추론 시 얻을 수 있을 때만 사용 |

실제 JSON 예시:

```json
{
  "label": {
    "PD_type": 1,
    "image_path": "./원천데이터/노이즈/고체/ACSR-OC/노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.png",
    "timeseries_path": "./원천데이터/노이즈/고체/ACSR-OC/노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.csv"
  },
  "metadata": {
    "equipment_information": {
      "insulator_type": "고체",
      "insulator_name": "XLPE",
      "equipment_name": "ACSR-OC",
      "equipment_manufacturer": "금화전선",
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

## 12. 전력 설비 및 메타데이터 분포

데이터셋은 고체, 액체, 기체 절연체에 해당하는 9가지 전력 설비를 포함한다.

| 절연체 종류 | 전력 설비 |
| --- | --- |
| 고체 절연체 | TFR-CV, CNCV-W, ACSR-OC |
| 액체 절연체 | 단상 유입변압기, 전력용 유입변압기, 계기용 변압기 |
| 기체 절연체 | 7.2kV 배전반, 22.9kV 배전반, 25.8kV GIS |

현재 로컬 Train 기준 분포:

| 항목 | 분포 |
| --- | --- |
| 절연체 종류 | 고체 10,005 / 액체 10,005 / 기체 10,000 |
| 센서 종류 | HFCT 29,010 / UHF 1,000 |

설비 종류별 분포:

| 설비명 | 개수 |
| --- | ---: |
| ACSR-OC | 3,335 |
| CNCV-W | 3,335 |
| TFR-CV | 3,335 |
| 계기용 변압기 | 3,335 |
| 단상 유입변압기 | 3,335 |
| 전력용 유입변압기 | 3,335 |
| 7.2kV 배전반 | 2,500 |
| 22.9kV 배전반 | 2,500 |
| 25.8kV GIS | 5,000 |

## 13. 권장 manifest 형식

모델 학습 전에 전체 파일 매칭 정보를 하나의 manifest 파일로 정리한다.

현재 manifest 주요 컬럼:

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

최소 필수 컬럼:

```text
sample_id,json_path,image_path,timeseries_path,label_id,label_name
```

## 14. 전처리 규칙

### 14.1 기본 로딩

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

### 14.2 Shape validation

학습 전에 모든 CSV에 대해 shape을 확인한다.

```python
assert x.ndim == 2
assert x.shape[0] == 20
assert x.shape[1] == 7680
```

shape이 다른 파일이 있으면 제외, padding/truncation, resampling, 별도 split 관리 중 하나를 선택한다.

### 14.3 Normalization

권장 후보:

```text
1. sample-wise z-score
2. channel-wise z-score
3. train-set global mean/std normalization
4. median/IQR robust scaling
```

부분방전에서는 amplitude 크기 자체가 중요한 feature일 수 있으므로 normalization ablation을 권장한다.

```text
A. raw amplitude 유지
B. sample-wise z-score
C. channel-wise z-score
D. log / robust scaling
```

## 15. Feature 기반 분류용 feature 설계

`feature_logistic`, `feature_svm`, `feature_random_forest`, `feature_tabpfn`은 raw CSV를 직접 받는 모델이 아니라, CSV에서 추출한 feature vector를 입력으로 받는 tabular classifier다.

```text
CSV time-series
-> feature extraction
-> feature vector
-> Logistic / Linear SVM / RandomForest / TabPFN
-> 5-class prediction
```

현재 코드의 feature extractor는 `--feature-set` 옵션으로 column 수를 조절한다. 기본값은 `small`이다.

| Feature set | CSV feature columns | Metadata 포함 시 | 설명 |
| --- | ---: | ---: | --- |
| `small` | 64 | 74 | 가장 빠른 baseline. global/stat, FFT, amplitude histogram, pulse, cycle, half-cycle 사용 |
| `medium` | 128 | 138 | balanced baseline. phase-bin count/max 포함 |
| `full` | 182 | 192 | compact numeric PRPD histogram 96개 포함 |

`small` 기준 feature extractor는 다음 그룹을 만든다.

| Feature group | Examples |
| --- | --- |
| global amplitude/stat | mean, std, median, MAD, min/max, percentile, IQR, RMS |
| fixed FFT/spectral | 고정 band power ratio, spectral centroid/bandwidth/entropy/flatness |
| amplitude histogram | absolute amplitude 12-bin histogram |
| pulse feature | robust threshold 기반 pulse count/rate/peak/interval |
| cycle feature | 60Hz cycle별 peak/RMS/pulse count |
| half-cycle/circular | 양/음 반주기 pulse 차이, phase entropy/concentration |

`medium`은 `small`보다 amplitude histogram을 줄이고 24개 phase bin별 count/max를 추가한다. 실제 구현에서는 segment summary를 18개로 압축해 총 128개를 맞춘다.

`full`은 phase 12 bin x amplitude 8 bin의 compact numeric PRPD histogram 96개를 포함해 총 182개 feature를 만든다.

기본 feature-only baseline:

```text
feature_logistic --feature-set small
feature_svm --feature-set small
feature_random_forest --feature-set small
feature_tabpfn --feature-set small
```

## 16. 사용하면 안 되는 feature

다음 정보는 label leakage를 만들 수 있으므로 학습 feature로 사용하지 않는다.

```text
label.PD_type
label.image_path
label.timeseries_path
sample_id
파일명 문자열
폴더명 문자열
방전 유형명이 들어간 path token
label_name
metadata.discharge_information.defect_nums
metadata.discharge_information.defect_details
metadata.discharge_evaluation_factors.max_discharge_value
```

`metadata.discharge_evaluation_factors.max_discharge_value`는 주의가 필요하다. 실제 추론 시점에도 얻을 수 있는 센서 기반 평가값이면 사용할 수 있지만, 라벨링 과정에서 생성된 사후 평가값이면 leakage가 될 수 있다. 기본 실험에서는 제외하고 별도 ablation에서만 사용한다.

## 17. Train / Validation / Test split

가장 권장되는 방식은 AI-Hub 공식 train/validation/test split을 그대로 사용하는 것이다.

직접 split을 구성할 경우 단순 random split만 사용하면 같은 설비, 같은 날짜, 같은 측정 조건이 train과 test에 동시에 들어갈 수 있다. 이 경우 실제 일반화 성능보다 과대평가될 수 있다.

권장 split 기준:

```text
1. file-level split
2. equipment-level group split
3. recording date/time group split
4. voltage / sensor distance group split
5. insulator type group split
```

최소 규칙:

```text
동일 CSV 파일 내부 row를 서로 다른 split으로 나누지 않는다.
같은 JSON/CSV/PNG sample이 train/test에 동시에 들어가지 않게 한다.
파일명, 경로 기반 leakage를 제거한다.
모델별 성능 비교에서는 manifest의 split 컬럼을 고정하고 재사용한다.
```

현재 학습 runner는 manifest에 `split=train`과 `split=valid`가 함께 있으면 해당 split을 우선 사용한다. `split` 컬럼이 없거나 valid 행이 없으면 기존처럼 label stratified random split을 만든다.

1차 모델 비교용 고정 split은 다음 명령으로 생성한다.

```powershell
python ml/scripts/make_splits.py --manifest Train/manifest.csv --output Train/manifest_random_split_seed42.csv --valid-ratio 0.2 --seed 42
```

## 18. 평가 지표

Accuracy만 사용하면 class imbalance나 특정 방전 유형 실패를 놓칠 수 있다. 다음 metric을 함께 기록한다.

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

특히 현장 진단에서는 실제 부분방전 클래스 recall을 반드시 확인한다.

```text
normal recall
noise recall
surface_discharge recall
corona_discharge recall
void_discharge recall
실제 방전인데 정상으로 예측한 개수
```

## 19. 추천 baseline 구성

### 19.1 Feature-only baseline

```text
feature_logistic
feature_svm
feature_random_forest
feature_tabpfn
```

입력:

```text
stat feature + pulse feature + phase-bin feature + FFT feature + numeric PRPD histogram
```

### 19.2 Raw time-series baseline

```text
GRU
InceptionTime
PatchTST
TimesNet
MOMENT
ModernTCN
MiniROCKET / MultiROCKET
```

입력:

```text
[B, 20, 7680]
또는 모델 wrapper에 따라 [B, 7680, 20]
```

### 19.3 Hybrid model

```text
raw time-series encoder embedding + hand-crafted feature embedding
-> concat
-> classifier
```

### 19.4 Late fusion / stacking

```text
feature model probability
raw time-series model probability
-> concat probabilities
-> meta classifier
-> final prediction
```
