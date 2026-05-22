# 데이터셋 설명

## 데이터셋 개요

- 데이터셋명: 산업 설비 전기 화재 사고 예방 부분방전 데이터
- 영문 작업명: Industrial Electrical Fire Prevention Partial Discharge Dataset
- 출처: AI-Hub
- 데이터 페이지: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&dataSetSn=71682&topMenu=100
- 도메인: 산업 안전, 전력 설비 진단, 부분방전 진단
- 데이터 유형: 센서 시계열 데이터 + PRPD 이미지 + 메타데이터

이 데이터셋은 산업 설비에서 발생하는 부분방전(Partial Discharge, PD)을 진단하기 위한 멀티모달 데이터셋이다. 하나의 부분방전 샘플은 PRPD 이미지, 부분방전 시계열 CSV, 라벨/메타데이터 JSON으로 구성된다.

즉, 이 프로젝트에서는 하나의 샘플을 다음과 같이 바라보면 된다.

```text
하나의 부분방전 샘플
├── PRPD 이미지 (*.PNG)
├── 부분방전 시계열 데이터 (*.CSV)
└── 라벨 및 메타데이터 (*.JSON)
```

## 핵심 예측 목표

기본 예측 목표는 부분방전 유형을 5개 클래스로 분류하는 것이다.

| 라벨 ID | 라벨명 |
| --- | --- |
| 0 | 정상 |
| 1 | 노이즈 |
| 2 | 표면 방전 |
| 3 | 코로나 방전 |
| 4 | 보이드 방전 |

이 프로젝트에서는 단순 분류뿐 아니라, 최종적으로 VLM을 활용해 자연어 진단 결과를 출력하는 것도 목표로 둘 수 있다.

예시:

```text
진단 결과는 코로나 방전입니다. PRPD 이미지에서 특정 위상 구간에 방전 패턴이 집중되어 있으며,
시계열 신호의 최대 방전값과 RMS 값도 비정상 패턴을 보입니다.
```

## 데이터 규모

AI-Hub 데이터 페이지 기준 원천 데이터는 총 300,000건이다.

| 구분 | 개수 | 비율 |
| --- | ---: | ---: |
| 학습 데이터 | 239,980 | 80% |
| 검증 데이터 | 30,010 | 10% |
| 테스트 데이터 | 30,010 | 10% |
| 전체 | 300,000 | 100% |

## 파일 구성

실제 모델 개발에 사용할 주요 파일은 다음 세 가지다.

| 데이터 | 파일 형식 | 개수 | 역할 |
| --- | --- | ---: | --- |
| PRPD 이미지 데이터 | `.PNG` | 300,000 | VLM 또는 비전 입력 |
| 부분방전 시계열 데이터 | `.CSV` | 300,000 | 시계열 모델 입력 |
| 라벨/메타데이터 | `.JSON` | 300,000 | 라벨, 파일 경로, 설비/환경 정보 |

AI-Hub 페이지에는 원시 수집 데이터도 함께 설명되어 있다.

| 원시 데이터 | 파일 형식 | 개수 |
| --- | --- | ---: |
| PRPD 이미지 데이터 | `.BMP` | 861,150 |
| 부분방전 시계열 데이터 | `.CSV` | 861,150 |
| 복합센서 데이터 | `.XLSX` | 256 |
| 수집환경 데이터 | `.XLSX` | 256 |

이 프로젝트에서는 우선 원천 데이터인 `.PNG`, `.CSV`, `.JSON`을 기준으로 개발을 시작하는 것이 적절하다.

## 현재 로컬 Train 데이터 확인 결과

현재 로컬에 해제된 `Train` 데이터 기준으로 파일 수는 다음과 같다.

| 파일 형식 | 개수 | 위치 |
| --- | ---: | --- |
| `.png` | 30,010 | `Train/01.원천데이터` |
| `.csv` | 30,010 | `Train/01.원천데이터` |
| `.json` | 30,010 | `Train/02.라벨링데이터` |

압축 해제 후 `.zip` 파일은 제거했으며, 원천 파일과 라벨 파일만 남아 있다.

Train 데이터는 원천데이터와 라벨링데이터가 분리되어 있다.

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

`VS_`는 원천데이터(source)를 의미하고, `VL_`은 라벨링데이터(label)를 의미하는 것으로 볼 수 있다.

각 하위 폴더명은 다음 조합으로 구성된다.

```text
VS_{방전유형}_{절연체종류}_{설비명}
VL_{방전유형}_{절연체종류}_{설비명}
```

예시:

```text
VS_노이즈_고체_ACSR-OC
VL_노이즈_고체_ACSR-OC
```

## Train 라벨 분포

현재 해제된 Train 데이터는 5개 클래스가 동일한 개수로 구성되어 있다.

| 라벨 ID | 라벨명 | 개수 |
| --- | --- | ---: |
| 0 | 정상 | 6,002 |
| 1 | 노이즈 | 6,002 |
| 2 | 표면 방전 | 6,002 |
| 3 | 코로나 방전 | 6,002 |
| 4 | 보이드 방전 | 6,002 |
| 전체 | - | 30,010 |

절연체 종류별 분포는 다음과 같다.

| 절연체 종류 | 개수 |
| --- | ---: |
| 고체 | 10,005 |
| 액체 | 10,005 |
| 기체 | 10,000 |

센서 종류별 분포는 다음과 같다.

| 센서 | 개수 |
| --- | ---: |
| HFCT | 29,010 |
| UHF | 1,000 |

설비 종류별 분포는 다음과 같다.

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

## 하나의 샘플 구조

AI-Hub 설명에 따르면 PRPD 이미지와 부분방전 시계열 CSV는 하나의 부분방전 데이터에서 1:1로 매칭 생성된다.

따라서 JSON 파일을 기준으로 다음 정보를 연결해야 한다.

```text
JSON 파일
├── label.PD_type
├── label.image_path
└── label.timeseries_path
```

이 구조 때문에 가장 먼저 만들어야 할 것은 전체 데이터의 매칭 정보를 담은 `manifest.csv` 또는 `manifest.jsonl`이다.

예상되는 manifest 구조:

```text
json_path, image_path, timeseries_path, label_id, 설비 정보, 환경 정보
```

현재 로컬 Train 데이터에서는 `.csv`, `.png`, `.json` 파일의 base filename이 모두 일치한다.

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

따라서 실제 manifest 생성 시에는 다음 방식이 안전하다.

```text
1. JSON 파일을 기준으로 순회한다.
2. JSON 파일의 stem을 sample_id로 사용한다.
3. 같은 stem을 가진 PNG와 CSV를 원천데이터 폴더에서 찾는다.
4. label.PD_type과 metadata를 함께 추출한다.
```

주의할 점은 JSON 내부의 `label.image_path`, `label.timeseries_path`가 AI-Hub 논리 경로 형태라는 점이다.

예시:

```text
./원천데이터/노이즈/고체/ACSR-OC/노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.png
```

현재 해제된 로컬 폴더 구조는 `Train/01.원천데이터/VS_노이즈_고체_ACSR-OC/...` 형태이므로, JSON 내부 경로를 그대로 사용하기보다는 파일명 기준으로 실제 경로를 매칭하는 편이 적절하다.

## 파일명 규칙

현재 확인된 원천 파일명은 다음 형태를 따른다.

```text
{방전유형}_{절연체종류}_{설비명}_{측정일자}_{측정시간}_{센서종류}_{이격거리}.csv
{방전유형}_{절연체종류}_{설비명}_{측정일자}_{측정시간}_{센서종류}_{이격거리}.png
{방전유형}_{절연체종류}_{설비명}_{측정일자}_{측정시간}_{센서종류}_{이격거리}.json
```

예시:

```text
노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.csv
노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.png
노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.json
```

위 예시에서 추론할 수 있는 정보:

| 부분 | 값 | 의미 |
| --- | --- | --- |
| 방전유형 | 노이즈 | 라벨명 |
| 절연체종류 | 고체 | 절연체 분류 |
| 설비명 | ACSR-OC | 전력 설비명 |
| 측정일자 | 230910 | 2023-09-10 형식으로 해석 가능 |
| 측정시간 | 195222 | 19:52:22 형식으로 해석 가능 |
| 센서종류 | HFCT | 센서 |
| 이격거리 | 1000 | 센서 이격 거리로 추정 |

다만 최종 학습용 메타데이터는 파일명 파싱보다 JSON 값을 우선 사용하는 것이 안전하다.

## CSV 시계열 데이터 형태

현재 Train 샘플 기준 CSV 파일은 헤더가 없는 숫자 행렬 형태다.

확인된 샘플 구조:

```text
행 수: 20
열 수: 7,680
헤더: 없음
구분자: comma(,)
값 형태: 정수형 센서 값
```

샘플 200개를 확인했을 때 모두 다음 범위를 보였다.

| 항목 | 값 |
| --- | ---: |
| 행 수 | 20 |
| 열 수 | 7,680 |
| 파일 크기 | 약 460KB |

즉, 하나의 CSV는 대략 다음과 같은 2차원 배열로 로드할 수 있다.

```text
shape = (20, 7680)
```

주의할 점은 `20`축을 실제 센서 채널 수로 단정하면 안 된다는 것이다. JSON의 `recording_time_length`가 20이고 센서 타입은 보통 `HFCT` 또는 `UHF`로 기록되어 있으므로, 현재 단계에서는 다음처럼 해석하는 것이 더 안전하다.

```text
20 rows = 20개 측정 구간 또는 segment
7680 columns = 각 구간의 time points
```

따라서 모델 입력에서는 `20`축을 실제 physical channel이라기보다는 `pseudo-channel` 또는 `segment dimension`으로 취급한다.

모델 입력으로 사용할 때는 다음 선택지가 있다.

```text
방법 1: (20, 7680)을 pseudo-channel x time 형태로 사용
방법 2: flatten하여 길이 153,600의 1D sequence로 사용
방법 3: 행 단위 의미를 추가 확인한 뒤 segment x time 형태로 명확히 재정의
방법 4: 통계/FFT feature를 추출해 tabular feature로 사용
```

초기 시계열 baseline에서는 먼저 CSV를 `numpy.ndarray`로 로드하고, 전체 샘플의 shape이 항상 `(20, 7680)`인지 검증하는 것이 필요하다. 초기 모델링에서는 방법 1을 기본값으로 사용한다.

## PNG 이미지 데이터 형태

현재 확인한 PRPD PNG 샘플은 다음 형태다.

| 항목 | 값 |
| --- | --- |
| 해상도 | 256 x 256 |
| 색상 포맷 | RGB |
| Pixel format | 24-bit RGB |

예시 파일:

```text
Train/01.원천데이터/VS_노이즈_고체_ACSR-OC/
└── 노이즈_고체_ACSR-OC_230910_195222_HFCT_1000.png
```

VLM 입력으로 사용할 때는 대부분의 processor가 내부적으로 resize/normalize를 수행하지만, 원본 PRPD 이미지는 256x256 RGB 이미지라고 보고 시작하면 된다.

## JSON 어노테이션 구조

JSON 파일은 크게 `label`과 `metadata`로 구성된다.

```text
label
metadata
```

### 라벨 정보

| 필드 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `label.PD_type` | Number | 필수 | 부분방전 유형 라벨 |
| `label.image_path` | String | 필수 | 매칭되는 PRPD 이미지 경로 |
| `label.timeseries_path` | String | 필수 | 매칭되는 시계열 CSV 경로 |

예상 형태:

```json
{
  "label": {
    "PD_type": 3,
    "image_path": "path/to/image.png",
    "timeseries_path": "path/to/timeseries.csv"
  }
}
```

### 설비 메타데이터

| 필드 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `metadata.equipment_information.insulator_type` | String | 필수 | 절연체 종류 |
| `metadata.equipment_information.insulator_name` | String | 선택 | 절연체명 |
| `metadata.equipment_information.equipment_name` | String | 필수 | 전력 설비명 |
| `metadata.equipment_information.equipment_manufacturer` | String | 필수 | 제조사명 |
| `metadata.equipment_information.equipment_id` | String | 선택 | 설비 고유 식별 번호 |
| `metadata.equipment_information.equipment_rated_voltage` | String | 필수 | 정격 전압 |
| `metadata.equipment_information.equipment_rated_current` | String | 선택 | 정격 전류 |

예상 형태:

```json
{
  "metadata": {
    "equipment_information": {
      "insulator_type": "기체 절연체",
      "equipment_name": "22.9kV 배전반",
      "equipment_manufacturer": "제조사명",
      "equipment_rated_voltage": "22.9kV"
    }
  }
}
```

### 환경 메타데이터

| 필드 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `metadata.environment.recording_time` | String | 필수 | 부분방전 패턴 시작 시간 |
| `metadata.environment.recording_time_length` | Number | 필수 | 측정 시간 길이 |
| `metadata.environment.data_collector` | String | 필수 | 실험 수행자 식별 코드 |
| `metadata.environment.power_supply_id` | String | 선택 | 전원인가장치 식별 번호 |
| `metadata.environment.power_supply_voltage type` | String | 필수 | 전원인가장치 전원 종류 |
| `metadata.environment.power_supply_frequency` | String | 필수 | 정격 주파수 |
| `metadata.environment.power_supply_ramping_up_time` | String | 필수 | 전압 상승 속도 |
| `metadata.environment.power_supply_cutoff_current` | String | 필수 | 누설전류 임계치 |
| `metadata.environment.sensor_type` | String | 필수 | 센서 정보 |
| `metadata.environment.temperature` | String | 필수 | 주변 온도 |
| `metadata.environment.humidity` | String | 필수 | 주변 습도 |
| `metadata.environment.clearance_distance` | String | 필수 | 센서 이격 거리 |
| `metadata.environment.IEC_standard` | String | 필수 | 실험 규격 |
| `metadata.environment.engage_start_time` | String | 필수 | 설비 전압 인가 시작 시간 |

### 방전 관련 메타데이터

| 필드 | 타입 | 필수 여부 | 설명 |
| --- | --- | --- | --- |
| `metadata.discharge_information.defect_nums` | String | 선택 | 결함 모의 개수 |
| `metadata.discharge_information.defect_details` | Array | 선택 | 결함 세부 정보 |
| `metadata.discharge_evaluation_factors.max_discharge_value` | Number | 선택 | 방전 최대 크기 |

현재 JSON 파일은 UTF-8 BOM이 포함되어 있으므로 Python에서 읽을 때는 `utf-8-sig` 인코딩을 사용하는 것이 안전하다.

```python
import json
from pathlib import Path

path = Path("sample.json")
data = json.loads(path.read_text(encoding="utf-8-sig"))
```

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
      "data_collector": "KE-01, KE-02",
      "power_supply_id": "YWHVTR160701",
      "power_supply_voltage type": "AC",
      "power_supply_frequency": "60Hz",
      "power_supply_ramping_up_time": "1000V/s",
      "power_supply_cutoff_current": "60mA",
      "sensor_type": "HFCT",
      "temperature": "19",
      "humidity": "66",
      "clearance_distance": "['1000mm']",
      "IEC_standard": "IEC-60270",
      "engage_start_time": "230910-114044"
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

## 전력 설비 종류

데이터셋은 고체, 액체, 기체 절연체에 해당하는 총 9가지 전력 설비를 포함한다.

| 절연체 종류 | 전력 설비 |
| --- | --- |
| 고체 절연체 | TFR-CV, CNCV-W, ACSR-OC |
| 액체 절연체 | 단상 유입변압기, 전력용 유입변압기, 계기용 변압기 |
| 기체 절연체 | 7.2kV 배전반, 22.9kV 배전반, 25.8kV GIS |

## 권장 manifest 형식

모델 학습 전에 전체 파일 매칭 정보를 하나의 manifest 파일로 정리하는 것이 좋다.

권장 컬럼:

```text
sample_id
split
json_path
image_path
timeseries_path
label_id
label_name
insulator_type
equipment_name
equipment_rated_voltage
equipment_rated_current
temperature
humidity
sensor_type
clearance_distance
recording_time_length
max_discharge_value
```

예시:

```csv
sample_id,split,json_path,image_path,timeseries_path,label_id,label_name,equipment_name,temperature,humidity
000001,train,labels/000001.json,images/000001.png,timeseries/000001.csv,3,코로나 방전,22.9kV 배전반,25,60
```
