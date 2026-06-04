# PRD: 부분방전 시계열 및 소형 VLM 진단 프로젝트

## 1. 목적

이 프로젝트는 AI-Hub 산업 설비 부분방전 데이터를 사용해 다음 두 가지 모델 개발을 목표로 한다.

1. CSV 시계열 데이터만 사용하는 5-class 부분방전 분류 모델
2. PRPD 이미지, 메타데이터, 시계열 요약 정보를 함께 사용하는 소형 VLM 진단 모델

이 프로젝트는 단일 이미지 비전 분류 프로젝트가 아니다. ResNet, EfficientNet 같은 PRPD 이미지 단독 분류 모델 개발은 핵심 트랙에서 제외한다.

## 2. 데이터 형태

로컬 학습 데이터는 `Train/` 아래에 있다.

현재 `Train/manifest.csv` 기준 데이터 형태:

- 전체 샘플 수: `30,010`
- 라벨 분포: `0~4` 각 `6,002`개
- 샘플 하나는 세 파일을 연결한다.
  - `timeseries_path`: 부분방전 CSV 시계열
  - `image_path`: PRPD PNG 이미지
  - `json_path`: 라벨 및 메타데이터 JSON
- CSV shape: `(20, 7680)`
  - `20`: pseudo-channel 또는 측정 segment 차원
  - `7680`: time axis

라벨 매핑:

| ID | 클래스 |
| --- | --- |
| 0 | 정상 |
| 1 | 노이즈 |
| 2 | 표면 방전 |
| 3 | 코로나 방전 |
| 4 | 보이드 방전 |

실험 코드는 `manifest.csv`의 `label_id`를 사용한다. manifest를 다시 생성할 때는 JSON의 `label.PD_type`을 기준으로 라벨을 만들어야 한다.

## 3. 시계열 트랙

시계열 작업은 forecasting이 아니라 classification이다.

기본 입력:

```text
CSV -> tensor shape (20, 7680) 또는 transpose 후 (7680, 20)
```

Core 모델:

```text
GRU
InceptionTime
PatchTST
TimesNet
MOMENT
```

Extended GPU 모델:

```text
TCN
ResNet1D
ModernTCN
iTransformer
TimeMixer
UniTS
GPT4TS / One-Fits-All
TS2Vec
```

CPU-only optional baseline:

```text
MiniROCKET
MultiROCKET
HYDRA
feature_logistic
feature_svm
feature_random_forest
feature_tabpfn
```

Extended 중 훈련 시간이 특히 길어질 수 있는 모델:

```text
TimeMixer
UniTS
GPT4TS
TS2Vec
```

위 모델들은 전체 `30,010`개 샘플로 바로 돌리지 않고, 반드시 small subset smoke run부터 시작한다.

## 4. Feature Baseline 트랙

Feature baseline은 raw 시계열 모델이 아니라, 시계열에서 추출한 feature를 입력으로 받는 tabular classifier다.

파이프라인:

```text
CSV 신호
-> amplitude / pulse / cycle / phase-bin / FFT / numeric PRPD histogram feature
-> Logistic / Linear SVM / RandomForest / TabPFN
-> 5-class prediction
```

즉 원본 CSV가 처음부터 table 데이터인 것이 아니라, CSV 하나를 feature vector 하나로 변환해서 table을 만든다. 기본값은 `--feature-set small`이며, 처음 baseline은 작은 feature set부터 시작한다.

```text
원본 입력 1개: (20, 7680) CSV
변환 후 입력 1개: 64개 tabular feature column

전체 Train 기준:
X.shape = (30010, 64)
y.shape = (30010,)
target = label_id
```

feature set별 column 수:

| Feature set | CSV feature columns | Metadata 포함 시 | 사용 목적 |
| --- | ---: | ---: | --- |
| `small` | 64 | 74 | 가장 빠른 baseline. global/stat, FFT, amplitude histogram, pulse, cycle, half-cycle 사용 |
| `medium` | 128 | 138 | balanced baseline. phase-bin count/max 포함 |
| `full` | 182 | 192 | compact numeric PRPD histogram 96개 포함 |

`--include-metadata`를 켜면 안전한 numeric metadata whitelist 10개가 추가된다.

현재 구현 기준 `small` feature column 구성:

| Feature group | Column count | 내용 |
| --- | ---: | --- |
| Global amplitude/stat | 14 | mean, std, median, MAD, min/max, percentile, IQR, RMS |
| Fixed FFT/spectral | 8 | 고정 band power ratio, spectral centroid/bandwidth/entropy/flatness |
| Amplitude histogram | 12 | 전체 absolute amplitude 12-bin histogram |
| Pulse features | 10 | robust threshold 기반 pulse count/rate/peak/interval 요약 |
| Cycle features | 12 | 60Hz cycle별 peak/RMS/pulse count 요약 |
| Half-cycle/circular | 8 | 양/음 반주기 pulse 차이, phase entropy/concentration |
| Total CSV features | 64 | metadata 제외 |

현재 구현 기준 `medium` feature column 구성:

| Feature group | Column count | 내용 |
| --- | ---: | --- |
| Global amplitude/stat | 16 | `small` global + skewness/kurtosis |
| Fixed FFT/spectral | 10 | `small` FFT + dominant frequency/power ratio |
| Pulse features | 12 | `small` pulse + peak width/burstiness 확장 |
| Cycle features | 16 | cycle feature 확장 |
| Segment summary | 18 | 20개 segment의 6개 feature를 mean/std/max로 압축 |
| Phase-bin count/max | 48 | 24개 phase bin별 event count + max amplitude |
| Half-cycle/circular | 8 | 양/음 반주기 및 phase 집중도 |
| Total CSV features | 128 | metadata 제외 |

현재 구현 기준 `full` feature column 구성:

| Feature group | Column count | 내용 |
| --- | ---: | --- |
| Global amplitude/stat | 16 | robust global amplitude 통계 |
| Fixed FFT/spectral | 10 | 고정 spectral 요약 |
| Pulse features | 12 | robust pulse 요약 |
| Cycle features | 16 | 60Hz cycle 요약 |
| Segment summary | 24 | 20개 segment의 6개 feature를 mean/std/min/max로 압축 |
| Half-cycle/circular | 8 | 반주기 및 phase 집중도 |
| Numeric PRPD histogram | 96 | phase 12 bin x amplitude 8 bin histogram |
| Total CSV features | 182 | metadata 제외 |
| Optional metadata features | 10 | `--include-metadata` 사용 시 안전한 숫자 필드만 추가 |

메타데이터는 기본적으로 제외한다. 메타데이터를 사용할 때도 안전한 숫자 whitelist만 사용한다. 파일 경로, 파일명, sample ID, label name, defect detail처럼 정답 정보가 섞일 수 있는 값은 feature로 사용하지 않는다.

## 5. VLM 트랙

VLM 프롬프트에 원본 CSV 전체 row를 텍스트로 넣지 않는다.

추천 VLM 입력:

```text
PRPD PNG 이미지
+ JSON 메타데이터 텍스트
+ 시계열 모델 예측값 / confidence
+ 시계열 feature 요약
```

우선 후보 모델:

```text
Qwen2.5-VL-3B-Instruct
```

smoke 또는 저자원 후보:

```text
Qwen2-VL-2B-Instruct
```

학습 전략:

```text
QLoRA SFT
초기에는 vision encoder freeze
LLM/projector LoRA 우선 학습
자연어보다 구조화된 JSON 진단 출력 우선
```

목표 VLM 출력 예시:

```json
{
  "label_id": 3,
  "label_name": "코로나 방전",
  "risk_level": "주의",
  "reason": "PRPD 패턴, 메타데이터, 시계열 근거를 종합한 짧은 진단 설명"
}
```

## 6. 초기 EDA

현재 Train 데이터와 manifest 기준으로 초기 EDA를 한 번 실행해 기준 리포트를 만든다. 이후 매 모델 훈련 전에 반복 실행할 필요는 없다. 데이터 압축 해제 상태, `manifest.csv`, label mapping, feature 설계, split 정책이 바뀐 경우에만 다시 실행한다.

기본 실행:

```powershell
python ml/scripts/run_eda.py
```

기본값은 전체 manifest 30,010개를 요약하고, 실제 CSV 신호는 class-balanced `sample-size=500`개만 읽어 빠르게 시각화한다.

산출물 기본 위치:

```text
results/eda/
```

주요 산출물:

| 파일 | 목적 |
| --- | --- |
| `eda_summary.json` | 전체 row 수, label 분포, 누수 위험 컬럼 존재 여부, missing path 개수 |
| `label_distribution.csv/png` | 클래스 균형 확인 |
| `metadata_distributions.png` | 절연체, 설비, 센서 분포 확인 |
| `signal_summary_sample.csv` | 샘플 CSV별 RMS, p99, max_abs, pulse_rate 등 통계 |
| `signal_stats_by_class.png` | 클래스별 신호 통계 boxplot |
| `phase_pulse_distribution.png` | 60Hz phase-bin 기반 pulse 분포 |
| `sample_waveforms_by_class.png` | 클래스별 원시 waveform 예시 |
| `class_mean_abs_waveform.png` | 클래스별 평균 absolute waveform |

전체 CSV 30,010개를 모두 읽는 signal-level EDA는 시간이 오래 걸릴 수 있으므로 명시적으로만 실행한다.

```powershell
python ml/scripts/run_eda.py --full-signal-eda
```

초기 EDA에서 확인할 항목:

- label `0~4`가 균형인지
- `timeseries_path`, `image_path`, `label_name`, 파일명 문자열처럼 label leakage 위험이 있는 컬럼이 feature로 들어가지 않는지
- 모든 CSV shape이 `(20, 7680)`인지
- 클래스별 RMS, p99, max_abs, pulse_rate 분포가 어떻게 다른지
- 60Hz phase-bin pulse 분포가 클래스별로 다른지
- 특정 설비, 센서, 절연체가 특정 label에만 몰려 있지 않은지

## 7. 실행 규칙

`train.py`는 한 번 실행할 때 정확히 하나의 neural model만 훈련해야 한다.

허용:

```powershell
python train.py --model gru --sample-size 100
python train.py --model moderntcn --sample-size 100
```

CPU-only baseline은 `train.py`가 직접 학습하지 않는다. `train.py --list-models`의 `cpu_only` 목록은 안내용이며, 각 모델은 전용 runner로 하나씩 실행한다.

예:

```powershell
python ml/scripts/run_feature_baseline.py --model logistic
python ml/scripts/run_minirocket.py
python ml/scripts/run_multirocket.py
python ml/scripts/run_sktime_classifier.py --model catch22
```

금지:

```powershell
python train.py --model core
python train.py --model extended
python train.py --model gru,patchtst
```

GPU neural model은 `.venv`를 사용한다.

CPU-only baseline은 `.venv` 또는 `.venv314t`를 사용할 수 있다.

- `.venv`: 안정적인 기본 환경
- `.venv314t`: free-threaded Python 3.14t 환경. CPU-only smoke 및 멀티스레드 실험용

## 8. 성공 기준

최소 성공 기준:

- manifest 기반 dataset loading이 동작한다.
- Core 시계열 모델 하나 이상이 학습 및 평가된다.
- leakage-safe feature baseline이 동작한다.
- metric은 accuracy, macro F1, per-class F1, confusion matrix를 포함한다.
- VLM 전략과 데이터 변환 계획이 문서화되어 있다.

강한 포트폴리오 기준:

- Core 시계열 모델들이 비교된다.
- MiniROCKET/MultiROCKET 및 feature baseline이 비교된다.
- pretrained 또는 foundation 시계열 모델 하나 이상을 fine-tuning 또는 probing한다.
- PRPD 이미지 + 메타데이터 + 시계열 요약으로 VLM instruction dataset을 생성한다.
- 소형 VLM이 구조화된 JSON 진단 결과를 생성한다.

## 9. 상세 문서

- `docs/PRD.md`: 상세 프로젝트 PRD
- `docs/DATASET_EXPLAIN.md`: 통합 데이터 구조 및 manifest 설명
- `docs/TIMESERIES_MODELS.md`: 시계열 모델 후보 및 실험 순서
- `docs/MODEL_IMPLEMENTATION_SOURCES.md`: 공식 구현체 및 wrapper 정책
- `docs/VLM_STRATEGY.md`: VLM 모델 및 fine-tuning 전략
- `AGENT.md`: 코딩 및 실행 규칙
