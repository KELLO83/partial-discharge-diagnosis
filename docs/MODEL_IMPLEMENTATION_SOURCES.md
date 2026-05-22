# 시계열 모델 공식 구현 소스 정리

이 문서는 부분방전 CSV 시계열 분류 실험에서 사용할 모델을 직접 구현할지, 공식 repo/package를 wrapper로 연결할지 결정하기 위한 기준 문서이다.

원칙:

- GRU처럼 PyTorch 기본 모듈로 안정적으로 표현되는 baseline은 직접 구현해도 된다.
- 논문 모델은 가능한 한 공식 repo, Hugging Face, 또는 널리 쓰이는 검증 라이브러리를 import해서 사용한다.
- 공식 구현이 forecasting 중심이면 classification head를 붙이는 adapter만 프로젝트 내부에 작성한다.
- 모델 본체를 프로젝트 내부에 임의로 재구현하지 않는다.

## 구현 소스 표

| 모델 | 우선 구현 소스 | 공식/준공식 여부 | Classification 적합성 | 우리 프로젝트 적용 방향 |
|---|---|---:|---:|---|
| GRU | PyTorch `torch.nn.GRU` | 공식 기본 모듈 | 높음 | 직접 구현 baseline 유지 |
| TCN | `locuslab/TCN`, 또는 `tsai` TCN | 공식/검증 라이브러리 | 높음 | 현재 `tsai` TCN wrapper 사용 |
| InceptionTime | `hfawaz/InceptionTime`, 또는 `tsai` InceptionTime | 공식/검증 라이브러리 | 높음 | 현재 `tsai` InceptionTime wrapper 사용 |
| ResNet1D | `tsai` ResNet 계열 | 검증 라이브러리 | 높음 | 현재 `tsai` ResNet wrapper 사용 |
| MiniROCKET | `sktime` `MiniRocketMultivariate` | 검증 라이브러리 | 높음 | Extended 후보이지만 CPU-only classical baseline으로 GPU train.py 라인업과 분리 |
| PatchTST | Hugging Face `PatchTSTForClassification`, 공식 `yuqinie98/PatchTST` | 공식/HF | 높음 | 현재 HF wrapper 방향 유지 |
| TimesNet | `thuml/Time-Series-Library` | 공식 | 높음 | 현재 TSLib 공식 wrapper 사용 |
| iTransformer | `thuml/iTransformer`, `thuml/Time-Series-Library` | 공식 | 중간 | 공식 구현은 forecasting 중심, classification adapter 검토 |
| TimeMixer | `kwuking/TimeMixer`, `thuml/Time-Series-Library`, PyPOTS/NeuralForecast 포함 구현 | 공식/검증 라이브러리 | 중간~높음 | 공식 repo가 classification 지원을 확장했으므로 공식 wrapper 우선 검토 |
| MOMENT | `momentfm`, `moment-timeseries-foundation-model/moment` | 공식 package/repo | 높음 | 공식 classification pipeline 사용 |
| UniTS | `mims-harvard/UniTS` | 공식 | 높음 | 공식 repo 기반 wrapper 작성 |
| GPT4TS / One-Fits-All | `DAMO-DI-ML/One_Fits_All` | 공식 | 중간 | GPT-2 adapter 기반 외부 구현 연결, classification head 필요 여부 검토 |
| TS2Vec | `yuezhihan/ts2vec` | 공식 | 높음 | representation encoder + classifier head로 사용 |

## 공식/준공식 링크

- GRU: PyTorch `torch.nn.GRU`
- TCN official: https://github.com/locuslab/TCN
- TCN tsai: https://timeseriesai.github.io/tsai/models.tcn.html
- InceptionTime official: https://github.com/hfawaz/InceptionTime
- tsai library: https://github.com/timeseriesAI/tsai
- MiniROCKET sktime: https://www.sktime.net/en/v0.20.0/api_reference/auto_generated/sktime.transformations.panel.rocket.MiniRocket.html
- PatchTST official: https://github.com/yuqinie98/PatchTST
- PatchTST Hugging Face: https://huggingface.co/docs/transformers/model_doc/patchtst
- Time-Series-Library: https://github.com/thuml/Time-Series-Library
- TimesNet official implementation source: https://github.com/thuml/Time-Series-Library
- iTransformer official: https://github.com/thuml/iTransformer
- TimeMixer official: https://github.com/kwuking/TimeMixer
- MOMENT official: https://github.com/moment-timeseries-foundation-model/moment
- UniTS official: https://github.com/mims-harvard/UniTS
- GPT4TS / One-Fits-All official: https://github.com/DAMO-DI-ML/One_Fits_All
- TS2Vec official: https://github.com/yuezhihan/ts2vec

## 현재 코드 상태 해석

현재 `ml/src/models`는 GRU를 제외하고 공식/외부 wrapper 방향으로 구성한다.

직접 구현 유지:

- `gru.py`: PyTorch `torch.nn.GRU` 기반 baseline

공식/외부 wrapper:

- `patchtst.py`: Hugging Face `PatchTSTForClassification` 우선 사용
- `tcn.py`: `tsai` TCN 또는 locuslab 공식 TCN adapter
- `inception_time.py`: `tsai` 또는 `hfawaz/InceptionTime` adapter
- `resnet1d.py`: `tsai` ResNet adapter
- `minirocket.py`: `sktime` 설치 후 sklearn-style pipeline으로 연결
- `moment.py`: `momentfm` 설치 후 `MOMENTPipeline` 연결
- `units.py`: `mims-harvard/UniTS` repo 연결
- `gpt4ts.py`: `DAMO-DI-ML/One_Fits_All` repo 연결
- `ts2vec.py`: `yuezhihan/ts2vec` repo 연결
- `timesnet.py`: `thuml/Time-Series-Library` repo 연결
- `itransformer.py`: `thuml/iTransformer` 또는 `thuml/Time-Series-Library` repo 연결
- `timemixer.py`: `kwuking/TimeMixer` 또는 `thuml/Time-Series-Library` repo 연결

따라서 공식 dependency가 설치 또는 clone되어 있지 않은 모델은 명확한 `ImportError`를 내는 것이 정상 동작이다. 임의 fallback 구현으로 논문 모델처럼 실험하지 않는다.

## 적용 우선순위

1. `PatchTST`: 이미 Hugging Face로 연결되어 있으므로 유지 및 config 정리
2. `MOMENT`: `momentfm` 설치 가능 여부 확인 후 classification pipeline 연결
3. `MiniROCKET`: CPU-only optional baseline으로 필요할 때 별도 runner 실행
4. `TimesNet`: `thuml/Time-Series-Library`를 vendor/clone 또는 submodule 방식으로 연결
5. `iTransformer`: 공식 repo 또는 TSLib 기반 adapter 검토
6. `TimeMixer`: 공식 repo의 classification 지원 방식 확인 후 adapter 작성
7. `TS2Vec`: 공식 encoder 학습 후 linear classifier head 연결
8. `UniTS`, `GPT4TS`: dependency와 입력 포맷이 무거우므로 foundation extended track으로 후순위 진행
