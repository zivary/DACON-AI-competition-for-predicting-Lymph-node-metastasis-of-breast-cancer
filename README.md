# AI competition for predicting Lymph node metastasis of breast cancer ([DACON_유방암의 임파선 전이 예측 AI 경진대회](https://dacon.io/competitions/official/236011/overview/description))

## [배경]

- 림프절(임파선)은 암의 전이, 암이 퍼지는 데 매우 치명적인 역할을 합니다.

  병원에서 암 진단을 받았을 때 가장 많이 듣는 말이자 우리가 관심을 가져야 하는 것이 림프절 전이이며, 

  이러한 림프절 전이 여부에 따라 치료와 예후가 달라집니다.

  따라서 림프절 전이 여부와 전이 단계를 파악하는 것이 암을 치료하고 진단하는 것에 있어서 매우 핵심적인 역할을 합니다.

- 이번 '유방암의 임파선 전이 예측 AI경진대회'에서 유방암 병리 슬라이드 영상과 임상 항목 데이터를 이용하여,

  유방암 치료에 핵심적인 역할을 할 수 있는 최적의 AI 모델을 만들어 유방암의 임파선 전이 여부를 예측해 보고자 합니다.

## [주최 / 주관]

- 주최 : 연세대학교 의과대학, JLK, MTS
- 후원 : 보건산업진흥원
- 주관 : 데이콘

## [모델링]

- MIL(Multiple Instance Learning) 방식 적용
  - 여러 개의 인스턴스(샘플)로 이루어진 가방(bag) 데이터셋을 분류하는 머신러닝 알고리즘입니다.
  - 가방 데이터셋의 인스턴스중 일부는 양(positive) 클래스에 속하고, 일부 인스턴스는 음(negative) 클래스에 속합니다.
  - 가방의 각 인스턴스를 독립적으로 분류하는 것이 아니라, 전체 가방 데이터셋을 분류합니다.
  - MIL은 의학 분야를 비롯한 다양한 분야에서 사용되는 유용한 머신러닝 기술 중 하나입니다.
- 적용 조건
  - 각 Patch의 크기는 128*128 픽셀로 Crop합니다.
  - GRAYSCALE 값이 240 이하인 펙셀의 비울이 70% 이상인 Patch만 사용합니다.
  - 각 가방에는 10개의 Patch를 담습니다.
  - 각 환자의 최소 가방 수는 10개로 부족한 Patch는 해당 환자의 Patch를 증식하여 사용합니다.

## [평가 방법]

- Macro F1 Score로 결과를 평가합니다.
- Early Stopping F1 Score는 0.95로 합니다. 

## [참조]

- https://github.com/bupt-ai-cz/BALNMP

---

## [Setup]

### Clone this repo

```bash
$ git clone https://github.com/zivary/DACON_AI_competition_for_predicting_Lymph_node_metastasis_of_breast_cancer.git
```

### Environment

Create environment and install dependencies.

```bash
$ conda create -n env python=3.8 -y
$ conda activate env
$ pip install -r ./code/requirements.txt
```

### Data Preparation

```text
code/Data Preparation.ipynb 실행
```

### MIL Modeling

```text
code/MIL.ipynb 실행
```
