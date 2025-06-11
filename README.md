# Title
### Wine Quality Prediction using Machine Learning
# Member
양시영 건설환경공학과 2021042424 kongou1324@gmail.com

# 1. Overview
## 1.1 Red to White Generalization
Red Wine 데이터를 통해 모델을 학습하고, 이를 White Wine 데이터에 적용해 일반화 성능을 평가하는 프로젝트입니다. 와인의 11개의 수치형 화학 성분을 통해 품질을 예측하며, Wine의 종류가 바뀌었을 때 발생하는 domain shift 현상에 대해 분석합니다.

# 2. Dataset
- `winequality-red.csv` (1,599 samples)
- `winequality-white.csv` (4,898 samples)
- Columns: 11개의 수치형 화학 성분 + `quality` (score: 0~10)

# 3. Analysis based on strategy
## 3.1 Random Forest
### 3.1.1 Labeling Strategy
- 원본 'quality' 는 정수 점수(0~10)
- 본 프로젝트에서는 이진 분류로 변환:
  - `quality 6 이상` → **good (1)**
  - `quality 5 이하` → **bad (0)**

### 3.1.2 Baseline model - Random Forest
Red Wine 데이터로 학습 후, 같은 도메인(red) 및 다른 도메인(white)에서 평가를 진행


### 3.1.3 Red Wine Test Results
![Confusion Matrix - Red](image/rf_confusion_red.png)
- Accuracy: 79.4%

### 3.1.4 White Wine Test Results
![Confusion Matrix - White](image/rf_confusion_white.png)

