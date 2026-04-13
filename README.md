# Loan Approval Classification Project

## 1. Project Overview
공개 Loan Prediction 데이터를 이용해 대출 승인 여부를 예측하는 이진 분류 프로젝트입니다.
데이터 로드, 결측치 처리, DB 적재/조회, 시각화, 전처리, 모델 비교를 수행했습니다.

## 2. Dataset
This project uses the public Loan Prediction dataset.

- Source: [Kaggle Loan Prediction Dataset](https://www.kaggle.com/datasets/shaijudatascience/loan-prediction-practice-av-competition)
- Original context: Analytics Vidhya practice problem
- Please download the dataset manually and place it in the project directory as `train_csv.csv`.

### Features
- Loan_ID
- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status

## 3. Preprocessing
- 범주형 결측치: 최빈값 처리
- LoanAmount: 중앙값 처리
- Loan_Amount_Term: 최빈값 처리
- Credit_History: -1로 별도 처리
- Loan_Status: Y/N -> 1/0
- Dependents: "3+" -> 3
- One-hot encoding 적용

## 4. Database Workflow
- CSV 데이터를 읽고 전처리
- MySQL `loan_train` 테이블에 저장
- DB에서 다시 조회하여 분석 및 모델링에 사용

## 5. Visualization
- 수치형 변수 히스토그램
- 범주형 변수별 승인 여부 그래프
- 수치형 변수의 승인 여부별 박스플롯
- 산점도
- 상관관계 히트맵

## 6. Models
- Logistic Regression
- KNN
- SVM

## 7. Results
### Train/Test Split Results

| Model | Scaling | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Logistic Regression | No | 0.6703 | 0.6746 | 0.9500 | 0.7889 |
| KNN | No | 0.5622 | 0.6275 | 0.8000 | 0.7033 |
| SVM | No | 0.6486 | 0.6486 | 1.0000 | 0.7869 |
| Logistic Regression | Yes | 0.6703 | 0.6746 | 0.9500 | 0.7889 |
| KNN | Yes | 0.6649 | 0.6959 | 0.8583 | 0.7687 |
| SVM | Yes | 0.7622 | 0.7533 | 0.9417 | 0.8370 |

### K-Fold Cross Validation Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6906 | 0.7166 | 0.9131 | 0.8018 |
| KNN | 0.6222 | 0.6853 | 0.8340 | 0.7516 |
| SVM | 0.6841 | 0.6864 | 0.9954 | 0.8121 |

## 8. Key Findings
- Credit_History가 승인 여부와 가장 관련성이 커 보였다.
- KNN은 스케일링 전 성능이 낮았고, 스케일링 후 개선되었다.
- 최종적으로 SVM + Standard Scaling이 가장 높은 F1-score를 보였다.

## 9. Conclusion
본 프로젝트에서는 SVM이 가장 우수한 성능을 보였다.
특히 스케일링이 거리/경계 기반 모델 성능에 중요한 영향을 줄 수 있음을 확인했다.
