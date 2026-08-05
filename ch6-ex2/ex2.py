'''
이직 여부 예측
- 예측할 컬럼： target(0: 새 일자리를 찾지 않음 , 1: 새 일자리를 찾음
- ROC-AUC 평가지표
- 제출
    - pred: 예측값（이직할 확률)
    - 파일명： 'result. csv'
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train = pd.read_csv('hr_train.csv')
test = pd.read_csv('hr_test.csv')
# print(train.head())
# print(test.head())
# print(train.shape, test.shape)  # (15326, 14) (3832, 13)

# print(train.info())
# print(test.info())

# print(train.select_dtypes(include='object').columns) 
# print(test.select_dtypes(include='object').columns)
# ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']

# --- 결측치 ---
# print(train.isnull().sum())
# print(test.isnull().sum())
# ['gender', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']

for col in train.select_dtypes(include='object').columns:
    train[col] = train[col].fillna(train[col].mode()[0])
for col in test.select_dtypes(include='object').columns:
    test[col] = test[col].fillna(test[col].mode()[0])

# print(train.isnull().sum())
# print(test.isnull().sum())

# --- 인코딩 ---
data = pd.concat([train, test])
data_oh = pd.get_dummies(data)
train = data_oh.iloc[:len(train)]
test = data_oh.iloc[len(train):]

# print(train.shape, test.shape)  # (15326, 188) (3832, 188)

# --- 데이터 분할 ---
target = train.pop('target')

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)   # (12260, 187) (3066, 187) (12260,) (3066,)

# --- 머신러닝 학습 및 평가 ---
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

pred = rf.predict_proba(X_val)
# print(pred[:10])

roc_auc = roc_auc_score(y_val, pred[:, 1])
# print(roc_auc)  # 0.7587682793206625

# --- 결과 파일 저장 ---
# pred = rf.predict_proba(test)
# submit = pd.DataFrame({'pred':pred[:, 1]})
# submit.to_csv('result.csv', index=False)
# print(pd.read_csv('result.csv').head())
