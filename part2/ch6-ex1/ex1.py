'''
당뇨병 여부 예측
- 예측할 컬럼： Outcome (0: 정상 , 1: 당뇨병)
- ROC-AUC 평가지표
- 제출
    - pred: 예측값（당뇨병일 확률)
    - 파일명： 'result.csv'
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train = pd.read_csv('diabetes_train.csv')
test = pd.read_csv('diabetes_test.csv')

# print(train.head())
# print(test.head())

# print(train.shape, test.shape)  # (614, 9) (154, 8)

# print(train.info())
# print(test.info())

# --- 결측치 ---
# print(train.isnull().sum())
# print(test.isnull().sum())

# --- 타겟 분리 ---
# print(train['Outcome'].value_counts())
y_train = train.pop('Outcome')

# --- 데이터 분할 ---
X_train, X_val, y_train, y_val = train_test_split(train, y_train, test_size=0.2, random_state=0)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)   # (491, 8) (123, 8) (491,) (123,)

# --- 랜덤포레스트 분류 ---
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
pred = rf.predict_proba(X_val)
# print(pred[:10])
# print(rf.classes_)    # [0 1]

# --- 평가 지표 ---
roc_auc = roc_auc_score(y_val, pred[:, 1]) 
# print(roc_auc)  # 0.8002739726027398

# --- 최종 파일 저장 ---
pred = rf.predict_proba(test)
submit = pd.DataFrame({'pred':pred[:, 1]})
submit.to_csv('result.csv', index=False)
print(pd.read_csv('result.csv').head())