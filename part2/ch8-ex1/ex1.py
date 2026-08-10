'''
항공권 티켓 가격 예측
- 예측할 컬럼： price
- RMSE 평가지표
- 제출
    - pred: 예측값（가격） 
    - 제출 파일명： 'result. csv' 
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

train = pd.read_csv('flight_train.csv')
test = pd.read_csv('flight_test.csv')
# print(train.head())
# print(test.head())

# print(train.shape, test.shape)  # (10505, 11) (4502, 10)

# print(train.info())
# print(test.info())

# --- 타겟 분리 ---
target = train.pop('price')
# print(train.shape)

# --- 컬럼 삭제 ---
train = train.drop('airline', axis=1)
test = test.drop('airline', axis=1)

# --- 인코딩 ---
cols = train.select_dtypes(include='object').columns
# print(cols)
    # ['flight', 'source_city', 'departure_time', 'stops', arrival_time', 'destination_city', 'class']

df = pd.concat([train, test])
# print(df.shape) # (15007, 9)

le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])    

# df_oh = pd.get_dummies(df)
train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):].copy()

# --- 검증 데이터 분할 ---
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)   # (8404, 1261) (2101, 1261) (8404,) (2101,)

# --- 머신러닝 학습 및 평가 ---
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)
pred = rf.predict(X_val)
result = root_mean_squared_error(y_val, pred)
print(result)   # 3785.4225749390203

# --- 예측 결과 파일 생성 ---
pred = rf.predict(test)
submit = pd.DataFrame({'pred':pred})
submit.to_csv('result.csv', index=False)
print(pd.read_csv('result.csv').head())