'''
노트북 가격 예측
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
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

train = pd.read_csv('laptop_train.csv')
test = pd.read_csv('laptop_test.csv')
# print(train.head())
# print(test.head())

# print(train.shape, test.shape)  # (91, 10) (39, 9)

# print(train.info())
# print(test.info())

# --- 결측치 ---
# print(train.isnull().sum()) 
# print(test.isnull().sum())  # ['Model', 'Series', 'Processor', 'Processor_Gen', 'RAM', 'Hard_Disk_Capacity', 'OS']

cat_cols_na = ['Model', 'Series', 'Processor', 'Processor_Gen', 'RAM', 'Hard_Disk_Capacity', 'OS']

for col in cat_cols_na:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

train['RAM'] = train['RAM'].fillna(train['RAM'].median())
test['RAM'] = test['RAM'].fillna(test['RAM'].median())

# print(train.isnull().sum()) 
# print(test.isnull().sum())

# --- 타겟 분리 ---
target = train.pop('Price')
# print(train.shape)  # (91, 9)

# --- 인코딩 ---
df = pd.concat([train, test])

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

train = df.iloc[:len(train)]
test = df.iloc[len(train):]

# --- 검증 데이터 분할 ---
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)   # (72, 9) (19, 9) (72,) (19,)

# --- 머신러닝 학습 및 평가 ---
# model = lgb.LGBMRegressor(random_state=0, verbose=-1)
# model.fit(X_train, y_train)
# pred = model.predict(X_val)
# result = root_mean_squared_error(y_val, pred)
# print(result)   # 24303.330738904362

rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)
pred = rf.predict(X_val)
result = root_mean_squared_error(y_val, pred)
# print(result)   # 16448.179551465968

# --- 예측 결과 파일 생성 ---
pred = rf.predict(test)
submit = pd.DataFrame({'pred':pred})
submit.to_csv('result.csv', index=False)
print(pd.read_csv('result.csv').head())