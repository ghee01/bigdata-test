'''
중고차 가격 예측
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

train = pd.read_csv('car_train.csv')
test = pd.read_csv('car_test.csv')
# print(train.head())
# print(test.head())

# print(train.shape, test.shape)  # (6732, 17) (5772, 16)

# print(train.info())
# print(test.info())

# --- 타겟 분리 ---
target = train.pop('Price')
# print(train.shape)

# --- 인코딩 ---
df = pd.concat([train, test])

cat_cols = df.select_dtypes(include='object').columns
# print(cat_cols)
    # ['Levy', 'Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Mileage', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color']

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

train = df.iloc[:len(train)]
test = df.iloc[len(train):]

# --- 검증 데이터 분할 ---
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)   # (5385, 16) (1347, 16) (5385,) (1347,)

# --- 머신러닝 학습 및 평가 ---
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)
pred = rf.predict(X_val)
result = root_mean_squared_error(y_val, pred)   # 10349.216597976096
# print(result)

# --- 예측 결과 파일 생성 ---
pred = rf.predict(test)
submit = pd.DataFrame({'pred':pred})
submit.to_csv('result.csv', index=False)
print(pd.read_csv('result.csv').head())