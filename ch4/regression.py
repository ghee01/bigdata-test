'''
bigdata-test/ch4/regression.py

빅분기 실기 ch4 작업형 2번 회귀(Regression) 실습

10개의 아울렛 매장에서 1,500여 개의 제품에 대한 판매 데이터를 수집했다.
예측 모델을 만들고 아울렛 특정 매장에서 각 제품의 판매금액을 예측하시오.
· 평가 기준은 RMSE로 평가
· label(target)은 판매금액(Item_Outlet_Sales) 
· 제출 파일은 예측값만 result.csv 파일로 생성해 제출（컬럼명 : pred, 1개)

-----------------------------------

1. 라이브러리 불러오기
2. 데이터 불러오기
3. 탐색적 데이터 분석 (EDA)
4. 데이터 전처리 - 인코딩, 스케일링
5. 검증 데이터 분할
6. 머신러닝 학습 및 평가
7. 예측 및 결과 파일 생성

'''

# --- 1. 라이브러리 불러오기 ---
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error # RMSE

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None) 

# --- 2. 데이터 불러오기 ---
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.head())
# print(test.head())

# --- 3. 탐색적 데이터 분석 (EDA) ---

## 행과 열의 개수
# print(train.shape, test.shape)  # (6818, 12) (1705, 11)

## 컬럼과 자료형 등
# print(train.info())
# print(test.info())

## 기초통계량
# print(train.describe()) # 숫자형
# print(train.describe(include='O')) # 범주형
# print(test.describe())  # 숫자형
# print(test.describe(include='O')) # 범주형

## 결측치
# print(train.isnull().sum())
# print(test.isnull().sum())

## 타겟 변수 분포(히스토그램)
# print(train['Item_Outlet_Sales'].hist())

# --- 4. 데이터 전처리 - 인코딩, 스케일링 ---

## 범주형(object 컬럼 목록 탐지)
# print(train.columns[train.dtypes == 'object'])
#   ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

## 인코딩 대상 컬럼 지정
cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

## target(매출액) 분리
target = train.pop('Item_Outlet_Sales')
# print(target)
# print(train.shape, test.shape)  # (6818, 11) (1705, 11)

## train + test
df = pd.concat([train, test])
# print(df.shape) # (8523, 11)

## 레이블 인코딩
le = LabelEncoder()

for col in cols:
    df[col] = le.fit_transform(df[col])

# print(df.head())

## 다시 원래 길이대로 train/test 분리
train = df.iloc[:len(train)].copy()
test = df.iloc[len(train):].copy()

# print(train.shape, test.shape)  # (6818, 11) (1705, 11)

## 결측치 처리
##  - Item_Weight (상품 무게) → 최소값
##  - Outlet_Size (매장 크기) → 최빈값
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].min())
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
# print(train.isnull().sum())

test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].min())
test['Outlet_Size'] = test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])
# print(test.isnull().sum())

## 예측에 도움 안 되는 식별자(ID) 컬럼 제거
train = train.drop('Item_Identifier', axis=1)
test = test.drop('Item_Identifier', axis=1)
# print(train.shape, test.shape)  # (6818, 10) (1705, 10)

# --- 5. 검증 데이터 분할 ---

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=0)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)   # (5454, 10) (1364, 10) (5454,) (1364,)

# --- 6. 머신러닝 학습 및 평가 ---

# lr = LinearRegression()

# lr.fit(X_train, y_train)

## 회귀는 predict()가 바로 예측값(숫자) 반환
# y_pred = lr.predict(X_val)
# print(y_pred)   # [3062.23766209 1553.92870689 1563.56372011 ... 1999.04752946 4090.88328544 3402.82766932]

## 평가 : RMSE - 작을수록 좋다
# result = root_mean_squared_error(y_val, y_pred)
# print(result)   # 1132.6619411737074

## 랜덤포레스트 회귀 - 여러 결정트리를 앙상블하여 비선형 관계를 찾아낸다 / 효과 좋은 편
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)    # 학습
y_pred = rf.predict(X_val)  # 예측
result = root_mean_squared_error(y_val, y_pred) # 평가
# print(result)   # 1049.6679530854844

## LightGBM 회귀 - 부스팅 계열, 보통 랜덤포레스트보다 성능 좋다
# model = lgb.LGBMRegressor(random_state=0, verbose=-1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val)
# result = root_mean_squared_error(y_val, y_pred)
# print(result)   # 1056.2454015155554

# --- 7. 예측 및 결과 파일 생성 ---

## 최종 선택한 모델로 실제 test 데이터 예측
pred = rf.predict(test)
# print(pred) # [1536.79956   787.135392 2192.499374 ... 4095.70199   967.493954   2001.12848 ]

## 데이터프레임으로 생성
submit = pd.DataFrame({'pred':pred})
# print(submit.head())

## CSV로 내보내기
submit.to_csv('result.csv', index=False)
print(pd.read_csv('result.csv').head())