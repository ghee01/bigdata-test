# 라이브러리 불러오기
import pandas as pd

# 데이터 불러오기
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.head())
# print(test.head())

# print(train.info())
# print(test.info())

# print(train.shape, test.shape)

# print(train.describe()) # 기초통계량(숫자형)
# print(test.describe())
# print()
# print(train.describe(include='all'))
# print(test.describe(include='all'))
# print()
# print(train.describe(include='O'))  # 기초통계량(object형)
# print(test.describe(include='object'))

print(train.isnull().sum())
print(test.isnull().sum())