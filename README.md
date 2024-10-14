113年國道智慧交通管理創意競賽
===
# 專案緣起
我與小組隊友們參與了【113年國道智慧交通管理創意競賽】並獲選社會組優選
我在小組中負責的部分為資料預處理與模型訓練，並藉由此專案紀錄訓練模型的過程
# 資料探勘
首先觀察原始資料集有哪些欄位適合做為訓練模型的特徵，利用seaborn套件進行圖表視覺化，通過圖表更直觀地理解欄位之間的關聯性。
# 特徵工程
此專案的最終預測目標為預測事故處理時間
經過圖表分析與特徵篩選，我選擇了部分可能高度影響處理時間的欄位作為模型訓練的特徵，並針對類別型變量進行了編碼轉換，使用了 One-Hot Encoding 和 Label-Encoding 方法，以確保這些變量能夠被模型有效學習。
```python
df_columns = ['年', '月', '日', '時', '分', '國道名稱', '方向', '里程','處理分鐘', '事故類型', '死亡', '受傷',
        '內路肩', '內車道', '中內車道','中車道', '中外車道', '外車道', '外路肩','匝道','翻覆事故\n註記',
        '施工事故\n註記','危險物品車輛\n註記', '車輛起火\n註記', '冒煙車事故\n註記','主線中斷\n註記','肇事車輛']
df = df[df_columns]

df.fillna(0, inplace=True)

translation_dict_road = {
    '國道1號': 'National Highway 1',
    '國道2號': 'National Highway 2',
    '國道3號': 'National Highway 3',
    '國道4號': 'National Highway 4',
    '國道5號': 'National Highway 5',
    '國道6號': 'National Highway 6',
    '國3甲': 'National Highway 3A',
    '國道10號': 'National Highway 10',
    '國道8號': 'National Highway 8',
    '國道3甲': 'National Highway 3A',
    '港西聯外道路': 'Port West External Road',
    '南港連絡道': 'Nangang Link Road',
    '國2甲': 'National Highway 2A'
}

translation_dict_direction = {
    '南': 'South',
    '北': 'North',
    '西': 'West',
    '南向': 'Southbound',
    '北向': 'Northbound',
    '西向': 'Westbound',
    '東向': 'Eastbound',
    '東': 'East',
    '南北': 'South-North',
    '雙向': 'Two-way'
}

df['國道名稱'] = df['國道名稱'].map(translation_dict_road)
df['方向'] = df['方向'].map(translation_dict_direction)

df = df.dropna()

feature_columns = ['年','月','日','時','分','國道名稱','方向','里程','處理分鐘','事故類型','死亡','受傷',
           '內路肩','內車道','中內車道','中車道','中外車道','外車道','外路肩','匝道','翻覆事故註記',
          '施工事故註記','危險物品車輛註記','車輛起火註記','冒煙車事故註記','主線中斷註記','肇事車輛']
df.columns = feature_columns
```
**One-Hot Encoding**
**Label-Encoding**
```python
#進行編碼
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = df.astype(str)
#進行Label Encoding
le_road = LabelEncoder()
df['國道名稱'] = le_road.fit_transform(df['國道名稱'])

#進行One-Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
encoded_features = encoder.fit_transform(df[['方向', '事故類型']])

#編碼後的特徵欄位
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['方向', '事故類型']))

# 刪除編碼完成的欄位並與編碼後的特徵合併
df = df.drop(['方向', '事故類型'], axis=1)
df = pd.concat([df, encoded_df], axis=1)

#修正部分數值
df['翻覆事故註記'] = df['翻覆事故註記'].replace(' ', '0')
df = df.dropna()
df
```
# 模型訓練
最初我選擇了隨機森林回歸 (RandomForest Regressor) 進行模型訓練。然而通過舉辦方後續提供的測試集進行驗證後，發現隨機森林模型存在過擬合的跡象。為了減少過擬合問題，我採用了集成學習技術中的和投票法 (Voting)和堆疊法 (Stacking) ，透過組合多個模型的組合來提升模型的泛化能力。
---
將資料集分割成訓練集與測試集
```python
# 準備數據集
X = df.drop(columns=['處理分鐘','年','日'])  # 特徵
y = df['處理分鐘']  # 目標變量

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
隨機森林(RandomForest Regressor)
```python
rf_regressor = RandomForestRegressor()

# 訓練模型
rf_regressor.fit(X_train, y_train)
```
投票法 (Voting)
```python
# Voting Regressor
# 準備多個基模型
gb_regressor = GradientBoostingRegressor()
svr_regressor = SVR()

# 初始化投票回歸器
voting_regressor = VotingRegressor(estimators=[
    ('rf', rf_regressor),         # 使用已經訓練好的隨機森林模型
    ('gb', gb_regressor),
    ('svr', svr_regressor)
])

# 訓練投票回歸器
voting_regressor.fit(X_train, y_train)
```
堆疊法(Stacking)
```python
# Stacking Regressor
# 使用多個基模型，並使用線性回歸作為元學習器
stacking_regressor = StackingRegressor(
    estimators=[
        ('rf', rf_regressor),
        ('gb', gb_regressor),
        ('svr', svr_regressor)
    ],
    final_estimator=LinearRegression()
)

# 訓練Stacking回歸器
stacking_regressor.fit(X_train, y_train)
```
# 訓練結果比對
回歸模型用於預測連續數值，常見的評估指標之一是均方誤差 (Mean Squared Error, MSE)，MSE 能夠衡量模型預測結果的好壞。我計算了隨機森林、投票法與堆疊法三個模型的 MSE，並進行了結果比對。最終發現，投票法 (Voting) 的過擬合現象最為輕微，顯示其在此專案中的表現較為穩定，基於其優異的穩定性與較低的過擬合風險，最終決定將投票法作為後續應用的模型。
---

