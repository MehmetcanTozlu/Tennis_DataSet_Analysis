import numpy as np
import pandas as pd

df = pd.read_csv('tennis.csv')
"""
# Categoric olan outlook Column'unu Numeric Yapalim
outlook = df.iloc[:, 0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
outlook[:, 0] = le.fit_transform(outlook)
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
outlookFrame = pd.DataFrame(data=outlook, index=range(14), columns=['Overcast', 'Rainy', 'Sunny'])

# Boolean olan windy Column'unu Numeric Yapalim
windy = df.iloc[:, 3:4].values
def convert(i):
    if i == True:
        return 1
    else:
        return 0
windy = list(map(convert, windy))
windy = np.array(windy)
windyFrame = pd.DataFrame(data=windy, index=range(14),  dtype=(int), columns=['Windy'])

# Categoric olan play Column'unu Numeric Yapalim
play = df.iloc[:, 4:5].values
play = le.fit_transform(df.iloc[:, 4:5].values)
playFrame = pd.DataFrame(data=play, index=range(14), columns=['Play']) # Dependent Variable

# Frame'lerimizi birlestirelim ve veri setimizi son haline guncelleyelim
tempFrame = pd.concat([outlookFrame, df.iloc[:, 1:3]], axis=1)
tempFrame2 = pd.concat([tempFrame, windyFrame], axis=1) # Independent Variables
resultDataFrame = pd.concat([tempFrame2, playFrame], axis=1)
"""
# Ustteki yorum satirini daha kisa sekilde yapalim;
from sklearn import preprocessing
tempFrame = df.apply(preprocessing.LabelEncoder().fit_transform)

outlook = tempFrame.iloc[:, 0:1].values
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
outlookFrame = pd.DataFrame(data=outlook, index=range(14), columns=['Overcast', 'Rainy', 'Sunny'])

numericValues = df.iloc[:, 1:3].values
numericValuesFrame = pd.DataFrame(data=numericValues, index=range(14), columns=['Temperature', 'Humidity'])

windy_play = tempFrame.iloc[:, 3:5].values
windy_playFrame = pd.DataFrame(data=windy_play, index=range(14), columns=['Windy', 'Play'])

# Frame'leri concat edelim
frame1 = pd.concat([outlookFrame, numericValuesFrame], axis=1)
dataFrame = pd.concat([frame1, windy_playFrame], axis=1) # Data Frame'imizin son hali

dependentVariables = dataFrame.iloc[:, 4:5]
independentVariables = dataFrame.drop('Humidity', axis=1)

# resultDataFrame'imizi train ve test olarak split edelim
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(independentVariables, dependentVariables, test_size=0.3, random_state=0)

# Modelimizi Olusturalim - Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Modelimizi Bacward Elimination ile p-value'lara bakalim
import statsmodels.api as sm
# y = B0 + B1*X1 + B2*X2 + B3*X3 + ..... + Bn*Xn + E formulundeki B0 degiskenini alttaki satirla veri setimizde olusturuyoruz
X = np.append(arr=np.ones((14,1)).astype(int), values=independentVariables, axis=1)
X_list = independentVariables.iloc[:, [0,1,2,3,4,5]].values
X_list = np.array(X_list, dtype=float)
model = sm.OLS(dependentVariables, X_list).fit()
print(model.summary())

# 0.593 ile Windy Column'unu cikaralim
X_list2 = independentVariables.iloc[:, [0,1,2,3,5]].values # 4. indexte Windy vardi
X_list2 = np.array(X_list2, dtype=float)
model2 = sm.OLS(dependentVariables, X_list2).fit()
print(model2.summary())

# Windy degerlerinin p-value yuksek old. icin x'in egitim ve test verilerinden cikartip tekrar predict islemini gerceklestirelim
# ve sonuca nasil etki edecek gorelim
x_train = x_train.drop('Windy', axis=1)
x_test = x_test.drop('Windy', axis=1)

regressor.fit(x_train, y_train)

y_pred2 = regressor.predict(x_test)
