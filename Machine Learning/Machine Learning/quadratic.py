import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.utils import shuffle


def quad():
    df = pd.read_csv('2017.csv',usecols=['Month', 'Hour','Minute', 'Dew Point', 'Solar Zenith Angle', 'Wind Speed', 'Precipitable Water', 'Relative Humidity', 'Temperature', 'GHI'])
    df['Hour'] = df['Hour'] + df['Minute']/60
    df = df.drop('Minute',1)
    print(df['Hour'])
    x = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values
    x,y = shuffle(x,y)
   # print(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = MinMaxScaler()
    X_train =scaler.fit_transform(X_train)
    X_test =scaler.fit_transform(X_test)
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(X_train)
    X_test_ = poly.fit_transform(X_test)
    lg = LinearRegression()
    lg.fit(X_, y_train)
    y_pred = lg.predict(X_test_)
    rmse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    print("R2 SCORE QUADRATIC REGRESSION",r2)
    #print("RMSE SCORE QUDRATIC REGRESSION",rmse)
    y_pred = y_pred.astype(int)
    y_pred = abs(y_pred)
    print(y_pred[155], y_test[155])
    return "Quadratic",rmse


if __name__ =='__main__':
    a,b = quad()
    print(np.sqrt(b))