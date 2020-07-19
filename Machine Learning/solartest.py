import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# from sklearn.externals import joblib
import joblib
def quad():
    df = pd.read_csv('2017.csv')
    x = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = MinMaxScaler()
    X_train =scaler.fit_transform(X_train)
    X_test =scaler.fit_transform(X_test)
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(X_train)
    X_test_ = poly.fit_transform(X_test)
    lg = LinearRegression()
    lg.fit(X_, y_train)

    # joblib.dump(lg,'solarmodel.pkl')

    print(X_test)
    y_pred = lg.predict(X_test_)
    print(y_pred)
    rmse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    #print("R2 SCORE QUADRATIC REGRESSION",r2)
    #print("RMSE SCORE QUDRATIC REGRESSION",rmse)
    y_pred = y_pred.astype(int)
    y_pred = abs(y_pred)
    print(y_pred)
    return "Quadratic",rmse


def predict():
    df = pd.read_csv('2017.csv')
    x = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    poly = PolynomialFeatures(degree=2)
    X_ = poly.fit_transform(X_train)
    X_test_ = poly.fit_transform(X_test)


    model=joblib.load('solarmodel.pkl')

    # print("_----------?-----------------------------------------")
    y_pred=model.predict(X_test_)
    y_pred = y_pred.astype(int)
    y_pred = abs(y_pred)
    print(type(y_pred))

    print(y_pred[0])
    return y_pred[0]

    print('yo')

def predict2():
    scaler = MinMaxScaler()
    X_test=[2017,1,1,16,30,25,249,3,6,86.28,0.128,2,1.117,227.2,90.08,9,980]
    X_test=np.array(X_test)
    X_test=[X_test]
    poly = PolynomialFeatures(degree=2)
    X_test_ = poly.fit_transform(X_test)
    model=joblib.load('solarmodel.pkl')
    y_pred=model.predict(X_test_)
    y_pred = y_pred.astype(int)
    y_pred = abs(y_pred)
    return (y_pred[0])

if __name__ =='__main__':
    quad()