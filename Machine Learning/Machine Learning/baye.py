import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB


def baye():
    df = pd.read_csv('2017.csv')
    x = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    model = GaussianNB()
    model.fit(X_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred = y_pred.astype(int)
    y_pred = abs(y_pred)
    r2 = r2_score(y_test,y_pred)
    rmse =mean_squared_error(y_test,y_pred)
    #print("RMSE OF BAYES THEORMN",rmse)
    #print("R2 SCORE OF BAYES THEORM",r2)

    return "BAYES",rmse



if __name__ =='__main__':
    a,b = baye()