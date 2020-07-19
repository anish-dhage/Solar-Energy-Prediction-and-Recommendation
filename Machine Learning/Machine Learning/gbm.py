import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def GBM():
    df = pd.read_csv('2017.csv')

    x = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    params = {
        'n_estimators': 3500,
        'max_depth': 1,
        'learning_rate': 1,
    }


    gbm = GradientBoostingRegressor(**params)

    gbm.fit(X_scaled,y_train)

    y_pred = gbm.predict(X_test_scaled)
    y_pred = y_pred.astype(int)
    y_pred = abs(y_pred)
    r2 = r2_score(y_test,y_pred)
    rmse =mean_squared_error(y_test,y_pred)
    #print("RMSE OF gbm THEORMN",rmse)
    #print("R2 SCORE OF gbm THEORM",r2)
    return "GBM",rmse

if __name__ =='__main__':
    a,b= GBM()