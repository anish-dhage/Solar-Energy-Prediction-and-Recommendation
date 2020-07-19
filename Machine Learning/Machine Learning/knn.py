import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

def KNN():
    df = pd.read_csv('2017.csv',usecols=['Month', 'Hour', 'Cloud Type', 'Dew Point', 'Solar Zenith Angle', 'Wind Speed', 'Precipitable Water', 'Relative Humidity', 'Temperature','GHI'])

    x = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values

    #x,y = shuffle(x,y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    knn2 = KNeighborsRegressor(n_neighbors=7,metric='manhattan')
    knn2.fit(X_scaled,y_train)
    y_pred = knn2.predict(X_test_scaled)
    y_pred = y_pred.astype(int)
    y_pred = abs(y_pred)
    print()
    rmse = mean_squared_error(y_test,y_pred)
    print("R2 SCORE :",r2_score(y_test,y_pred))
    #print(rmse)

    return "KNN",rmse

if __name__ == '__main__':
    a,b = KNN()
    print(np.sqrt(b))

