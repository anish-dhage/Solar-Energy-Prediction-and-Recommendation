import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import MinMaxScaler

def liner():
    df = pd.read_csv('2017.csv')
    x = df.iloc[:,0:-1].values
    y = df.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = MinMaxScaler()
    X_train =scaler.fit_transform(X_train)
    X_test =scaler.fit_transform(X_test)

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    y_predicted = regression_model.predict(X_test)
    rmse = mean_squared_error(y_test,y_predicted)
    r2 = r2_score(y_test,y_predicted)
    #print("R2 SCORE LINEAR REGRESSION",r2)
    #print("RMSE SCORE LINEAR REGRESSION",rmse)
    y_predicted = y_predicted.astype(int)
    y_predicted = abs(y_predicted)

    return "linear",rmse

if __name__ =='__main__':
    a,b=liner()