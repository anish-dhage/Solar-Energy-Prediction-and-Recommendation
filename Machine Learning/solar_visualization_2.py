import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("83556_34.05_-118.26_2017.csv")
X = data.iloc[:, 0:14]  # independent columns
y = data.iloc[:, -1]    # target column i.e price range

# apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# concat two dataframes for better visualization

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
# plot heat map
print('heatmap')
g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
print("after heatmap")
plt.show()
'''
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) # use inbuilt class feature_importances of tree based classifiers
# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
'''