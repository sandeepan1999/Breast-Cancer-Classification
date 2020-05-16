#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(cancer['feature_names'],['target']))

#visualising the data
sns.pairplot(df_cancer,hue='target', vars=['mean radius','mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(df_cancer['target'],label='Count')

sns.scatterplot(x='mean area',y='mean smoothness', hue='target',data=df_cancer)

sns.lmplot('mean area', 'mean smoothness', hue ='target', data = df_cancer, fit_reg=False)

#checking correlation
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot=True)

#training the model
X=df_cancer.drop(['target'],axis=1)
y=df_cancer['target']

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=5)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)

#evaluating the model
y_pred = classifier.predict(X_test)
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)

print(classification_report(y_test, y_pred))

#improving the model
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train-min_train)/range_train
sns.scatterplot(data=X_train_scaled,x='mean area',y='mean smoothness',hue=y_train)

min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test_scaled = (X_test-min_test)/range_test

classifier = SVC()
classifier.fit(X_train_scaled,y_train)

y_pred = classifier.predict(X_test_scaled) 
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_test,y_pred))

#improving the model #2
param_grid = {'C':[0.1,1,10,100],'gamma':[0.1,1,10,100],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

grid.best_params_
grid.best_estimator_
grid_predictions= grid.predict(X_test_scaled)
cm = confusion_matrix(y_test,grid_predictions)
sns.heatmap(cm,annot=True)

print(classification_report(y_test,grid_predictions))




