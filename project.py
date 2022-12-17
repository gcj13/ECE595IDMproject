import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)



white = pd.read_csv("winequality-white.csv", sep = ';')
red = pd.read_csv("winequality-red.csv", sep = ';')

data = pd.concat([white, red], ignore_index=True, sort=False)
# print(data.head())
# print(data.info())
# print(data.describe(include='all'))

# f, ax = plt.subplots(figsize = (10,10))
# sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = ".2f", ax=ax)
# plt.show()
data = data.astype(float)

X = data.iloc[:,:11].values
Y = data.iloc[:,-1].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
regressors = [["SVC rbf C=0.1", SVC(kernel = 'rbf', C=0.1, gamma='scale', max_iter = 1500)],
              ["SVC rbf C=0.5", SVC(kernel = 'rbf', C=0.5, gamma='scale', max_iter = 1500)],
              ["SVC rbf C=1", SVC(kernel = 'rbf', C=1, gamma='scale', max_iter = 1500)],
              ["SVC rbf C=10", SVC(kernel = 'rbf', C=10.0, gamma='scale', max_iter = 1500)],
              ["SVC rbf C=20", SVC(kernel = 'rbf', C=20.0, gamma='scale', max_iter = 1500)],
              ["XGBclassifier", XGBClassifier()],
              ["Naive bayes Gaussian", GaussianNB()],
              # ["Naive bayes Multinomial", MultinomialNB()],
              ["K Nearest Neighbour", KNeighborsClassifier()],
              ["Decision Tree Classifier", DecisionTreeClassifier()],
              ["AdaBoostclassifier tree", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15))],
              ["K-Neighbors Classifier ", KNeighborsClassifier(n_neighbors=1)],
              ["Logistic Regression",LogisticRegression()]]
result1 = []
result2 = []
for reg in regressors:

     try:
           reg[1].fit(X_train,Y_train)
           train_score = cross_val_score(reg[1], X_train, Y_train, cv=5)
           scores = cross_val_score(reg[1], X_test, Y_test, cv=5)
           scores = np.average(scores)
           print(reg[0]+' cross val score', scores)
           if scores>= 0.55:
               cm = confusion_matrix(Y_test, reg[1].predict(X_test))
               print(cm)
           result1.append(reg[0])
           result2.append(scores)

     except:
          continue
# result1, result2 = zip(*sorted(zip(result1, result2), key=lambda x: x[1]))
# print("\n".join("{} {}".format(x, y) for x, y in zip(result1, result2)))
