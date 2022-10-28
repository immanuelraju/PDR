import pandas  as pd
import numpy as np
data=pd.read_csv("Iris.csv")
data
data= data.drop('Id',axis=1)
data
data.describe()
data.info()
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
LE=LabelEncoder()
data.iloc[:,-1]=LE.fit_transform(data.iloc[:,-1])
data
x=data.iloc[:,:-1]
x.head()
y=data.iloc[:,-1]
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=50)
X_train.head()
X_train.shape
y_train.shape
from sklearn.tree import DecisionTreeClassifier
data=DecisionTreeClassifier()
data.fit(X,y)
y_pred=data.predict(X_test)
y_pred
y_test=np.array(y_test)
y_test
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,y_test)
from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))
from sklearn import tree
DecisionTreeClassifier()
dataviz=tree.plot_tree(data,feature_names=x.columns,filled=True,fontsize=20)
