from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/omkar.LAPTOP-S9C2IDIL/PycharmProjects/streamlit/Iris.csv")
print(data)
encode_species = LabelEncoder()
data.iloc[:, 5] = encode_species.fit_transform(data.iloc[:, 5])
print(data)

iris = datasets.load_iris()

X = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=72)
lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_model = SVC()
dtree_model = DecisionTreeClassifier()
lin_reg = lin_reg.fit(x_train, y_train)
log_reg = log_reg.fit(x_train, y_train)
svc_model = svc_model.fit(x_train, y_train)
dtree_model = dtree_model.fit(x_train,y_train)
print(lin_reg.score(x_test,y_test))
print(log_reg.score(x_test,y_test))
print(svc_model.score(x_test,y_test))
print(dtree_model.score(x_test,y_test))

pickle.dump(lin_reg, open('lin_model.pkl', 'wb'))
pickle.dump(log_reg, open('log_model.pkl', 'wb'))
pickle.dump(svc_model, open('svc_model.pkl', 'wb'))
pickle.dump(dtree_model, open('dtree_model.pkl', 'wb'))