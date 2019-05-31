# example train notebook using sklearn iris dataset
from sklearn import svm
from sklearn import datasets
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
from azureml.core.run import Run

run = Run.get_context()

iris = datasets.load_iris()
X, y = iris['data'], iris['target']

# for simplicity's sake, just using the test set as the val set. not best practice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)  

train_acc = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=2)  
run.log('train accuracy', train_acc)

y_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
run.log('test accuracy', test_acc)

with open('outputs/model.pkl', 'wb') as f:
    pickle.dump(clf, f)