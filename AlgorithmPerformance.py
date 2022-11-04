# Import libraries

import sys
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
# pandas
import pandas
# scikit-learn
import sklearn
# XGBoost Algorithm
import xgboost as xgb
# Time Module
import time
# Plots
import matplotlib.pyplot as plt


from pandas.plotting import scatter_matrix # Scatter matrix
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler # Standardize data (0 mean, 1 stdev)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Training Data
url = "Set9 - Training.csv"
dataTrain = pandas.read_csv(url)

dataTrain = dataTrain[['glucose','finger stick','bolus amount','carb amount', 'sleep','stressors','hypo','illness','exercise', 'heart rate', 'classify risk', 'risk score']]

X = dataTrain[['glucose','finger stick','bolus amount','carb amount', 'sleep','stressors','hypo','illness','exercise', 'heart rate']]
Y = dataTrain[["classify risk"]]

X.columns = X.columns.str.replace(' ', '_')
Y.columns = Y.columns.str.replace(' ', '_')

# Testing Data
url = "mydata3.csv"
dataTest = pandas.read_csv(url)

dataTest = dataTest[['glucose','finger stick','bolus amount','carb amount', 'sleep','stressors','hypo','illness','exercise', 'heart rate', 'classify risk', 'risk score']]

XTest = dataTest[['glucose','finger stick','bolus amount','carb amount', 'sleep','stressors','hypo','illness','exercise', 'heart rate']]
YTest = dataTest[["classify risk"]]

XTest.columns = XTest.columns.str.replace(' ', '_')
YTest.columns = YTest.columns.str.replace(' ', '_')

scaler = StandardScaler()
Y = numpy.ravel(Y)
YTest = numpy.ravel(YTest)
print(numpy.any(numpy.isnan(XTest)))

XTest = numpy.nan_to_num(XTest)

# KNN
from sklearn.neighbors import KNeighborsClassifier

start_time = time.time()

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X,Y)

y_pred = knn.predict(XTest)

end_time = time.time()
exec_time = end_time - start_time
print("Execution time KNN:  %2.4f" %(exec_time), "seconds")

#print(confusion_matrix(YTest, y_pred))
#print(classification_report(YTest, y_pred))
print("Accuracy KNN: %2.2f" %(metrics.accuracy_score(YTest, y_pred)*100))
print("Precision KNN: %2.2f" %(metrics.precision_score(YTest, y_pred, average = 'macro')*100))
print("Recall KNN: %2.2f" %(metrics.recall_score(YTest, y_pred, average = 'macro')*100))
print("F1 Score KNN: %2.2f" %(metrics.f1_score(YTest, y_pred, average = 'macro')*100))

# SVM
from sklearn import svm

start_time = time.time()

#Create a svm Classifier
svmClassifier = svm.SVC(kernel='linear') # Linear Kernel

# Train the model using the training sets
svmClassifier.fit(X, Y)

# Predict the response for test dataset
y_predSVM = svmClassifier.predict(XTest)

end_time = time.time()
exec_time = end_time - start_time
print("Execution time SVM:  %2.4f" %(exec_time), "seconds")

#print(confusion_matrix(YTest, y_predSVM))
#print(classification_report(YTest, y_predSVM))
print("Accuracy SVM: %2.2f" %(metrics.accuracy_score(YTest, y_predSVM)*100))
print("Precision SVM: %2.2f" %(metrics.precision_score(YTest, y_predSVM, average = 'macro')*100))
print("Recall SVM: %2.2f" %(metrics.recall_score(YTest, y_predSVM, average = 'macro')*100))
print("F1 Score SVM: %2.2f" %(metrics.f1_score(YTest, y_predSVM, average = 'macro')*100))

# Random Forest
from sklearn.ensemble import RandomForestClassifier, RandomForestClassifier

start_time = time.time()
regressor = RandomForestClassifier(n_estimators=10, random_state=0)
regressor.fit(X, Y)
y_predRF = regressor.predict(XTest)

end_time = time.time()
exec_time = end_time - start_time
print("Execution time RF:", exec_time, "seconds")

#print('Mean Absolute Error:', metrics.mean_absolute_error(YTest, y_predRF))
#print('Mean Squared Error:', metrics.mean_squared_error(YTest, y_predRF))
#print('Root Mean Squared Error:', numpy.sqrt(metrics.mean_squared_error(YTest, y_predRF))) 

#print(confusion_matrix(YTest, y_predRF))
#print(classification_report(YTest, y_predRF))
print("Accuracy RF: %2.2f" %(metrics.accuracy_score(YTest, y_predRF)*100))
print("Precision RF: %2.2f" %(metrics.precision_score(YTest, y_predRF, average = 'macro')*100))
print("Recall RF: %2.2f" %(metrics.recall_score(YTest, y_predRF, average = 'macro')*100))
print("F1 Score RF: %2.2f" %(metrics.f1_score(YTest, y_predRF, average = 'macro')*100))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

start_time = time.time()
tree = DecisionTreeClassifier(criterion="entropy", max_depth = 8).fit(X,Y)
y_predDT = tree.predict(XTest)

end_time = time.time()
exec_time = end_time - start_time
print("Execution time DT:  %2.4f" %(exec_time), "seconds")

#print(confusion_matrix(YTest, y_predDT))
print("Accuracy DT: %2.2f" %(metrics.accuracy_score(YTest, y_predDT)*100))
print("Precision DT: %2.2f" %(metrics.precision_score(YTest, y_predDT, average = 'macro')*100))
print("Recall DT: %2.2f" %(metrics.recall_score(YTest, y_predDT, average = 'macro')*100))
print("F1 Score DT: %2.2f" %(metrics.f1_score(YTest, y_predDT, average = 'macro')*100))

from sklearn.naive_bayes import GaussianNB

start_time = time.time()
gnb = GaussianNB()
y_predNB = gnb.fit(X, Y).predict(XTest)

end_time = time.time()
exec_time = end_time - start_time
print("Execution time NB:  %2.4f" %(exec_time), "seconds")

#print(confusion_matrix(YTest, y_predNB))
print("Accuracy NB: %2.2f" %(metrics.accuracy_score(YTest, y_predNB)*100))
print("Precision NB: %2.2f" %(metrics.precision_score(YTest, y_predNB, average = 'macro')*100))
print("Recall NB: %2.2f" %(metrics.recall_score(YTest, y_predNB, average = 'macro')*100))
print("F1 Score NB: %2.2f" %(metrics.f1_score(YTest, y_predNB, average = 'macro')*100))
#print(classification_report(YTest, y_predNB))

conf_matrix = confusion_matrix(YTest, y_predNB)

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# fit model no training data
start_time = time.time()
model = XGBClassifier()


for i in range(0,len(Y)):
   Y[i] = Y[i] + 1

for k in range(0,len(YTest)):
   YTest[k] = YTest[k] + 1

y_predXGB = model.fit(X, Y).predict(XTest)

end_time = time.time()
exec_time = end_time - start_time
print("Execution time NB:  %2.4f" %(exec_time), "seconds")
# evaluate predictions
print("Accuracy XGB: %2.2f" %(metrics.accuracy_score(YTest, y_predXGB)*100))
print("Precision XGB: %2.2f" %(metrics.precision_score(YTest, y_predXGB, average = 'macro')*100))
print("Recall XGB: %2.2f" %(metrics.recall_score(YTest, y_predXGB, average = 'macro')*100))
print("F1 Score XGB: %2.2f" %(metrics.f1_score(YTest, y_predXGB, average = 'macro')*100))


fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()



from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

from sklearn.tree import plot_tree

feature_cols = ['glucose','finger stick','bolus amount','carb amount', 'sleep','stressors','hypo','illness','exercise','heart rate']


plt.figure(figsize=(20, 20))
fig = plot_tree(tree, filled=True, feature_names= feature_cols, class_names=['-1','0','1'])
plt.show()


dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['-1','0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
