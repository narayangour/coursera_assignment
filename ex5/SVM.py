import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold 		# import KFold
import sys


# Loading dataset from disk using scipy 
mat51 = scipy.io.loadmat('ex5data1.mat')
mat62 = scipy.io.loadmat('ex6data2.mat')

K_fold = 10						#K fold
print "\nRunning SVM classifier on %s dataset ...\n"%("ex5data1.mat")					
print "\nK-fold for cross validate is     - ",K_fold
kf = KFold(n_splits=K_fold) 				# Define the split - into K_fold folds 
kf.get_n_splits(mat51['X']) 				# returns the number of splitting iterations in the cross-validator
svclassifier = SVC(kernel='linear')  
count = 0
print "\n|---------------------------------------------------|"
print "|   Fold number       | Total Accuracy" 
print "|---------------------------------------------------|"
prev_val = 0
prev_index = 0
sum_acc = 0
for train_index, test_index in kf.split(mat51['X']):
	X_train, X_test = mat51['X'][train_index], mat51['X'][test_index]
 	y_train, y_test = mat51['y'][train_index], mat51['y'][test_index]
	
	# Coverting supported dimention
	y_train = np.ravel(y_train)
	y_test = np.ravel(y_test)
	
	# Training 
	svclassifier.fit(X_train, y_train)

	# Predict 
	y_pred = svclassifier.predict(X_test)  

	count = count + 1
	score = accuracy_score(y_test, y_pred)
	print "|\t",count,"\t\t|  ",score*100,"%"
	sum_acc = sum_acc + score

print "|---------------------------------------------------|"
print "|  final -",(sum_acc/count)*100
print "|---------------------------------------------------|"







