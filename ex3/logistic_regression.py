from sklearn.model_selection import KFold 		# import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  	# For Standardization
from sklearn import preprocessing
import csv
import numpy


# Loading dataset from disk using numpy 
filename = 'pima-indians-diabetes.csv'
X = numpy.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1,usecols = (0,1,2,3,4,5,6,7))
y = numpy.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1,usecols = (8))


kf = KFold(n_splits=10) 				# Define the split - into n_splits folds 
kf.get_n_splits(X) 					# returns the number of splitting iterations in the cross-validator

count = 0						# k-fold count
sm = 0		
print "\nRunning Logistic Regression classifier on %s dataset ...\n"%(filename)					
print "|---------------------------------------------------|"
print "| Fold number | Total Accuracy" 

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
 	y_train, y_test = y[train_index], y[test_index]

	# feature scalling....mostly use to get same accuracy at less k value
	scaler = StandardScaler()  
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test) 

	# linear model LogisticRegression
	clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
	count = count + 1
	score = clf.score(X_test, y_test) 
	print "|     ",count,"      |",score 
	sm = sm + score
print "|--------------------------------------------------------------------|"
print "| Avg Classification accuracy After",count ,"Fold is -",sm/count," |" 
print "|--------------------------------------------------------------------|"

