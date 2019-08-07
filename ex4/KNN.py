from sklearn.model_selection import KFold 		# import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt				# For plotting graph of data
import operator
import numpy
import csv

# Loading dataset from disk using numpy 
filename = 'winequality-red.csv'
X = numpy.loadtxt(open(filename, "rb"), delimiter=";", skiprows=1,usecols = (0,1,2,3,4,5,6,7,8,9,10))
y = numpy.loadtxt(open(filename, "rb"), delimiter=";", skiprows=1,usecols = (11))

neighbor_range = 14			# Set Manually neighbors range for knowing best fit value of K for this dataset
K_fold = 10				#K fold

kf = KFold(n_splits=K_fold) 		# Define the split - into K_fold folds 
kf.get_n_splits(X) 			# returns the number of splitting iterations in the cross-validator
count = 0

print "\nRunning KNN classifier on %s dataset ...\n"%(filename)					
print "K-fold  and k value for KNN is %d %d "%(K_fold,neighbor_range)
print "\n|---------------------------------------------------|"
print "|   Fold number       | Total Accuracy" 
print "|---------------------------------------------------|"
prev_val = 0
prev_index = 0
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
 	y_train, y_test = y[train_index], y[test_index]

	# feature scalling....mostly use to get same accuracy at less k value
	scaler = StandardScaler()  
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test) 

	#Setup arrays to store training and test accuracies
	neighbors = numpy.arange(1,neighbor_range)
	count = count + 1
	for i,k in enumerate(neighbors):
	    #Setup a knn classifier with k neighbors
	    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
	    
	    test_accuracy = knn.score(X_test, y_test)*100
	    if(test_accuracy>prev_val):
		prev_val = test_accuracy
		prev_index = k+1				# Plus one because loop is zero based

	# Un-comment if Generate plot

	#plt.title('k-NN Varying number of neighbors')
	#plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
	#plt.plot(neighbors, train_accuracy, label='Training accuracy')
	#plt.legend()
	#plt.xlabel('Number of neighbors')
	#plt.ylabel('Accuracy')
	#plt.show()
	print "| At %d Fold Accuracy is - %d"%(count,prev_val)
print "|-------------------------------------------------------------|"
print "| Maximum Accuracy can be achieved at k value",prev_index,"\n| So you can try to test by setting this latest neighbor value"
print "|-------------------------------------------------------------|"
















