from scipy import ndimage				# read image in numpy array
from scipy import misc					# resize image
from matplotlib import pyplot as plt			# show image
import os						# Read data from disk
import sys
from time import time
import numpy as np					# Numpy array
from sklearn.decomposition import PCA			# PCA sklearn lib
from sklearn.model_selection import train_test_split	# Split arrays or matrices into random train and test subsets
from sklearn.preprocessing import StandardScaler	# Standardize the Data
from sklearn.linear_model import LogisticRegression	# Apply Logistic Regression
from sklearn.model_selection import KFold 		# import KFold
from sklearn.metrics import accuracy_score		# calculate accuracy
from sklearn.svm import SVC  				# kernel for SVM
import logging						
from sklearn.model_selection import GridSearchCV	
from sklearn.metrics import classification_report		
from sklearn.metrics import confusion_matrix		
from sklearn.neighbors import KNeighborsClassifier


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=3):
    #Helper function to plot a gallery of portraits
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    if((n_row+n_col)>(len(images))):
	print "Please provide enough Test data..."
	sys.exit()
    for i in range(n_row * n_col):
	if(i>=len(images)):
		break
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
	plt.suptitle("Accuracy : "+ str(Accuracy_score)+"%") 

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

# create file list with there path levels 
def createFileListsWithPathLabels(RootFolder, PrefixFolderPath = False):
    if len(RootFolder.strip()) == 0 or not os.path.exists(RootFolder):
           raise ValueError('Error: createFilePathLists(): RootFolder path '\
                            'empty or invalid - "' + RootFolder + '"')
    # Create and fill the FolderLists
    FolderLists = []
    for rootdir,subdirs,files in os.walk(RootFolder):
        subdirs.sort() # walk in sorted order. in place sort
        if(len(files) > 0):
            folderFiles = []
            for fname in files:
                dirpath = os.path.relpath(rootdir,RootFolder) if \
                    not PrefixFolderPath else rootdir
                FilePath = os.path.join(dirpath,fname)
                csvList = [FilePath,dirpath]
                folderFiles.append(csvList)
            FolderLists.append(folderFiles)		# create subfolder + filename list
    return FolderLists

dpath = "/home/softnautics/narayan/exercises/ex7/dataset"
dataset = createFileListsWithPathLabels(dpath)
d = {}
img_count = 0
for category_name in dataset:
    for filename in category_name:
	level, name = filename[0].split("/")
    d[level] = img_count
    img_count = img_count + 1

#plt.figure()
images = []									# Total list from dataset	
levels = []
target_names = []								# levels category via
for category_name in dataset:
    for filename in category_name:
	image_path = dpath+"/"+filename[0]
	img = ndimage.imread(image_path)
	re_img = misc.imresize(img, (128, 128), interp='bilinear', mode=None)
	images.append(re_img)
	levels.append(d[filename[1]])
        #plt.imshow(re_img, cmap=plt.cm.gray)
	level, name = filename[0].split("/")
        #plt.title("Level-"+level+"\nName - "+name)
        #plt.xticks(())
        #plt.yticks(())
	#plt.show()
    target_names.append(filename[1])	

X = np.asarray(images)
y= np.asarray(levels)
target_names= np.asarray(target_names)
n_classes = target_names.shape[0]
n_samples, h, w = X.shape
X = np.reshape(X, (X.shape[0],X.shape[1]*X.shape[2]), order='C')
n_features = X.shape[1]
n_splits = 18

print "\nPCA applying on YALE face database ..."
print("\nTotal current dataset status :")
print"number of samples : %d \nnumber of features: %d \nnumber of classes : %d \nK-Fold\t\t  : %d"%(n_samples,n_features,n_classes,n_splits)


svclassifier = SVC(kernel='linear')  
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X) 
count = 0
sum_acc_svn = 0
print "\n1st :- Running SVM classifier... "					
print "\nFold Number | Training | Testing Images | Accuracy_score(%)"
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
 	y_train, y_test = y[train_index], y[test_index]
	# #############################################################################	
	# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
	# dataset): unsupervised feature extraction / dimensionality reduction
	n_components = 90
	#print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))
	t0 = time()
	pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
	#print("done in %0.3fs" % (time() - t0))
	eigenfaces = pca.components_.reshape((n_components, h, w))
	#print("Projecting the input data on the eigenfaces orthonormal basis")
	t0 = time()
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#print("done in %0.3fs" % (time() - t0))
	# Second approach
	svclassifier.fit(X_train, y_train)  
	y_pred = svclassifier.predict(X_test)  
	Accuracy_score = accuracy_score(y_test,y_pred)*100
	count = count + 1
	print "\t",count,"\t",len(X_train),"\t\t",len(X_test),"\t\t",Accuracy_score
	sum_acc_svn = sum_acc_svn + Accuracy_score
	if(count==n_splits):
		prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
		plot_gallery(X_test, prediction_titles, h, w)
		eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
		plot_gallery(eigenfaces, eigenface_titles, h, w)
		plt.suptitle('All eigenfaces') 
		plt.show()
print "|------------------------------------------------------------------|"
print "|  Final Average accuracy with SVM classifier - ",(sum_acc_svn/count)
print "|-------------------------------------------------------------------|"


clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X) 
count = 0
sum_acc_log = 0
print "\n2nd :- Running Logistic Regression classifier... "					
print "\nFold Number | Training | Testing Images | Accuracy_score"
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
 	y_train, y_test = y[train_index], y[test_index]
	# #############################################################################	
	# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
	# dataset): unsupervised feature extraction / dimensionality reduction
	n_components = 90
	#print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))
	t0 = time()
	pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
	#print("done in %0.3fs" % (time() - t0))
	eigenfaces = pca.components_.reshape((n_components, h, w))
	#print("Projecting the input data on the eigenfaces orthonormal basis")
	t0 = time()
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#print("done in %0.3fs" % (time() - t0))
	# Second approach
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)  
	Accuracy_score = clf.score(X_test, y_test)*100
	count = count + 1
	print "\t",count,"\t",len(X_train),"\t\t",len(X_test),"\t\t",Accuracy_score
	sum_acc_log = sum_acc_log + Accuracy_score
	if(count==n_splits):
		prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
		plot_gallery(X_test, prediction_titles, h, w)
		eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
		plot_gallery(eigenfaces, eigenface_titles, h, w)
		plt.suptitle('All eigenfaces') 
		plt.show()
print "|-----------------------------------------------------------------------------|"
print "|  Final Average accuracy with LogisticRegression classifier-",(sum_acc_log/count)
print "|------------------------------------------------------------------------------|"

k = 1
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X) 
count = 0
sum_acc_knn = 0
print "\n3rd :- Running KNN classifier... "					
print "\nFold Number | Training | Testing Images | Accuracy_score"
for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
 	y_train, y_test = y[train_index], y[test_index]
	# #############################################################################	
	# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
	# dataset): unsupervised feature extraction / dimensionality reduction
	n_components = 90
	#print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))
	t0 = time()
	pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
	#print("done in %0.3fs" % (time() - t0))
	eigenfaces = pca.components_.reshape((n_components, h, w))
	#print("Projecting the input data on the eigenfaces orthonormal basis")
	t0 = time()
	X_train_pca = pca.transform(X_train)
	X_test_pca = pca.transform(X_test)
	#print("done in %0.3fs" % (time() - t0))
	count = count + 1
        #Setup a knn classifier with k neighbors
	knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
	Accuracy_score = knn.score(X_test, y_test)*100
	y_pred = knn.predict(X_test) 
	print "\t",count,"\t",len(X_train),"\t\t",len(X_test),"\t\t",Accuracy_score
	sum_acc_knn = sum_acc_knn + Accuracy_score
	if(count==n_splits):
		prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
		plot_gallery(X_test, prediction_titles, h, w)
		eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
		plot_gallery(eigenfaces, eigenface_titles, h, w)
		plt.suptitle('All eigenfaces') 
		plt.show()
print "|-----------------------------------------------------------------------------|"
print "|  Final Average accuracy with KNN classifier-",(sum_acc_knn/count)
print "|------------------------------------------------------------------------------|"

num1 = sum_acc_svn/n_splits
num2 = sum_acc_log/n_splits
num3 = sum_acc_knn/n_splits
print "\n\nFinel result :\nSVM 		   - %d\nLogisticRegression - %d\nKNN 		   - %d"%(num1,num2,num3)

if (num1 >= num2) and (num1 >= num3):
    print "\nSVM have lagest accuracy amongs other classifier with this dataset\n\n"
elif (num2 >= num1) and (num2 >= num3):
    print "\nLogisticRegression have lagest accuracy amongs other classifier with this dataset\n\n"
else:
    print "\nKNN have lagest accuracy amongs other classifier with this dataset\n\n"


































