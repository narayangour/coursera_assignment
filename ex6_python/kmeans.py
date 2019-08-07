import scipy.io
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import io
import sys


# Loading dataset from disk using scipy 
mat61 = scipy.io.loadmat('ex6data1.mat')
X = mat61['X']

# Visualizing the Data
plt.scatter(X[:,0],X[:,1], label='True Position')  
plt.title('input Data point')
plt.show()
print "\nRunning KMeans classifier on %s dataset ...\n"%("ex6data1.mat")					
print "Creating Clusters with different number of iteration value ..."
max_iter = 10000			
while(max_iter):
	kmeans = KMeans(n_clusters=3,max_iter=5)  
	kmeans.fit(X)
	max_iter = max_iter - 10

# Final centroid values the algorithm generated after some iteration
print "\nFinal centroid values : "
print (kmeans.cluster_centers_)



# ploted the data points again on the graph and visualize how the data has been clustered
print "\nPloted the data points wrt clustered"
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow') 
plt.title('kmeans_labels') 
plt.show()


# ploted the points along with the centroid coordinates of each cluster to see how the centroid positions effects clustering
print "\nCentroid position "
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')  
plt.title('kmeans_cluster_centers') 
plt.show()
print "\nDone with Apply K-Means classifier on ex6data1.mat\n"



print('\nRunning K-Means clustering on pixels from an image.\n\n');
image = io.imread('bird_small.png')
io.imshow(image)
plt.title('original_bird_small_image') 
io.show()

rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(image.shape[0]*image.shape[1],3)

#  kmeans algorithms with with 16 colors and max iter 10
kmeans = KMeans(n_clusters = 128, n_init=10, max_iter=10)
kmeans.fit(image)

clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8) 
labels = np.asarray(kmeans.labels_,dtype=np.uint8 )  
labels = labels.reshape(rows,cols); 


# saving in standard binary file format 
np.save('codebook_tiger.npy',clusters)    
io.imsave('compressed_bird_small.png',labels);


print "\nReconstructing main features - "
# load saved numpy array of clusters and respective lebels
centers = np.load('codebook_tiger.npy')
c_image = io.imread('compressed_bird_small.png')


image = np.zeros((c_image.shape[0],c_image.shape[1],3),dtype=np.uint8 )
for i in range(c_image.shape[0]):
    for j in range(c_image.shape[1]):
            image[i,j,:] = centers[c_image[i,j],:]
    

print "\nReconstructing done..!!"

io.imsave('reconstructed_bird_small.png',image);
io.imshow(image)
plt.title('reconstructed_bird_small.png') 
io.show()
















