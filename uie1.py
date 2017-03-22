# Spyridon Angelopoulos 16-929-911


import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, decomposition, manifold
import pickle
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage import feature
from PIL import Image
from scipy import signal
from scipy import misc
from scipy import ndimage
from scipy.stats import multivariate_normal
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneOut, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint
from sklearn.model_selection import KFold, PredefinedSplit, ShuffleSplit
from sklearn.cluster import MiniBatchKMeans, KMeans
np.random.seed(1)
from numpy.linalg import norm
import numpy.linalg
from scipy.spatial import distance
import cv2



#---------------------------------------------------------6----------

def visualizeHOG(img, orientations=8, pixels_per_cell=(16,16), cells_per_block=(4,4), widthPadding=10, plotTitle=None):
    """
    Calculates HOG feature vector for the given image.
    
    img is a numpy array of 2- or 3-dimensional image (i.e., grayscale or rgb). 
    Color-images are first transformed to grayscale since HOG requires grayscale 
    images.
    
    Reference: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
    """
    if len(img.shape) > 2:
        im = Image.fromarray(img)
        im = im.convert('LA')
        img = np.asarray(im)
        img = img[:,:,0]
        
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]

    hog_features, hog_image = feature.hog(img, orientations, pixels_per_cell, cells_per_block, visualise=True)

    if plotTitle is not None:

        
        return hog_features
    

def bagOfWordshistograms(hogs, centers,N,K, kmeans):
    
    # Count the number appearances of each cluster inside every feature
    bows = np.zeros((N,K))
    
    for i in range(N):
        hog = hogs[i,:];
        hog=np.reshape(hog, (64,8))
        
        hog= hog[~np.all(hog == 0, axis=1)]
        
        tmp = np.zeros(K)

        for j in range(len(hog)):
            feat = hog[j]
            feat = np.reshape(feat,(1,-1))
            a = kmeans.predict(feat)
            tmp[a] = tmp[a] +1
            
        bows[i,:] = tmp          
       
    
    return bows


def DenseSiftHistograms(BOW,n,k,dictionary):
    
    
    # Count the number appearances of each cluster inside every descriptor
    BOW = np.reshape(BOW,(n,108*128)) 
    features = np.zeros((n,k))
    
    for i in range(n):
        
    
        descriptions = BOW[i,:];
        descriptions=np.reshape(descriptions, (108,128))
        #print(i)
        #print(descriptions.shape)
        tmp = np.zeros(k)
        
        for j in range(len(descriptions)):
            feat = descriptions[j]
            feat = np.reshape(feat,(1,-1))
            a = dictionary.predict(feat)
            tmp[a] = tmp[a] +1
    
        features[i,:] = tmp
                
    
    return features

    


def calculateHogs(n,data):

    #obtain hog features for a set of images
    hogs = np.zeros((n, 512))
    for index in range(n):
        
        maskedDepth = data['rgb'][index]
                          
        tmp = (visualizeHOG(maskedDepth, plotTitle='HOG-Masked RGB'));
        hogs[index,:]=tmp
    
            
       # if index%1000 ==0:
        #    print("+1000")
            
    return hogs

def RGBSiftExctract(n,datas):
    
    # Exctract sift descriptors for a set of rgb images
    BOW = np.zeros((n*108,128))
    counter = 0

    for i in range(n):

        #Apply Mask
        segmentedUser = datas['segmentation'][i]
        sourceImg = datas['rgb'][i]
        mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
        mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
        mask3 = mask3.transpose((1,2,0))
        img = sourceImg *mask3
        
        sift = cv2.xfeatures2d.SIFT_create()
        # Create grid of key points.
        keypointGrid = [cv2.KeyPoint(x, y, 10)
                        for y in range(0, img.shape[0], 10)
                            for x in range(0, img.shape[1], 10)]
        
        # Given the list of keypoints, compute the local descriptions for every keypoint.
        (kp, descriptions) = sift.compute(img, keypointGrid)
       # print(descriptions.shape)
       # count word apperances
        for row in range(108):
            BOW[counter]=descriptions[row,:]
            counter = counter+1
            
    
    return BOW

def DepthSiftExctract(n,datas):
    
    # Exctract sift descriptors for a set of depth images
    BOW = np.zeros((n*108,128))
    counter = 0

    for i in range(n):

        #Apply Mask
        segmentedUser = datas['segmentation'][i]
        sourceImg = datas['depth'][i]
        mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
        img = sourceImg *mask2
        
        sift = cv2.xfeatures2d.SIFT_create()
        # Create grid of key points.
        keypointGrid = [cv2.KeyPoint(x, y, 10)
                        for y in range(0, img.shape[0], 10)
                            for x in range(0, img.shape[1], 10)]
        
        # Given the list of keypoints, compute the local descriptions for every keypoint.
        (kp, descriptions) = sift.compute(img, keypointGrid)
       # print(descriptions.shape)
       # count word apperances
        for row in range(108):
            BOW[counter]=descriptions[row,:]
            counter = counter+1
            
    
    return BOW
            
     



def trainAndEvaluate(bows, data, n):
    
    # cv parameter of RandomizedSearchCV or GridSearchCV can be fed with a customized cross-validation object.
    ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
                      
    # Optimize the parameters by cross-validation.
    parameters = {
            "max_depth": sp_randint(20,30),
            "max_features": sp_randint(10,20),
            "min_samples_split": sp_randint(2, 6),
            "min_samples_leaf": sp_randint(2, 6),
            'n_estimators': [150,200,300],
        }
    
    # Random search object with SVM classifier.
    clf = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=1),
            param_distributions=parameters,
            n_iter=10,
            cv=10,
            random_state=1,
        )
    
    
    clf.fit(bows, data['gestureLabels'][0:n])
    
    print("Best parameters set found on training set:")
    print(clf.best_params_)
    print()
    
    
    means_valid = clf.cv_results_['mean_test_score']
    stds_valid = clf.cv_results_['std_test_score']
    means_train = clf.cv_results_['mean_train_score']
    
    print("Grid scores:")
    for mean_valid, std_valid, mean_train, params in zip(means_valid, stds_valid, means_train, clf.cv_results_['params']):
        print("Validation: %0.3f (+/-%0.03f), Training: %0.3f  for %r" % (mean_valid, std_valid, mean_train, params))
    print()

    
    return clf
    
    #labels_test, labels_predicted = labels_test, clf.predict(X_test)
    #print("Test Accuracy [%0.3f]" % ((labels_predicted == labels_test).mean()))
    

#-----------------------------------------------------------------
#Load Data

data = pickle.load(open('data/a1_dataTrain.pkl', 'rb'))
test = pickle.load(open('data/a1_dataTest.pkl', 'rb'))
segmentation_images = data['segmentation']

print("Data Loaded successfully!")


#-----------------------------------------------------------------

# Collect features HOG

n=1000;
k = 60;


#calculate histograms for all images
hogs = calculateHogs(n,data)

# reshape to obtain all hog words
words =  np.reshape(hogs, (n*64, 8))
print(words.shape)


# Cluster words into K clusters with kmeans
kmeans = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=50)
kmeans.fit(words)
print(len(kmeans.cluster_centers_))

#print(kmeans.cluster_centers_)

clustersHOG = kmeans.cluster_centers_

# Create bow histograms
bows = bagOfWordshistograms(hogs,clustersHOG,n,k, kmeans)
print(bows.shape)


print("Hog train features collected")

#-----------------------------------------------------------------

# Collect features DENSE SIFT RGB


dictionarySize = 50
BOWRGB = RGBSiftExctract(n,data)


#Apply Kmeans
dictionaryRGB = MiniBatchKMeans(init='k-means++', n_clusters=dictionarySize, batch_size=200)
dictionaryRGB.fit(BOWRGB)
print(len(dictionaryRGB.cluster_centers_))

clustersSiftRGB = dictionaryRGB.cluster_centers_

featuresRGB = DenseSiftHistograms(BOWRGB,n,dictionarySize,dictionaryRGB)

print("RGB dence sift train features collected")

#-----------------------------------------------------------------

# Collect features DENSE SIFT DEPTH


dictionarySize = 40
BOWDepth = DepthSiftExctract(n,data)


#Apply Kmeans
dictionaryDepth = MiniBatchKMeans(init='k-means++', n_clusters=dictionarySize, batch_size=200)
dictionaryDepth.fit(BOWDepth)
print(len(dictionaryDepth.cluster_centers_))

clustersSiftDepth = dictionaryDepth.cluster_centers_

featuresDepth = DenseSiftHistograms(BOWDepth,n,dictionarySize,dictionaryDepth)

print("Depth dence sift train features collected")

#------------------------------------------------------------------------

# Process test data HOG


ntest=200;
k = 60;


#calculate histograms for all images
hogs = calculateHogs(ntest,test)


# Create bow histograms
testhog = bagOfWordshistograms(hogs,clustersHOG,ntest,k, kmeans)
print(bows.shape)

print("Hog test features collected")

#-----------------------------------------------------------------

# Process features DENSE SIFT RGB


dictionarySize = 50
BOWRGB = RGBSiftExctract(ntest,test)


testRGB = DenseSiftHistograms(BOWRGB,ntest,dictionarySize,dictionaryRGB)

print("Rgb dence sift test features collected")

#-----------------------------------------------------------------

# Process features DENSE SIFT DEPTH


dictionarySize = 40
BOWDepth = DepthSiftExctract(ntest,test)


testDepth = DenseSiftHistograms(BOWDepth,ntest,dictionarySize,dictionaryDepth)

print("Depth dence sift test features collected")

#------------------------------------------------------------------------

# Train the classifier 


trainfeat = np.concatenate((bows,featuresRGB),axis=1)
trainfeat = np.concatenate((trainfeat,featuresDepth),axis=1)

testfeat = np.concatenate((testhog,testRGB),axis=1)
testfeat = np.concatenate((testfeat,testDepth),axis=1)

print("Training classifier....")

classifier = trainAndEvaluate(trainfeat, data, n)
print(classifier.best_params_)

# Test the classifier

print("Predicting test labels...")

labels_predicted = classifier.predict(testfeat)

np.save('labelstest.npy', labels_predicted)



#-------------------------------------------------------------------------
