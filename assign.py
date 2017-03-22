# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:27:18 2017
@author: Spyros
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, decomposition, manifold
import pickle
import cv2
#matplotlib inline
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
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        
    if widthPadding > 0:
        img = img[:, widthPadding:-widthPadding]

    hog_features, hog_image = feature.hog(img, orientations, pixels_per_cell, cells_per_block, visualise=True)

    if plotTitle is not None:
        #plt.figure()
        #plt.suptitle(plotTitle)
        #plt.subplot(1, 2, 1)
        #plt.imshow(img, cmap='gray')
        #plt.axis("off")
        #plt.subplot(1, 2, 2)
        #plt.imshow(hog_image)
        #plt.axis("off")
        
        return hog_features
    

def bagOfWordshistograms(hogs, centers,N,K):
    
    bows = np.zeros((N,K))
    
    for i in range(N):
        hog = hogs[i,:];
        hog=np.reshape(hog, (64,8))
        
        hog= hog[~np.all(hog == 0, axis=1)]
        
        tmp = np.zeros(K)
        
        distances= np.ones((len(hog),K))
        for j in range(len(hog)):
            for k in range(K):
                
                distances[j,k] = distance.euclidean(hog[j], centers[k])
                #print(distances[j,k] )
                #print(k)
            disttmp = distances[j,:].tolist()
            minimum = disttmp.index(min(disttmp))
            tmp[minimum] = tmp[minimum] +1
            
            #print("-----------------------------")
        #print("Image")
        #print(i)
        #print(tmp)
        bows[i,:] = tmp
    #print("+++++++++++++++++++++++++++++++++++")             
       
    
    return bows

def calculateHogs2(init,n,data,mask):

    
    hogs = np.zeros((n-init, 512))
    for index in range(init,n):
        
        
        user = data['segmentation'][index]
        maska2 = np.mean(user, axis=2) > 150 # For depth images.
        maska3 = np.tile(maska2, (3,1,1)) # For 3-channel images (rgb)
        maska3 = maska3.transpose((1,2,0))
        maskedDepth = data['rgb'][index] * maska3
                          
        tmp = (visualizeHOG(maskedDepth, plotTitle='HOG-Masked RGB'));
        hogs[index-10000,:]=tmp
    
            
        if index%1000 ==0:
            print("+1000")
            
    return hogs


def calculateHogs(n,data,mask):

    dictionarySize = 30

    hogs = cv2.BOWKMeansTrainer(dictionarySize)
    for index in range(n):
        
        user = data['segmentation'][index]
        maska2 = np.mean(user, axis=2) > 150 # For depth images.
        maska3 = np.tile(maska2, (3,1,1)) # For 3-channel images (rgb)
        maska3 = maska3.transpose((1,2,0))
        maskedDepth = data['rgb'][index] * maska3
                          
        tmp = (visualizeHOG(maskedDepth, plotTitle='HOG-Masked RGB'));
        hogs.add(tmp)
            
        if index%1000 ==0:
            print("+1000")
     
    
    print(hogs.shape)   
    dictionary = hogs.cluster()
    print(dictionary.shape)
    
    return hogs


def trainAndEvaluate(bows, data, n):
    
    # cv parameter of RandomizedSearchCV or GridSearchCV can be fed with a customized cross-validation object.
    ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
                      
    # Optimize the parameters by cross-validation.
    parameters = {
            "max_depth": sp_randint(15,55),
            "max_features": sp_randint(10,30),
            "min_samples_split": sp_randint(2, 10),
            "min_samples_leaf": sp_randint(2, 10),
            'n_estimators': [40,70,100,150,200,300,500],
        }
    
    # Random search object with SVM classifier.
    clf = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=1),
            param_distributions=parameters,
            n_iter=20,
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


data = pickle.load(open('data/a1_dataTrain.pkl', 'rb'))
#data = pickle.load(open('data/a1_dataTest.pkl', 'rb'))
segmentation_images = data['segmentation']



#-----------------------------------------------------------------


ind = 0
for image in segmentation_images:
    #if ind<10:
        #img = Image.fromarray(image, 'RGB')
        #str = 'my_image_%d.png' % ind
        #img.save(str)
        #img.show()
        ind += 1
        
print(len(segmentation_images))

sampleIdx = 5

# Fetch the segmentation mask of the reference image.
segmentedUser = data['segmentation'][sampleIdx]
mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
mask3 = mask3.transpose((1,2,0))

#tmp = data['gestureLabels']
#print (tmp[1:100])


#-----------------------------------------------------------------

# Collect features

n=1000;


dictionarySize = 50
BOW = np.zeros((n*84,128))
counter = 0

for i in range(n):
    
    segmentedUser = data['segmentation'][i]
    sourceImg = data['rgb'][i]
    mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
    mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
    mask3 = mask3.transpose((1,2,0))
    img = sourceImg *mask3
    img = img[:, 10:-10]
    sift = cv2.xfeatures2d.SIFT_create()
    # Create grid of key points.
    keypointGrid = [cv2.KeyPoint(x, y, 10)
                    for y in range(0, img.shape[0], 10)
                        for x in range(0, img.shape[1], 10)]
    
    # Given the list of keypoints, compute the local descriptions for every keypoint.
    (kp, descriptions) = sift.compute(img, keypointGrid)
   # print(descriptions)
   # print(counter) 
    for row in range(84):
        BOW[counter]=descriptions[row,:]
        counter = counter+1

np.save('testrgb.npy',BOW)
print("saved")

#dictionary created
dictionary = MiniBatchKMeans(init='k-means++', n_clusters=dictionarySize, batch_size=50)
dictionary.fit(BOW)
print(len(dictionary.cluster_centers_))

features = np.zeros((n,dictionarySize))

for i in range(n):
    
    segmentedUser = data['segmentation'][i]
    sourceImg = data['rgb'][i]
    mask2 = np.mean(segmentedUser, axis=2) > 150 # For depth images.
    mask3 = np.tile(mask2, (3,1,1)) # For 3-channel images (rgb)
    mask3 = mask3.transpose((1,2,0))
    img = sourceImg *mask3	
    img = img[:, 10:-10]
    sift = cv2.xfeatures2d.SIFT_create()
    # Create grid of key points.
    keypointGrid = [cv2.KeyPoint(x, y, 10)
                    for y in range(0, img.shape[0], 10)
                        for x in range(0, img.shape[1], 10)]
    
    # Given the list of keypoints, compute the local descriptions for every keypoint.
    (kp, descriptions) = sift.compute(img, keypointGrid)
    
    tmp = np.zeros(dictionarySize)
    distances= np.ones((len(descriptions),dictionarySize))
    
    for j in range(len(descriptions)):
        feat = descriptions[j]
        feat = np.reshape(feat,(1,-1))
        a = dictionary.predict(feat)
        tmp[a] = tmp[a] +1

    features[i,:] = tmp
        
    
    
print(features.shape)  


np.save('features1.npy', features) 
np.save('dict.npy', dictionary) 

classifier = trainAndEvaluate(features, data, n)


#calculate histograms for all images
#hogs = calculateHogs(n,data,mask3)
#print("done!")	

# reshape to obtain all hog words
#words =  np.reshape(hogs, (n*64, 8))
#print(words.shape)

# exclude zeros
#words= words[~np.all(words == 0, axis=1)]
#print(len(words))


# Cluster words into K clusters with kmeans
#kmeans = KMeans(n_clusters=k,max_iter=200, random_state=0).fit(words)
#print(kmeans.cluster_centers_)

#clusters = kmeans.cluster_centers_

# Create bow histograms
#bows = bagOfWordshistograms(hogs,clusters,n,k)
#print(bows.shape)



#------------------------------------------------------------------------

#classifier = trainAndEvaluate(bows, data, n)
#print(classifier.best_params_)

#labels_test = data['gestureLabels'][20000:25000]
#hogs2 = calculateHogs2(20000,25000,data,mask3);
#bows2 = bagOfWordshistograms(hogs2,clusters,20000,k)

#hogs2 = calculateHogs(n,test,mask3)
#bows2 = bagOfWordshistograms(hogs2,clusters,n,k)

#labels_predicted = classifier.predict(bows2)
#print("Result")
#print(classifier.score(bows2, test['gestureLabels']))
#
#print(clf.score(bows, data['gestureLabels'][0:n]) )
#print("Test Accuracy [%0.3f]" % ((labels_predicted == labels_test).mean()))


#print (data['gestureLabels'][0:100])
#print (labels_predicted[0:100])



#-------------------------------------------------------------------------
