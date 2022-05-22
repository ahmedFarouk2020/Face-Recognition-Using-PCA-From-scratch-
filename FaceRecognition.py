import os
from sklearn.preprocessing import normalize
import numpy as np
import cv2 as cv

import pdb

class FaceRecongnition:
    def __init__(self, dirToInput:str, sample_size=300) -> None:
        self.__FilesList = 0  # list of sample's files names
        self.__DirToInput = dirToInput # dir to the sample
        self.__FaceMatrix = np.array([])
        self.__EigenFaces = np.array([])
        self.__SampleSize = sample_size

    def __getFilesList(self):
        """ Retrieve all files in a dir into a list """
        self.__FilesList = os.listdir(self.__DirToInput)
        print(self.__FilesList)

    def __constructFaceMatrix(self):
        
        face_matrix = []
        for i in range(len(self.__FilesList)):
            
            # read img as greyscale
            img = cv.imread(self.__DirToInput+"/"+self.__FilesList[i], cv.IMREAD_GRAYSCALE)
            print(img.shape)
            # resize img
            img = cv.resize(img, (100, 100))
            # convert img into vector
            numOfDimensions = img.shape[0]*img.shape[1]
            img = np.reshape(img, numOfDimensions)
            # add to face matrix
            face_matrix.append(img)

        self.__FaceMatrix = np.array(face_matrix)

    def __getZeroMeanMatrix(self) -> np.ndarray:
        """ get the mean sample and subtract it from all samples """
        mean_sample = np.mean(self.__FaceMatrix, axis=0)
        return np.subtract(self.__FaceMatrix, mean_sample) # zero-mean array

    def __getCovarianceMatrix(self, zero_mean_mat:np.ndarray):
        cov = (zero_mean_mat.dot(zero_mean_mat.T)) / (len(self.__FilesList)-1)

        # get eigenvalues and eigenvectors of cov
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        print("Eigen Values", eigenvalues)
        print("Eigen Vectors", eigenvectors)
        # Order eigenvalues by index desendingly
        idx = eigenvalues.argsort()[::-1]
        # sort eigenvectors according to eigen values order
        eigenvectors = eigenvectors[:, idx]
        # linear combination of each column of zero_mean_mat
        eigenvectors_c = zero_mean_mat.T @ eigenvectors
        # normalize the eigenvectors
        # normalize only accepts matrix with n_samples, n_feature. Hence the transpose.
        self.__EigenFaces = normalize(eigenvectors_c.T, axis=1)
        print("EigenFaces", self.__EigenFaces)

    def fit(self):
        self.__getFilesList()
        self.__constructFaceMatrix()
        zero_mean_arr = self.__getZeroMeanMatrix()
        print("zero_mean_arr", zero_mean_arr)
        self.__getCovarianceMatrix(zero_mean_arr)




recognizer = FaceRecongnition("train_data")
recognizer.fit()