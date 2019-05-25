from mnist import MNIST
import numpy as np



# Load MNIST fashion data from a given path
def load_data(path):
    mndata = MNIST(path)

    images_train,labels_train = mndata.load_training() #Training set loaded
    images_test, labels_test = mndata.load_testing() #Test set loaded

    labels_train = np.array(labels_train) #Converting the array into np arrays
    labels_test = np.array(labels_test)
    images_test = np.array(images_test)/255#Normalising the inputs on a scale of 0-1
    images_train = np.array(images_train)/255

    labels_train = labels_train.reshape(1,(len(labels_train))); #Reshaping the arays in congruous with the convention
    labels_test = labels_test.reshape(1,(len(labels_test)));
    images_test = images_test.T;
    images_train = images_train.T;


    Y1 = np.zeros((10,60000));
    Y2 = np.zeros((10,10000));

    for i in range(0,60000):
        Y1[labels_train[0][i],i] = 1;

    for i in range(0,10000):
        Y2[labels_test[0][i],i] = 1;


    return images_train,Y1,images_test, Y2
