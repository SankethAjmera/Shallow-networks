from dataset import load_data
import numpy as np
import random
from PIL import Image
##########################################################

##########################################################
# Load MNIST fashion data from a given path

data = load_data('/Users/sanketh/Documents/Python/fashion-MNIST')
print("Data loaded......................................")



#Retrieveing and defining sizes of the NN layers
def getsizes(data):
    input_layer_size = np.shape(data[0])[0]
    hidden_layer_size = 200
    output_layer_size = 10
    return input_layer_size, hidden_layer_size, output_layer_size



#Random initialisation of weights and biases, setting their dimesnions using the layer sizes
def random_initialise(nx, nh, ny):
    W1 = np.random.randn(nh,nx)*0.01
    b1 = np.zeros((nh,1))
    W2 = np.random.randn(ny,nh)*0.01
    b2 = np.zeros((ny,1))

    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters



#Sigmoid function
def sigmoid(z):
    sig = 1/(1+np.exp(-1*z))
    return sig


#Forward Propagation
def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)


    output = {"Z1": Z1, "A1": A1,"Z2": Z2,"A2": A2}
    return output, A2



#Compute the cost
def cost(A2,data):
    Y = data[1]
    m = np.shape(data[1])[1]
    temp = Y*np.log(A2)+ (1-Y)*(np.log(1-A2))
    cost = -np.sum(temp)/m
    cost = np.squeeze(cost)

    return cost



#Do back Propagation
def back_propagation(A2,Y,A1,X,parameters):
    m = 60000;
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis= 1, keepdims= True)/m
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis = 1, keepdims = True)/m

    gradients = grads = {"dW1": dW1, "db1": db1,"dW2": dW2,"db2": db2}

    return gradients



#Perform optimization through gradient descent
def gradient_descent(parameters, gradients, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = W1-dW1*learning_rate
    b1 = b1-db1*learning_rate
    W2 = W2-dW2*learning_rate
    b2 = b2-db2*learning_rate

    parameters = {"W1": W1, "b1": b1,"W2": W2,"b2": b2}

    return parameters

#Combine everything and learn parameters
def model(data, epochs, print_cost,batch_size):
    X = data[0]
    Y = data[1]

    (nx, nh, ny) = getsizes(data)
    parameters = random_initialise(nx, nh, ny)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    for i in range(0,epochs):
        j = 0;
        while j< 60000:
        #Forward propagation
            output, A2 = forward_propagation(data[0],parameters)
        #Compute the cost function
            cost1 = cost(A2,data)
        #Back Propagation
            #np.shape(A2)
            a2 = A2[:,j:j+batch_size]
            #a2 = a2[:, np.newaxis]
            y = Y[:,j:j+batch_size]
            #y = y[:, np.newaxis]
            A1 = output["A1"]
            A1 = A1[:,j:j+batch_size]
            #A1 = A1[:, np.newaxis]
            X = data[0][:,j:j+batch_size]
            #X = X[:, np.newaxis]
            np.shape(X)

            gradients = back_propagation(a2,y,A1,X,parameters,)
        #Update parameters
            parameters = gradient_descent(parameters, gradients, 1) #Set learning rate here

            if print_cost:
                print ("Cost after epoch " + str(i) + " batch " + str(round(j/600,2)) +  "% : " + str(round(cost1,4)))
            j = j+ batch_size

    return parameters

parameters = model(data,epochs= 1, print_cost=True,batch_size=10000)

def predict(parameters,data):
    X = data[0]
    Y = data[1]
    X_test = data[2]
    Y_test = data[3]
    output_train, A2_train = forward_propagation(X,parameters)
    output_test, A2_test = forward_propagation(X_test, parameters)
    A2_train = np.argmax(A2_train, axis = 0)
    A2_test = np.argmax(A2_test,axis=0)

    Y = np.argmax(Y, axis = 0)
    Y_test = np.argmax(Y_test, axis = 0)

    predictions_train = (A2_train == Y)
    predictions_test = (A2_test == Y_test)

    return  predictions_train, predictions_test, Y_test, A2_test

predictions_train,predictions_test,Y_test,A2_test = predict(parameters,data)
print("Training Accuracy : " + str(np.mean(predictions_train)*100))
print("Test set Accuracy : " + str(np.mean(predictions_test)*100))


def predict_single(index,data,A2_test, Y_test):

    X_test = data[2] #test data
    X = X_test[:,index]*255 #reverting back to original scale
    X = X.reshape(28,28) #reshaping the 784 dim vector to 28*28 matrix
    im = Image.fromarray(X)
    im.show()
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    print("Predicted class is "+ classes[A2_test[index]])
    print("Actual class is "+ classes[Y_test[index]])

index = random.randint(0,10000)
predict_single(index,data, A2_test, Y_test)
