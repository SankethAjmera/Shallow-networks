from dataset import load_data
import numpy as np
##########################################################

##########################################################
# Load MNIST fashion data from a given path

data = load_data('/Users/sanketh/Documents/Python/fashion-MNIST')
print("Data loaded......................................")

#Retrieveing and defining sizes of the NN layers
def getsizes(data):
    input_layer_size = np.shape(data[0])[0]
    hidden_layer_size = 100
    output_layer_size = 10
    return input_layer_size, hidden_layer_size, output_layer_size


(nx, nh, ny) = getsizes(data)




#Random initialisation of weights and biases, setting their dimesnions using the layer sizes
def random_initialise(nx, nh, ny):
    W1 = np.random.randn(nh,nx)*0.01
    b1 = np.zeros((nh,1))
    W2 = np.random.randn(ny,nh)*0.01
    b2 = np.zeros((ny,1))

    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters

parameters = random_initialise(nx, nh, ny)


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

output, A2 = forward_propagation(data[0],parameters)


#Compute the cost

def cost(A2,data):
    Y = data[1]
    m = np.shape(data[1])[1]
    temp = Y*np.log(A2)+ (1-Y)*(np.log(1-A2))
    cost = -np.sum(temp)/m
    cost = np.squeeze(cost)

    return cost

cost(A2,data)


#Do back Propagation

def back_propagation(parameters, output, data):
    m = np.shape(data[1])[1];


    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = output["A1"]
    A2 = output["A2"]

    X = data[0]
    Y = data[1]

    temp = np.zeros((10,m))
    for i in range(10):
        for j in range(m):
            temp[i,j] = A2[i,j]-Y[i,j]
    dZ2 = temp
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


def model(data, iterations, print_cost ):
    X = data[0]
    Y = data[1]

    (nx, nh, ny) = getsizes(data)

    parameters = random_initialise(nx, nh, ny)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0,iterations):
        #Forward propagation
        output, A2 = forward_propagation(X,parameters)
        #Compute the cost function
        cost1 = cost(A2,data)
        #Back Propagation
        gradients = back_propagation(parameters,output,data)
        #Update parameters
        parameters = gradient_descent(parameters, gradients, 0.5)
        if print_cost and  i%10  ==0:
            print ("Cost after iteration %i: %f" %(i, cost1))

    return parameters

parameters = model(data,iterations= 200, print_cost=True)

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

    return  predictions_train, predictions_test





predictions_train,predictions_test = predict(parameters,data)


print(np.mean(predictions_train)*100,np.mean(predictions_test)*100)
