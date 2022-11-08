import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data from https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/ann_logistic_extra/ecommerce_data.csv
#the data is a csv containing 6 columns and 500 rows of data. The each row is a seperate user and shows whether they shop online, visit duration, etc+.
#the last column is the user_action, which is what we eventually want to predict
def cleandata():
    df = pd.read_csv('ecommerce_data.csv') #import data
    #convert to numpy array
    data = df.to_numpy()

    #split into x and y as y is our user action and the output, while all others are the input features
    Inputs = data[:, :-1]
    Targets = data[:, -1]

    #these numerical columns are products viewed, and visit duration. Because these columns can have any value >0, we have to put them on a similar scale to the rest of the data
    #normalize the columns, this is just x - mean(x) over the standard dev of x (Z-score)
    #this way, we center the values around zero with a standard deviation of 1, making functions like tanh and sigmoid more effective
    Inputs[:, 1] = (Inputs[:,1] - Inputs[:,1].mean()) / Inputs[:,1].std()
    Inputs[:, 2] = (Inputs[:, 2] - Inputs[:, 2].mean()) / Inputs[:, 2].std()

    #for the time of day column, the data is split into 4 categories, one for 00-06, 06-12, 12-18, etc.
    #We cant feed categories to the model, so we need to one-hot encode (more below)
    Rows, Cols = Inputs.shape #get dimensions of input data
    
    Inputs_new = np.zeros((Rows, Cols+3)) #To one hot encode, we need one column for each category, we have one already, add 3 more
    #copy the other data from old Inputs to new Inputs excluding the time of day.
    Inputs_new[:, 0:(Cols-1)] = Inputs[:, 0:(Cols-1)]

    #iteratively one hot encode
    for n in range(Rows):
        t = int(Inputs[n, Cols-1]) #get the index of where to encode from the time of day column in the old input matrix
        #The index where we place the one is the same as the column #. As the encoded matrix is at the end of inputs, we have to add t to the original length - 1
        Inputs_new[n, t+Cols-1] = 1 #encode

    return Inputs_new, Targets

def getbinarydata():
    #here we just want the data that is binary like whether the user is on mobile or if their a returning visitor
    Inputs, Targets = cleandata()
    binaryInputs = Inputs[Targets <= 1] #all the inputs where the targets are <= 1
    binaryTargets = Targets[Targets <= 1]

    return binaryInputs, binaryTargets


#indicator matrix
#if there are k classes, there will be k class indicators y1, y2, y3, ... yk
#for example, we give labels like "Corn", "Apple", "Banana" numerical values 0, 1, 2.
#this is because we need to index arrays of data, and we cant use strings for that.
#Additionally, scikit-learn used to require you to convert labels to values 0 -> k-1. This is no longer required, but a good exercise.
#After converting labels to integers, this array is one-hot encoded:
#for example, there are 5 categories (classes) and 6 samples,
#After converting labels to numbers: [2 4 3 1 2 0]
#By one hot encoding, we create a matrix with 6 (samples) rows, and 5 (classes) columns
#[0 0 1 0 0    Here, the prior array with numerical labels shows where to place the 1 in each row
# 0 0 0 0 1    This is what we want the indicator matrix to look like.
# 0 0 0 1 0    The reason for this is these are targets, what we want to correctly guess.
# 0 1 0 0 0    These are like the answers for the problem, each row has only one 1 because that is a probability
# 0 0 1 0 0    in the first row (0), it is 100% in category 2 and 0% in any other because this is already known
# 1 0 0 0 0]
def target_indicator(targets, K):
    numClasses = len(targets)

    #initialize a indicator matrix with zeroes
    indicator_matrix = np.zeros((numClasses, K))
    #fill in the indicator matrix, we place the ones in the column equal to the target label (targets[i])
    for i in range(numClasses):
        indicator_matrix[i, targets[i]] = 1
    return indicator_matrix



Inputs, Targets = cleandata()
#convert targets to integers, as they are currently floats
Targets = Targets.astype(int)

Rows = Inputs.shape[0]
Cols = Inputs.shape[1]
#Classes (K) is the number of distance possible outcomes
Classes = len(set(Targets))

#train using the first 250 samples, this can be changed
Train_Inputs = Inputs[:250]
Train_Targets = Targets[:250]
#we pass in the number of targets for training and # of classes to create the indicator matrix
Train_Targets_ind_matrix = target_indicator(Train_Targets, Classes)

#test our model on the remaining 250 samples and create another indicator matrix
Test_Inputs = Inputs[250:]
Test_Targets = Targets[250:]
Test_Targets_ind_matrix = target_indicator(Test_Targets, Classes)


'''Weights and Biases'''
#To put it simply, logistic regression is similar to one layer of a neural network, each layer is made up of a set of neurons
#Each neuron has its own weight. When an input is passed through a neuron, its value is multiplied by the weight. Weights can be thought of the strength of the connection to that neuron, large weights affect the output significantly more than low weights
#Biases sit between the neurons and the output and usually add or subtract a certain value. Biases are useful as they allow you to shift functions left or right. While weights change the curvature of the function
#as a simple example: y = mx + b, the weights are like the slope and can make the curve steeper or more flat. The bias (b) allows you to shift the curve up or down, allowing for a better fit.
#in the case of non linear activation functions like sigmoid or tanh, the weights change the steepness of the curve, while the bias shifts the entire function left or right.
#At first, we intialize weights normally (gaussian) and set all biases to 0

Weights = np.random.randn(Cols, Classes)

Biases = np.zeros(Classes)

'''Activation Function'''
#activation functions are used to to get the output of a node.
#in this case we are using softmax as this is problem has multiple classes. Softmax provides a probability for each output and sums to one (100%)
#if this were a binary classification problem, we could use a function such as sigmoid
#The softmax formula for yi is the exponential divided by the sum of the exponential of yi for the entire vector y
def softmax(var):
    return np.exp(var) / np.sum(np.exp(var), axis = 1, keepdims = True) #axis 1 is summing across rows, keepdims may not be necessary but is helpful to maintain the dimension of the input.

'''Feed Forward'''
#here we pass in our data, multiply by weights, add biases, and apply the softmax function
def feed_forward(Input, Weights, Biases):
    return softmax(Input.dot(Weights) + Biases)

'''Prediction'''
#here we use the argmax function to find the output with the highest probability, this is our prediction
def Prediction(Probability_of_Targets):
    return np.argmax(Probability_of_Targets, axis = 1)

'''Accuracy'''
#Here we find how accurate the model is, the classification rate is the number of correct predictions divided by the total predictions
def accuracy(Targets, Outputs):
    count = 0
    for i in range(len(Targets)):
        if Targets[i] == Outputs[i]:
            count += 1
    return count / len(Targets)

'''Loss Function'''
#The loss function dictates how the model calculates how far away it is from the correct result and how well the model fits the data
#Because this is a multi-class model, we are using the cross-entropy loss function. The formula can be found here: https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e
def loss_function(Targets, Probability_of_Targets):
    return -(np.mean(Targets * np.log(Probability_of_Targets)))


'''Training'''
#initialize arrays to store our training cost and test cost values (the output from the loss function when we train and test)
training_costs = []
test_costs = []

#learning rate determines how much we adjust the weights with respect to the gradient
#In gradient descent, we want to find the minimum of the function (to minimize losses), the learning rate determines how fast we move along the gradient.
#the larger the learning rate, the faster we can move down the gradient, but it becomes more likely to overshoot the minimum and may never converge/diverge on it. The opposite occurs with a smaller learning rate, slower but more accurate.
#The learning rate is usually set naively but there are models and methods to choose a better learning rate
learning_rate = 0.0001

#training loop, here we can set how many times we want to train, here we just call our functions
for i in range(20000):
    #here we train the inputs with the allocated samples, and then test using the rest of the samples
    train = feed_forward(Train_Inputs, Weights, Biases)
    test = feed_forward(Test_Inputs, Weights, Biases)

    #here, we using the indicator matrix (the targets we want to hit) from before and compare it to the data we just trained and tested.
    train_loss = loss_function(Train_Targets_ind_matrix, train)
    test_loss = loss_function(Test_Targets_ind_matrix, test)

    #record our loss data
    training_costs.append(train_loss)
    test_costs.append(test_loss)

    #update our weights using the learning rates
    #The formula for gradient descent is used as an iterative method to find the minimum of the cost function.
    #To find the minimum, we want to solve the derivative of the Loss function with respect to the weights when it equals 0
    #In that case, we need to solve for the weights, but we want algebraically, so we use an iterative method like gradient descent instead.
    #The formula for gradient descent is derived intuitively in lesson 33 of the Lazy Programmer's Deep learning and Neural Networks in Python
    Weights -= learning_rate * (Train_Inputs.T.dot(train - Train_Targets_ind_matrix))

    Biases -= learning_rate * (Train_Targets.T.dot(test - Test_Targets_ind_matrix))

    #report some statistics
    if i % 100 == 0:
        print(f"{i}: Train loss: {train_loss}, Test Loss: {test_loss}")

#report final stats
print(f"Final Training Accuracy: {accuracy(Train_Targets, Prediction(train))}")
print(f"Final Test Accuracy: {accuracy(Test_Targets, Prediction(test))}")
#plot stats
plt.title("Logistic Regression Accuracy")
legend_train, = plt.plot(training_costs)
legend_test, = plt.plot(test_costs)
plt.legend(["Training Costs", "Test Costs"])
plt.show()
