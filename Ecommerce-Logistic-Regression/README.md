# Logistic Regression "By Hand"

## Intro
The goal for this project is to apply my learning and create a neural network without the use of libraries like Tensorflow and Scikit in order to help understand how machine learning works. Additionally, I want to compare logistic regression and neural networks using the same ecommerce dataset.

Logistic regression is usually used for binary classification problems but can be modified for multi class classification (also called one-vs-all).
The data used for this model is Ecommerce data used to represent what would be found in the real world.
The data is a csv file and contains categories such as whether the user is on a mobile device, how many products they viewed, visit duration, time of day, and more.
The Result of the data is the user action and is used for both training and testing.
Furthermore, the data contains different categories like binary values, categoriacal values, and continuous data like visit duration.
Because of the different types of data, it needs to be cleaned and prepared using techniques like one-hot encoding and indicator matrices for the categorical data.

## The Data
The data in this project is an ecommerce dataset feature in the course ["Data Science: Deep Learning and Neural Networks in Python" by The Lazy Programmer](https://www.udemy.com/course/data-science-deep-learning-in-python/) and can be found on [Github](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv). The dataset is meant to replicate a real world data from an online store. The set contains 6 categories and 500 data points in each category. 

The categories include:
- Whether the user is on a mobile device (Binary)
- The # of products viewed (Continuous)
- Visit duration (Continuous)
- Whether the user has visited before (Binary)
- Time of day (Categorical)

The final category is the user action (Categorical) and is the targets we want to accurately predict

### Processing the Data

Between Logistic Regression and a Neural Network, there is little difference in the data cleaning and processing for this dataset.
First, we have to split the data into the inputs and targets. The targets are the last column, while the rest are the inputs.  
```
Inputs = data[:, :-1]
Targets = data[:, -1]
```
Then, we have to normalize the columns containing continuous data such as the visit duration and # of products viewed. In this case, we can normalize using the Z-score. The formula for which is (x - mean) / standard deviation.  
```
Inputs[:, 1] = (Inputs[:,1] - Inputs[:,1].mean()) / Inputs[:,1].std()
Inputs[:, 2] = (Inputs[:, 2] - Inputs[:, 2].mean()) / Inputs[:, 2].std()
```
Next, we have to use one-hot encoding to format the categorical data (Time of day). There are four time categories, 0-6, 6-12, 12-18, and 18-0. One-hot encoding converts categorical data into a matrix where each column is a category and a "1" is placed in the column representing the category, leaving the rest "0"s. For example, a category of 3 would be [0,0,1,0] when one-hot encoded.  

In this case, there are 4 different categories, meaning we have to add 3 more columns to the input data array (we already have one with the original data).  
```
Inputs_new = np.zeros((Rows, Cols+3))
Inputs_new[:, 0:(Cols-1)] = Inputs[:, 0:(Cols-1)]
```
To one hot encode, we iteratively find the index to place the "1" and add it to the zero matrix we added to the input data. Recall the index where the "1" is placed is the same as the category number.  
```
for n in range(Rows):
    ind = int(Inputs[n, Cols-1])
    Inputs_new[n, ind+Cols-1] = 1
```  

Then, we have to create an indicator matrix for the targets. Because our targets are categorical, we need to one-hot encode them as well, this is called an indicator matrix. The method is similar to one-hot encoding the input data. We first create an empty indicator matrix, which is the number of samples by the number of categories. Then, iteratively place "1"s in their respective positions.  

```
def target_indicator(targets, K):
    numClasses = len(targets)

    #initialize a indicator matrix with zeroes
    indicator_matrix = np.zeros((numClasses, K))
    #fill in the indicator matrix, we place the ones in the column equal to the target label (targets[i])
    for i in range(numClasses):
        indicator_matrix[i, targets[i]] = 1
    return indicator_matrix
 ```
Finally, we specify how much of the data we want to allocate for training and testing and then create the indicator matrix for testing and training based on that.
This part is guesswork and can be changed depending on the situation. For now, half the data is used for training, while half is for testing.  

```
#Classes (K) is the number of distinct possible outcomes
Classes = len(set(Targets))

Train_Inputs = Inputs[:250]
Train_Targets = Targets[:250]

#we pass in the number of targets for training and # of classes to create the indicator matrix
Train_Targets_ind_matrix = target_indicator(Train_Targets, Classes)

#test our model on the remaining 250 samples and create another indicator matrix
Test_Inputs = Inputs[250:]
Test_Targets = Targets[250:]
Test_Targets_ind_matrix = target_indicator(Test_Targets, Classes)
```

### Weights and Biases
Unlike a neural network, Logistic Regression can be thought of a as a single layer of neurons in a network. In our case, we have 5 distinct categories, so 5 "neurons". Input data is passed into the layer. The data is multiplied by the weights and bias is added or subtraction. Then an activation function like softmax is applied to return a single output. To initialize the weights, distribute them normally in a matrix with size # of input columns by # of output classes. Biases are intialized at 0 and placed in a 1D array with # of classes as rows. This is because there is a bias term for every weight.
```
Weights = np.random.randn(Cols, Classes)

Biases = np.zeros(Classes)
```
### Activation Function
Activation functions are used to find the output from multple inputs. Softmax is usually used for multi-classification problems because it outputs a probability that sums to 1.
```
return np.exp(var) / np.sum(np.exp(var), axis = 1, keepdims = True)
```
### Feed Forward
Simply, feed forward is the process of passing the data throught the neural network. Here we multiply the input by the weights, add the bias, and apply the softmax function. We do this for both the hidden layer and the output layer.
```
def feed_forward(Input, Weights, Biases):
    return softmax(Input.dot(Weights) + Biases)
```
### Prediction
Here we just find the prediction of our output. The output produced by softmax is an array of probabilities. By using the argmax function, we can find the greatest probability and the prediction.
```
def Prediction(Probability_of_Targets):
    return np.argmax(Probability_of_Targets, axis = 1)
```
### Accuracy / Classification Rate
To find how accurate the model is, count how many times our predictions matched the target predictions. Dividing the total count by the amount of targets, we can find the % accuracy.
```
def accuracy(Targets, Outputs):
    count = 0
    for i in range(len(Targets)):
        if Targets[i] == Outputs[i]:
            count += 1
    return count / len(Targets)
```
### Loss Function
The loss function calculates how far from the correct result the model is. Because this is a multi-class model, the function used is cross-entropy loss. The formula for which can be found at https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e.
```
def loss_function(Targets, Probability_of_Targets):
    return -np.mean(Targets * np.log(Probability_of_Targets))
```
### Training
To train the model we have to use backpropagation, where we alter weights and biases as we move bakcwards through the model. To control the speed of learning, we define the learning rate. The learning rate is arbitrary and should be tweaked.
```
learning_rate = 0.0001
```
Next, we create a loop for training. The number of loops is again arbitrary and should be tweaked. Then, we pass the inputs, weights, and biases through the feed forward function. Here, regression differs from a Neural Network. Because there is only one "layer" of nodes, we only need to adjust weights once. This uses the same formula for gradient descent shown in the Neural Network portion. In the first part of training, we feed forward the training and test data.
```
for i in range(100000):
    #here we train the inputs with the allocated samples, and then test using the rest of the samples
    train = feed_forward(Train_Inputs, Weights, Biases)
    test = feed_forward(Test_Inputs, Weights, Biases)

    #here, we using the indicator matrix (the targets we want to hit) from before and compare it to the data we just trained and tested.
    train_loss = loss_function(Train_Targets_ind_matrix, train)
    test_loss = loss_function(Test_Targets_ind_matrix, test)

    #record our loss data
    training_costs.append(train_loss)
    test_costs.append(test_loss)
```
Using the formula for gradient descent we update the weights to try and try to minimze our loss. 
```#update our weights using the learning rates
    #The formula for gradient descent is used as an iterative method to find the minimum of the cost function.
    #To find the minimum, we want to solve the derivative of the Loss function with respect to the weights when it equals 0
    #In that case, we need to solve for the weights, but we want algebraically, so we use an iterative method like gradient descent instead.
    #The formula for gradient descent is derived intuitively in lesson 33 of the Lazy Programmer's Deep learning and Neural Networks in Python
    Weights -= learning_rate * (Train_Inputs.T.dot(train - Train_Targets_ind_matrix))

    Biases -= learning_rate * (Train_Targets.T.dot(test - Test_Targets_ind_matrix))
```




The csv file can be found at https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
