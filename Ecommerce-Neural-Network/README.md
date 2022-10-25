# Neural Network "By Hand"
## Intro
The goal for this project is to apply my learning and create a neural network without the use of libraries like Tensorflow and Scikit in order to help understand how machine learning works. Additionally, I want to compare logistic regression and neural networks using the same ecommerce dataset.

## The Data
The data in this project is an ecommerce dataset feature in the course ["Data Science: Deep Learning and Neural Networks in Python" by The Lazy Programmer](https://www.udemy.com/course/data-science-deep-learning-in-python/) and can be found on [Github](https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv). The dataset is meant to replicate a real world data from an online store. The set contains 6 categories and 500 data points in each category. 

The categories include:
- Whether the user is on a mobile device (Binary)
- The # of products viewed (Continuous)
- Visit duration (Continuous)
- Whether the user has visited before (Binary)
- Time of day (Categorical)

The final category is the user action (Categorical) and will act as the targets for the neural network.

### Processing the Data

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
This neural network has only one hidden layer that contains 5 nodes. This is an arbitrary amount and can be increased or decreased depending on the use case.  
Because of this, there are two sets of weights and biases to be accounted for. The first set works between the input and hidden layer and the second set between the hidden and output layer.  

To initialize weights, we cant have them all be zero as training would become a very slow process. Instead, we can create a generate a normally distrubuted set of weights of size (Columns, Hidden Nodes) for the first set as this represents the number of input nodes and hidden nodes. For biases, we can initialize them as zeroes as the weights already fire the neurons in the beginning. Although, in certain cases, it may be necessary to set biases to a small amount such as 0.001.
```
Weights1 = np.random.randn(Cols, Hidden_nodes)
Bias1 = np.zeros(Hidden_nodes)
```
In the second set of weights and biases, the weights are distrubuted normally in an array of size (Hidden_nodes, Classes) as this represents the connection between the hidden layer nodes and output nodes. Again, the biases are initialized as zero.
```
Weights2 = np.random.randn(Hidden_nodes, Classes)
Bias2 = np.zeros(Classes)
```

### Activation Function
Activation functions are used to find the output of a node. Softmax is usually used for multi-classification problems. This is because softmax outputs a probability for each ouput node that sums to 1.
```
return np.exp(var) / np.sum(np.exp(var), axis = 1, keepdims = True)
```

### Feed Forward
Simply, feed forward is the process of passing the data throught the neural network. Here we multiply the input by the weights, add the bias, and apply the softmax function. We do this for both the hidden layer and the output layer.
```
def feed_forward(Input, Weights1, Bias1, Weights2, Bias2):
    Hidden_values = softmax(Input.dot(Weights1) + Bias1)
    return softmax(Hidden_values.dot(Weights2) + Bias2), Hidden_values
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
Next, we create a loop for training. The number of loops is again arbitrary and should be tweaked. Then, we pass the inputs, weights, and biases through the feed forward function. We do this twice, for training and testing data.
```
for i in range(100000):
    #here we train the inputs with the allocated samples, and then test using the rest of the samples, recall that there are two sets of weights and biases
    A_train, B_train = feed_forward(Train_Inputs, Weights1, Bias1, Weights2, Bias2)

    A_test, B_test = feed_forward(Test_Inputs, Weights1, Bias1, Weights2, Bias2)
```
Now, we iteratively use gradient descent and our learning rate to find the minimum loss (best accuracy) of our network. The formula for gradient descent is simplified, but is derived through the use of derivatives in Deep Learning and Neural Networks in Python by the Lazy Programmer. To update our weights using this formula, we use backpropagation. Moving backwards through the network, meaning we update our second layer of weights and biases before the first. The adjustment for the first layer weights is based on the adjustment for the second layer and using the _hidden_ variable makes this easier.
```
Weights2 -= learning_rate * B_train.T.dot(A_train - Train_Targets_ind_matrix)
Bias2 -= learning_rate * (A_train - Train_Targets_ind_matrix).sum()

hidden = (A_train - Train_Targets_ind_matrix).dot(Weights2.T) * (1 - B_train * B_train)

Weights1 -= learning_rate * Train_Inputs.T.dot(hidden)
Bias1 -= learning_rate * hidden.sum(axis = 0)
```
Here is an example of the formula for the gradient of our loss function in matrix form. In this case, _J_ is the loss function, _X_ is the output values, _T_ is the hidden_values in between the layers as described in the _feed_forward_ function. Finally, _Y_ is the target data (values we want to hit).
![Screenshot 2022-10-24 222242](https://user-images.githubusercontent.com/103123677/197667226-32040ae2-3875-49b7-a950-cfefbeadf5cb.png)
