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
