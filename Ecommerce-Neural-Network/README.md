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
`Inputs = data[:, :-1]`  
 `Targets = data[:, -1]`  
  
Then, we have to normalize the columns containing continuous data such as the visit duration and # of products viewed. In this case, we can normalize using the Z-score. The formula for which is (x - mean) / standard deviation.  
`Inputs[:, 1] = (Inputs[:,1] - Inputs[:,1].mean()) / Inputs[:,1].std()`  
`Inputs[:, 2] = (Inputs[:, 2] - Inputs[:, 2].mean()) / Inputs[:, 2].std()`  
