# Logistic Regression "By Hand"

Logistic regression is usually used for binary classification problems but can be modified for multi class classification (also called one-vs-all).
The data used for this model is Ecommerce data used to represent what would be found in the real world.
The data is a csv file and contains categories such as whether the user is on a mobile device, how many products they viewed, visit duration, time of day, and more.
The Result of the data is the user action and is used for both training and testing.
Furthermore, the data contains different categories like binary values, categoriacal values, and continuous data like visit duration.
Because of the different types of data, it needs to be cleaned and prepared using techniques like one-hot encoding and indicator matrices for the categorical data.

## How it Works
Simply, Logistic Regression is similar to how a neuron in a neural network works.
The inputs are multiplied by the weights and added to the bias. Then, the sigmoid function is applied to provide an output.
Weights are trained using gradient descent and after training, the model is tested on a different portion of the data.
Throughout the code I try to comment in as much detail as needed.

The csv file can be found at https://github.com/lazyprogrammer/machine_learning_examples/blob/master/ann_logistic_extra/ecommerce_data.csv
