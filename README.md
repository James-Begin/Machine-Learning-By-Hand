# Machine-Learning-By-Hand

Python libraries like Scikit-Learn and TensorFlow 2, it is incredibly easy to deploy machine learning models.
However, as an introduction to machine learning, it is important to understand the intricacies and details of how functions are derived and how the different parts work together. And, to understand what happens behind the scenes when using tools like TensorFlow. 

This project is an attempt to utilize and apply concepts that I have learned. This learning has come from many sources, some noteable ones are:

- [Introduction to Machine Learning with Python: A Guide for Data Scientists by Andreas C. Muller and Sarah Guido](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Deep Learning Prerequisites: Linear Regression in Python by The Lazy Programmer](https://www.udemy.com/course/data-science-linear-regression-in-python/)
- [Data Science: Deep Learning and Neural Networks in Python by The Lazy Programmer](https://www.udemy.com/course/data-science-deep-learning-in-python/)
- [Introduction to Deep Learning by MIT OpenCourseWare](https://ocw.mit.edu/courses/6-036-introduction-to-machine-learning-fall-2020/)

### Comparing Logistic Regression and Neural Networks
Because both methods of predictions (Regression and NN) are using the exact same dataset, they can be compared head-to-head with the same learning rate and training iterations. To compare, we can look at the final test accuracy of the models. This value ranges from 0 to 1, 1 being perfect accuracy. Because the weights are normally distributed the final accuracy can differ slightly.  

On average, Logistic regression reached an accuracy of 0.86 or 86% while the neural network reached an accuracy of 0.91 or 91%. This result is somewhat expected considering how the neural network is more complex and adaptable/flexible. It is crucial to note that Neural Networks require tweaking and refining to reach optimum results. This involves altering parameters like training iterations, the learning rate, and more. In some cases, a neural network can perform worse than regression without tuning.  
Here is an example of the accuracy curve of the neural network:
![NNplot](https://user-images.githubusercontent.com/103123677/200468932-900b8b2a-795f-4bab-9e27-b91ed88be57c.png)  
An example of the logitistic regression curve:
![REGplot](https://user-images.githubusercontent.com/103123677/200468945-2ea2aaf6-f634-4807-a092-3a919747b643.png)  
