##### Name(s) :
Anthony Grant-Cook

##### Date :
13 December 2021

##### Report Title :

```
Neural Networks in R
```

For this project I wanted to analzyze how Neural Networks and what its use cases may be. I also wanted to know its efficiency in regression problems and compare it to a common machine learning algorith. The algorithm I chose to compare was the K-Nearest-Neighbor model (KNN). This model is something we have talked about briefly in class. The KNN model is one of the most famous machine learning algorithms. It is also one of the first algorithms that people learn when studying machine learning.

I wanted to answer three questions for my project.

1. What is a Artificial Neural Network (ANN) and how can one be implemented?

2. How does a Neural Network's "Step Function" relate to it "Error"?

3. What is the efficiency of Neural Networks compared to that of K-Nearest-Neighbors?

For the implementation of the code, I planned to split this into two sections. The first section deals with the comparison of both algorithms using a smaller data set. The second section focuses on the comparison of both algorithms in larger data sets. For the first section, I utilized a tutorial for how Neural Networks can be implemented. This resource also gave some context behind Neural Networks. Furthermore, I created my own data sets with 3 colomns and 6 rows. For the KNN algorithm, I simply used the built in iris data set. However, I wrangled it to be that of the same rows as ANN to give a reasonably unbiased comparison. For the second section, I utilized one of google's open source data repositories. This repository centers around the question of how likely Neural abstractive models "hallucinate" summarizations of texts. In other words, how likely are they to alter or change summarizations. Basically increasing the liklihood of falsifying summarizations. I compared this to the KNN model and utilized the built in iris data set.

##### Analysis 

- Question 01 **What is a Artificial Neural Network (ANN) and how can one be implemented?**
An Artificial Neual Network is a deep learning algorith that is structured and organized to be similar that of how neurons work. Thus, they essentially solve problems that are harder computers but easier for humans to figure out. Some examples of these problems include pattern detection, speech recognition, and facial recognition. A neural network is seperated into individual nodes. Each node has data associated to it as well as weights and a bias. The nodes themselves can be thought of as a linear regression model.

`Output = sum(weight * inputs) + bias`

The statement here signifies how a node works. Although, it should be noted that there are three layers to the node. That being the input layer, the hidden layer, and the output layer. The input layer is were we assign our input variables to train off of. The hidden layer is where transformations occurr, and the output layer is where the output is utilized to see if it passes a certain threshold. By looking at the equation above, we see that the output is equal to the sum of the weight times the input. The bias can be thought of as the y-intercept of this linear regression model. If we were to scale the weights or bias, you would essentially change the output. For regression analysis of the data, it is important to have some sort of binary or categorical value. When this is present, we are able to classify where data belongs to. In the first section (`analysis.r`) I created a variable for Technical Knowledge scores (`TKS`)and a variable for Communication skill score (`CSS`). I also created a column for placement based off of the score for those variables. The placement column consisted of binary values `1` and `0`. 1 would indicate that the student is placed and 0 would indicate that the student is not placed. Based upon the pattern of the training data which includes all 3 variables, the ANN train on this to predict the placement of students. Although, ANN algorithms really shine when they predict the placement using a test data set after training. This concept is the same for bigger data sets. For example, in section 02, I have selected an open source data set from google wich has implied binary values which indicates whether a particular neural abstractive summarization model is factual or not factual. This was a bit more of a challenge as I had to round some of the values to get it to a whole number for binary outputs. For the training data, I used the columns `BERTscore`, `Faithful`, and `Factual`. `Factual` is categorical while `BERTscore` and `Faithful` are non-categorical. The ANN is then used to train off of the training data (or reference) and will then make a predictive model based off of testing data. In general, this is evidence that we are doing supervised machine learning.

- Question 02 **How does a Neural Network's "Step Function" relate to it "Error"?**
A Neural Network's "step function" is how many times a node activates within the Artificial Neural Network. As demonstrated in the code of the first section, we see that as the number of steps increases, the error decreases. Thus, making it more accurate when compared with the reference data. In section 01 (`analysis.r`) we have a very strong negative correlation of `-0.9893943179563979` which indicates the conclusion that the error decreases with respect to the node activation.

- Questions 03 **What is the efficiency of Neural Networks compared to that of K-Nearest-Neighbors?**

When comparing the elapsed time of both algorithms using the same size data set, we see that the average for the ANN computation takes about 0.000875 seconds. The KNN computation takes about 0.00025 seconds. Although, in real life situations, data will often not be that small. In many cases, we would usually use machine learning algorithms when we are dealing with regression problems with a higher quantity of objects. When we look at section two of the code (`analysis2.r`) we see that the KNN algorithm takes about 0.6907499999313131 seconds whilst the ANN algorithm takes about 24.44395 seconds to compute. This may be heavily due to the fact that I am utilizing a recursive "feedback" algorithm rather than a linear "feedforward" algorithm. This could also be heavily due to the fact that the we are rendering plots based off of the data. Thus, in situations where we need to find answers in the most efficient timely way, we would utilize the K-Nearest-Neighbor algorithm. It should also be noted that the step function for section 02 had a very high variance. This variance which leads to a high standard deviation. Thus the step functions output is more spread out and is not consistent. This could lead to varying errors.



(Did you remember to add your name(s) to the top of this document?)
