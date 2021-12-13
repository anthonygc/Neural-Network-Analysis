### Name:
# Anthony Grant-Cook

# Date: 13 December 2021

# Final project code

rm(list = ls()) # remove all variables in memory
graphics.off() # clear out all plots from previous work.
cat("\014") # clear the console

# add your libraries here

library(tidyverse)

# install.packages("psych")
library(psych)

library(tibble)

# add your code here. Be sure to leave your data file(s) in the data/ directory of this repository.

# Section 01: ANN & KNN analysis on smaller data set



# install package

install.packages("neuralnet")

# Libraries

require(neuralnet)



# Neural Network analysis
# creating a data set with variables that contain non-categorical numerical values.
# TKS is a variable which stands for Technical Knowledge Score which ranges from 1-100.
# CSS is a variable which stands for Communication Skill Score which ranges from 1-100 as well.
# The context of this data set is based around if the student is placed or not place depending on how well the student's score are for both variables.
# `1` means that the student is placed while `0` means that the student is not placed.

# I utlized information about neural networks and how they can be implemented using this website: https://www.datacamp.com/community/tutorials/neural-network-models-r

probtopredict <- function(g){
  TKS=c(20,10,30,20,80,30)
  CSS=c(90,20,40,50,50,80)
  Placed=c(1,0,0,0,1,1)
  
  # combination of columns into single data set
  
  df=data.frame(TKS,CSS,Placed)
  
  summary(df)
  # fitting the neural network
  
  ptm <- proc.time()
  
  nn=neuralnet(Placed~TKS+CSS,data=df, hidden=3,act.fct = "logistic",
               linear.output = FALSE)
  
  # plot the neural network
  
  plot(nn)
  
  # Create test data set using technical knowledge score and communication skills score
  
  TKS=c(30,40,85)
  CSS=c(85,50,40)
  
  # Predict the result
  
  test=data.frame(TKS,CSS)
  Predict=compute(nn,test)
  Predict$net.result
  
  # Convert prediction to probabilities. Convert probabilities into binary classes.
  
  
  prob <- Predict$net.result
  pred_time <- list(a = ifelse(prob>0.5, 1, 0), b = proc.time() - ptm)
  return(pred_time)
}

probtopredict(g)

# Calculate the average error upon 8 iterations of code.
# I made sure to record data after every iteration.

error_sum <- sum(0.749999, 0.000832, 0.006433, 0.002052, 0.002546, 0.750049, 0.75, 0.003247)
error_avg <- error_sum / 8
error_avg

# Calculate the average of time (elapsed) upon 8 iterations.
# I made sure to record the data found
time_sum <- sum(.001, 0, 0, .001, 0, .004, 0, .001)
time_avg <- time_sum / 8
time_avg # time average = 0.000875

error <- c(0.749999, 0.000832, 0.006433, 0.002052, 0.002546, 0.750049, 0.75, 0.003247)

# Calculate the variance and standard deviation of the steps

var_error <- var(error)
var_error # variance = 0.1494668
stdev_error <-  sqrt(var(error))
stdev_error # standard deviation = 0.3866094

steps <- c(13, 105, 124, 127, 107, 3, 10, 113)

# Correlation analysis

plot(x = error, y = steps)
cor(error, steps) # Here we have a very strong negative correlation


# Calculate the average amount of steps taken.
sum_steps <- sum(steps)
avg_steps <- sum(steps) / 2
avg_steps # 306 Step functions on average.

# Calculate the variance and standard deviation of the steps
var_steps <- var(steps)
var_steps # Variance = 3188.75 (This is a very high variance)
stdev_steps <- sqrt(var(steps))
stdev_steps # Standard deviation = 56.46902 (The standard deviation is very high thus extremely spread out)

# --------------------------------------------------------------------

# KNN analysis

# Clear data
rm(list = ls()) # remove all variables in memory
graphics.off() # clear out all plots from previous work.
cat("\014") # clear the console

# Load Library
library(tidyverse)
library(gmodels)

install.packages("class")
library(class)

# For this analysis, I am using a segment of the iris data set that is provided natively in R. 
# However, to ensure that the efficiency is un-biased (meaning that there is more data utilized), I will only use the same amount of rows as
# the data set that I had initially created.

# It should be noted that I followed a tutorial and gained some understanding on how KNN algorithms work using: https://www.datacamp.com/community/tutorials/machine-learning-in-r#six

# Correlation of iris species with length and width. This is an example we did in class

View(iris)
summary(iris)
ggplot(data = iris) + geom_point(mapping = aes(x = Sepal.Length, y = Sepal.Width,color = Species ))
cor(iris$Petal.Length, iris$Petal.Width)

# Here we see there is a high correlation.

# Next, we must wrangle the data in order to match the same number of rows and columns as the initial data set.

setosa_ <- filter(iris, Species == "setosa")
setosa_ <- head(setosa_,2)
setosa_

versicolor_ <- filter(iris, Species == "versicolor")
versicolor_ <- head(versicolor_, 2)
versicolor_

virginica_ <- filter(iris, Species == "virginica")
virginica_ <- head(virginica_, 2)
virginica_

iris_df <- data.frame(setosa_, versicolor_, virginica_)
iris_df
summary(iris_df)

# Next we normalize the function. I will time this as it is part of the KNN Machine Learning Process.

normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

iris_norm <- as.data.frame(lapply(iris_df[1:4], normalize))
summary(iris_norm)

# Generate random seed

set.seed(1234)

# Take sample of rows to get an equal amount of each species
ind <- sample(2, nrow(iris_df), replace=TRUE, prob=c(0.67, 0.33))

# Split into Training & Testing

# Compose training set
iris.training <- iris_df[ind==1, 1:4]

# Inspect training set
head(iris.training)

# Compose test set
iris.test <- iris_df[ind==2, 1:4]

# Inspect test set
head(iris.test)

iris.trainLabels <- iris_df[ind==1,5]
print(iris.trainLabels)

iris.testLabels <- iris[ind==2, 5]
print(iris.testLabels)


# building the model and timing the knn algorithm.
# It should be noted that I am not trying to figure out the efficiency of KNN itself.
# I will use a function to encapsulate the KNN model so that I am able to calculate the time that it takes.

KNN <- function(g){
  ptm <- proc.time()
  iris_pred <- knn(train = iris.training, test = iris.test, cl = iris.trainLabels, k=2)
  pred_time <- list(a = iris_pred, b = proc.time() - ptm)
  return(pred_time)
}

# Rerun the function for 8 iterations.
# To do this, we can use the purrr library and call the method `rerun`

library(purrr)
n <- 8
rerun(8, KNN(g))

elapsed_time_sum <- sum(0.001, 0, 0, 0.001, 0, 0, 0, 0)
elapsed_time_avg <- elapsed_time_sum / 8
elapsed_time_avg # The elapsed time average is equal to 0.00025


# (Did you remember to add your name to this script?)
