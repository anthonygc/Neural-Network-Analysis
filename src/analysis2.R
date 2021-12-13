# Section 02: ANN & KNN analysis on bigger data set


rm(list = ls()) # remove all variables in memory
graphics.off() # clear out all plots from previous work.
cat("\014") # clear the console

# Load Library
install.packages("neuralnet")
library(neuralnet)
library(dplyr)

# Make sure to import the data set first
my_data <- eval_scores_xsum_summaries

# Factual
factual <- select(my_data, V8)
factual <- factual[-1,]
options(digits=1)
factual <- as.double(factual)
factual <- round(factual, digits = 0)
factual

# Faithful
faithful <- select(my_data, V7)
faithful <- faithful[-1,]
options(digits=16)
faithful <- as.double(faithful)
faithful

# Entailment

bertscores <- select(my_data, V5)
bertscores <- bertscores[-1,]
bertscores <- as.double(bertscores)
options(digits=16)
bertscores


# data frame


df=data.frame(bertscores,faithful,factual)
summary(df)

# The Neural Network algorithm is encapsulated by a function so that the time can be calculated
probtopredict <- function(g){
ptm <- proc.time() # Start timer
nn=neuralnet(factual~bertscores+faithful,data=df, hidden=3 ,act.fct = "logistic", stepmax=1e+07,
             linear.output = FALSE)

plot(nn)

# Create test data using BERT scores and Faithful scores
test=data.frame(bertscores,faithful)

# Predict the Result (since we have multiple libraries with the function compute, we must specify which library we are using)
Predict=neuralnet::compute(nn,test)
Predict$net.result

# convert prediction to probabilities. We then convert probabilities to binary classes
prob <- Predict$net.result
pred_time <- list(a = ifelse(prob>0.5, 1, 0), b = proc.time() - ptm) # End timer and store binary classes
return(pred_time)
}

probtopredict(g)
# It can be noticed that the error is very large. 

# Calculate the average error
error_sum <- sum(88.9981418, 89.00899, 88.299014, 89.009602, 89.047039, 88.3659, 88.256844, 89.005351)
error_avg <- error_sum / 8
error_avg # There is a very high error average for this data set. This might be due to the fact that we have a low hidden layer. 

# Calculate Standard Deviation of Error
error <- c(88.9981418, 89.00899, 88.299014, 89.009602, 89.047039, 88.3659, 88.256844, 89.005351)
var_error <- var(error)
var_error # Variance = 0.1347990431384053
stdev_error <-  sqrt(var(error))
stdev_error # Stdev = 0.367149891922094


# Calculate the step average
step_sum <- sum(160438, 97987, 79289, 161154, 66552, 291487, 164150, 139339)
step_avg <- step_sum / 8
step_avg

# Calculate Standard Deviation of Steps
step <- c(160438, 97987, 79289, 161154, 66552, 291487, 164150, 139339)
Var_step <- var(step)
Var_step # Variance = 5005531563.142858
stdev_step <-  sqrt(var(Var_step))
stdev_step 


# Calculate the time average (elapsed)
 
time_sum <- sum(27.218, 17.0149, 13.3240, 27.0230, 11.2459, 49.123, 27.4269, 23.1759)
time_avg <- time_sum / 8
time_avg # takes 24.44395 seconds

# Calculate the Standard Deviation of Time

time <- c(27.218, 17.0149, 13.3240, 27.0230, 11.2459, 49.123, 27.4269, 23.1759)
var_time <- var(time)
var_time # Variance = 140.9915569171428
stdev_time <- sqrt(var(var_time))
stdev_time # NA


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


# Create Normalize function

normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return (num/denom)
}

iris_norm <- as.data.frame(lapply(iris[1:4], normalize))
summary(iris_norm)

# Generate Random seed

set.seed(1234)

# Take sample of rows to get an equal amount of each species
ind <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))

# Split into Training & Testing

# Compose training set
iris.training <- iris[ind==1, 1:4]

# Inspect training set
head(iris.training)

# Compose test set
iris.test <- iris[ind==2, 1:4]

# Inspect test set
head(iris.test)

iris.trainLabels <- iris[ind==1,5]
print(iris.trainLabels)

iris.testLabels <- iris[ind==2, 5]
print(iris.testLabels)

# building the model and timing the knn algorithm.

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

elapsed_time_sum <- sum(0.0009999999992942321, 0, 0.0010000000002037268, 0, 0, 0.001000000000203727, 0, 0)
elapsed_time_avg <- elapsed_time_sum / 8
elapsed_time_avg # elapsed time average = 0.0003749999999627107

# I multiply the time with the difference in the number of objects for the ANN data set
time_comparison <- (0.0003749999999627107) * 1842
time_comparison # If the KNN algorithm had the same amount of objects it would equal 0.6907499999313131 seconds

