#Kaggle project

#We are going to predict weather an individual has survived or dead in titanic disaster

df.train <- read.csv("titanic_train.csv")

###Exploratory analysis of data

#install Amelia package. We use this to get few functions on missingness using Math and visualize it
library(Amelia)
missmap(df.train, main = "missing map", col = c("red", "black"))

library(ggplot2)
ggplot(data = df.train, aes(Pclass)) + geom_bar(aes(fill = factor(Pclass)))
ggplot(data = df.train, aes(Age)) + geom_histogram(bins = 20, fill = "red", color = "black", alpha = 0.6)

# from missmap function we can see that there are few blank values. We are missing 20% of
# age fields. We can sometimes take the average value if there is any missing data. 
# In this example we are imputing age based on class. 
# We build a function where we allote some random age as if lower class it is 29, middle 
# class 33. and so on. 

###function to impute missing age values based on class

age_replacement <- function(age, class){
  out <- age
  for(i in 1:length(age)){
    if(is.na(age[i])){
    if(class[i] == 1){
      out[i] <- 37
      }
      else if(class[i] == 2){
        out[i] <- 29
      }
      else{
        out[i] <- 24
      }
    }
    else{
      out[i] <- age[i]
    }
  }
  return(out)
}

#running the function to save it in a variable
fixed_ages <- age_replacement(df.train$Age, df.train$Pclass)
#Assigning the output of the function to train data
df.train$Age  <- fixed_ages

#checking if we still have any NA values for age
missmap(df.train, main = "Check after function", col = c("red", "black"))

#We have few varibales in the data frame that we don't use for building model.
# To get rid of them, we use dplyr library
library(dplyr)
#Now remove the variable we don't use from the data frame
df.train <- select(df.train, -PassengerId, -Name, -Ticket, -Cabin)
head(df.train)
str(df.train)

#To make the integer values as continuous for better performance of the model, we 
#convert integer values into continuous using factor function
df.train$Survived <- factor(df.train$Survived)
df.train$Pclass <- factor(df.train$Pclass)
df.train$SibSp <- factor(df.train$SibSp)
df.train$Parch <- factor(df.train$Parch)

####Begin logistic regression model
## Build model

log.model <- glm(formula = Survived ~ .,family =binomial(link = "logit"), data = df.train)
#find out what is family and link
log.model
summary(log.model)
####Use the same procedure for titanic test data as an example

#### Predict the survivors based on the model we have created.

library(caTools)
set.seed(101)
split <- sample.split(df.train$Survived, SplitRatio = 0.7)
final.train <- subset(df.train, split == TRUE)
final.test <- subset(df.train, split == FALSE)

final.log.model <- glm(formula = Survived ~ ., family = binomial(link = "logit"), 
                                                                 data = final.train)
summary(final.log.model)

fitted.probabilities <- predict(final.log.model, final.test, type = "response") 
#type is used as we needed the classification of model that is 0 or 1. 

fitted.result <- ifelse(fitted.probabilities>0.5, 1, 0)

# Misclassification error 
misclass_error <- mean(fitted.result != final.test$Survived)
accuracy <- 1-misclass_error
accuracy

#COnfusion Matrix

confusion_matrix <- table(final.test$Survived, fitted.result)
confusion_matrix

#misclassification error based on confusion matrix

false_positive <- confusion_matrix[1,2] #type 1 error
false_negative <- confusion_matrix[2,1] #type 2 error
misclassification <- (false_negative + false_positive) / length(final.test$Survived)
misclassification
new_accuracy <- 1-misclassification
new_accuracy

cat("The test error of the model is", misclassification)
cat("Accuracy of the model is", new_accuracy*100, "%")
