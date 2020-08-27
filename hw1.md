Human Activity Recognition
========================================================

### Shenda Hong

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. 

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, we use the dataset from the reference, and my goal will be:

1. Built the model to predict the manner in which they did the exercise.

2. Using cross validation and choose the best model.

More information is available from the website here: 

http://groupware.les.inf.puc-rio.br/har

## Preliminary

First of all, Let's change the workspace to proper directory and library machine learning packages. An important thing is to set a constant seed for reproducible research.


```r
setwd("G:\\Learn\\Coursera\\Data Science\\08_PracticalMachineLearning\\homework\\hw1")
library(caret)
library(e1071)
library(corrgram)
library(ggplot2)
library(psych)
set.seed(123123)
```

## Data preprocessing

### Loading

loading the training and testing datasets, assigning missing values to entries that are currently 'NA' or blank.


```r
training <- read.table("pml-training.csv", sep = ",", head = TRUE, na.strings = c("NA", ""))
testing <- read.table("pml-testing.csv", sep = ",", head = TRUE, na.strings = c("NA", ""))
```

### Dealing with missing data

Removing variables that have too much missing value.


```r
training <- training[, (colSums(is.na(training)) == 0)]
testing <- testing[, (colSums(is.na(testing)) == 0)]
```

### Selecting variables

Removing the first seven variables that unrelative with analysis.


```r
names(training)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

```r
training <- training[,-1:-7]
testing <- testing[,-1:-7]
```

### Data slicing

We now slice the training dataset into a training dataset (70% of the observations) and a validation dataset (30% of the observations). This validation dataset will allow us to perform cross validation when choosing our model.

inTrain is a random index list for slicing.


```r
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
validation <- training[-inTrain, ]
training <- training[inTrain, ]
```

## Reducing dimension

Now, after removing 107 variables, our training set contains 53 variables. It's easier to build a prediction model. After a brief look at intercorrelations figure, we found out that many variables have high correlation. Which means that we can using PCA for dimention reducing.

We display the scree test based on the observed eigenvalues (as straight-line segments and x’s), the mean eigenvalues derived from 100 random data matrices (as dashed lines), and the eigenvalues greater than 1 criteria (as a horizontal line at y=1).

Parallel analysis suggests that the number of components =  13.


```r
fa.parallel(training[,-53], fa="PC", main="Scree plot with parallel analysis")
```

```
## Loading required package: parallel
## Loading required package: MASS
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6.png) 

```
## Parallel analysis suggests that the number of factors =  13  and the number of components =  13
```

```r
preProc <- preProcess(training[,-53], method = "pca", pcaComp = 13)
trainPC <- predict(preProc, training[,-53])
validationPC <- predict(preProc, validation[,-53])
```

## Learning a model

### Model 1: Support Vector Machine (SVM)

The first model is support vector machine, which is among the best (and many believe is indeed the best) "off-the-shelf" supervised learning algorithm. 


```r
start <- Sys.time()
fit1 <- svm(training$classe ~ ., data = trainPC)
pred1 <- predict(fit1, validationPC)
end <- Sys.time()

time1 <- end - start
time1
```

```
## Time difference of 35.57 secs
```

```r
res1 <- confusionMatrix(pred1, validation$classe)
res1$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1572   92   65   30   19
##          B   16  959   31    7   24
##          C   51   77  894  103   54
##          D   28    6   23  822   37
##          E    7    5   13    2  948
```

```r
acc1 <- res1$overall["Accuracy"]
acc1 <- as.numeric(acc1)
acc1
```

```
## [1] 0.8828
```

```r
out_of_sample_error1 <- 1 - acc1
out_of_sample_error1
```

```
## [1] 0.1172
```

The estimated accuracy of svm is 88.2% and the estimated out-of-sample error based on svm applied to the cross validation dataset is 11.8%.

### Model 2: Linear Discriminant Analysis (LDA)

Our second model is Linear Discriminant Analysis, which is to find a linear combination of features which characterizes or separates two or more classes of objects or events. 


```r
start <- Sys.time()
fit2 <- train(training$classe ~ ., method = "lda", data = trainPC)
pred2 <- predict(fit2, validationPC)
end <- Sys.time()

time2 <- end - start
time2
```

```
## Time difference of 4.811 secs
```

```r
res2 <- confusionMatrix(pred2, validation$classe)
res2$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1142  304  425  111  158
##          B  112  398   91  204  189
##          C  151  161  351  117  158
##          D  210  137   82  423   98
##          E   59  139   77  109  479
```

```r
acc2 <- res2$overall["Accuracy"]
acc2 <- as.numeric(acc2)
acc2
```

```
## [1] 0.4746
```

```r
out_of_sample_error2 <- 1 - acc2
out_of_sample_error2
```

```
## [1] 0.5254
```

The estimated accuracy of svm is 47.5% and the estimated out-of-sample error based on svm applied to the cross validation dataset is 52.5%.

### Model 3: Random Forests (RF)

Random forests are an ensemble learning method for classification (and regression) that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes output by individual trees.


```r
start <- Sys.time()
fit3 <- train(training$classe ~ ., method = "rf", data = trainPC,
              trControl = trainControl(method = "cv", number = 3))
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
pred3 <- predict(fit3, validationPC)
end <- Sys.time()

time3 <- end - start
time3
```

```
## Time difference of 2.76 mins
```

```r
res3 <- confusionMatrix(pred3, validation$classe)
res3$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1634   33    6   13    4
##          B    4 1063   12    6   10
##          C   26   39  987   46    8
##          D    6    1   16  897   13
##          E    4    3    5    2 1047
```

```r
acc3 <- res3$overall["Accuracy"]
acc3 <- as.numeric(acc3)
acc3
```

```
## [1] 0.9563
```

```r
out_of_sample_error3 <- 1 - acc3
out_of_sample_error3
```

```
## [1] 0.04367
```

The estimated accuracy of rf is 95.6% and the estimated out-of-sample error based on svm applied to the cross validation dataset is 4.4%.

### Model Comparasion

In this section, we compare our three model based on their accurate on validation set and time consuming.


```r
Accurate <- c(acc1, acc2, acc3)
model <- as.factor(c("SVM", "LDA", "RF"))
qplot(model, Accurate, geom = "bar", stat="identity", main = "Accuracy Comparasion")
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-101.png) 

```r
Time <- c(time1, time2, time3 * 60)
model <- as.factor(c("SVM", "LDA", "RF"))
qplot(model, Time, geom = "bar", stat="identity", main = "Time Comparasion")
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-102.png) 

The above figures tell us that:

1. Random Forests have the highest accuracy but the longest time consuming.

2. Linear Discriminant Analysis have the shortest time consuming but the lowest accuracy.

3. Support Vector Machine have pretty high accuracy with the kind of short time consuming.

## Prediction

According to the analysis above, we choose svm to be our prediction model. And adjust the pca thresh to 0.99 to improve the accuracy while spending more time for a balance.


```r
preProc <- preProcess(training[,-53], method = "pca", thresh = 0.99)
trainPC <- predict(preProc, training[,-53])
validationPC <- predict(preProc, validation[,-53])
start <- Sys.time()
fitBest <- svm(training$classe ~ ., data = trainPC)
pred <- predict(fitBest, validationPC)
end <- Sys.time()
time <- end - start
res <- confusionMatrix(pred, validation$classe)
res$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1669   51    0    1    0
##          B    1 1071   17    0    2
##          C    1   15  997   72   13
##          D    0    1    7  888   17
##          E    3    1    5    3 1050
```

```r
acc <- res$overall["Accuracy"]
acc <- as.numeric(acc)
out_of_sample_error <- 1 - acc
testPC <- predict(preProc, testing[,-53])
myPred <- predict(fitBest, testPC)
```

The time consuming of the best model is:


```
## Time difference of 58.27 secs
```

The accuracy is:


```
## [1] 0.9643
```

The out of sample error is:


```
## [1] 0.03568
```

Finally, my prediction on test set is:


```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

## Reference

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
