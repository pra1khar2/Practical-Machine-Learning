setwd("G:\\Learn\\Coursera\\Data Science\\08_PracticalMachineLearning\\homework\\hw1")
library(caret)
library(e1071)
set.seed(123123)

training <- read.table("pml-training.csv", sep = ",", head = TRUE, na.strings = c("NA", ""))
testing <- read.table("pml-testing.csv", sep = ",", head = TRUE, na.strings = c("NA", ""))

training <- training[, (colSums(is.na(training)) == 0)]
testing <- testing[, (colSums(is.na(testing)) == 0)]

names(training)

training <- training[,-1:-7]
testing <- testing[,-1:-7]

inTrain = createDataPartition(training$classe, p = 0.7, list = FALSE)
training = training[inTrain, ]
validation = training[-inTrain, ]

# pre
preProc <- preProcess(training[,-53], method = "pca", thresh = 0.99)
trainPC <- predict(preProc, training[,-53])
validationPC <- predict(preProc, validation[,-53])

#svm
fit <- svm(training$classe ~ ., data = trainPC)
pred <- predict(fit, validationPC)

confusionMatrix(pred, validation$classe)

# test
testPC <- predict(preProc, testing[,-53])
myPred <- predict(fit, testPC)
myPred
