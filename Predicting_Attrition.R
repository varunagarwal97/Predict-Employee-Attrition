#Please run the below commands to install the libraries used.
# install.packages("knitr")
# install.packages("ggplot2")
# install.packages("mlr")
# install.packages("corrplot")

library(mlr)
library(knitr)
library(ggplot2)
library(corrplot)


set.seed(2281)

#Reading the data set
df = read.csv("HR.csv")

#Summary of the dataset
summary(df)

df_copy = df
  
#Changing the factor level for the Attrition variable
df$Attrition = factor(df$Attrition, levels = c("Yes", "No"))
df_copy$Attrition = factor(df_copy$Attrition, levels = c("Yes", "No"))

#Deleting variables which have constant value throughout the dataset
df$EmployeeCount = NULL
df$StandardHours = NULL
df$Over18 = NULL

##Exploratory Data Analysis
#Class distribution
ggplot(df, aes(x = Attrition, fill = Attrition)) + geom_bar() + ylab("Count") + ggtitle("Class distribution")


#Correlation plot for continuous variables
ind = sapply(df, is.factor) 
ind = which(ind == TRUE)
corr_mat <- cor(df[,-c(2, 3, 5, 8, 11, 15, 17, 21)])
corrplot(corr_mat)

#Density distribution of Attrition, grouped by Age
ggplot(df, aes(x = Age)) + geom_histogram(binwidth = 10, aes(y = ..density.., fill = Attrition), position = "dodge") + ggtitle("Class distribution density, grouped by Age bins")

#Density distribution of Attrition, grouped by Monthly Income
ggplot(df, aes(x = MonthlyIncome)) + geom_histogram(binwidth = 5000, aes(y = ..density.., fill = Attrition), position = "dodge") + ggtitle("Class distribution density, grouped by Monthly Income bins")



#Splitting Data into test and train
##splitIndex = createDataPartition(df_copy$Attrition, p = .70, list = FALSE, times =1 )

splitIndex = sample(1:nrow(df_copy), nrow(df_copy)*0.7, replace=FALSE)

trainSplit = df_copy[splitIndex, ]
testSplit = df_copy[-splitIndex, ]

prop.table(table(trainSplit$Attrition))

##Setting up training and test tasks
#Training task
attrition.task = makeClassifTask(data = trainSplit, target = "Attrition")
#Removing features with constant values
attrition.task = removeConstantFeatures(attrition.task)

#Testing task
test.task = makeClassifTask(data = testSplit, target = "Attrition")
test.task = removeConstantFeatures(test.task)


#Setting up costs manually
costs = matrix(c(0.6, 0.2, 0.9, 0), 2)
colnames(costs) = rownames(costs) = getTaskClassLevels(attrition.task)
costs


attrition.costs = makeCostMeasure(id = "attrition.costs", name = "Attrition costs", costs = costs,
                                  best = 0, worst = 0.9)
attrition.costs


###Thresholding without parameter training
## Train and predict posterior probabilities
lrn = makeLearner("classif.multinom", predict.type = "prob", trace = FALSE)
mod = train(lrn, attrition.task)
pred = predict(mod, task = attrition.task)
pred


#Performance on training set without using parameter training
performance(pred, measures = list(attrition.costs, f1))

performance(pred, measures = list(attrition.costs, mmce))

#Performance on test set without using parameter training
pred = predict(mod, task = test.task)

performance(pred, measures = list(attrition.costs, f1))

performance(pred, measures = list(attrition.costs, mmce))


###Thresholding using Parameter Training

## 3-fold cross-validation
lrn = makeLearner("classif.multinom", predict.type = "prob", trace = FALSE)
rin = makeResampleInstance("CV", iters = 5, task = attrition.task)
r = resample(lrn, attrition.task, resampling = rin, measures = list(attrition.costs, f1), show.info = FALSE)
r

#Tuning Threshold Parameters
tune.res = tuneThreshold(pred = r$pred, measure = attrition.costs)
tune.res

#Now that we have obtained our threshold values for Attrition being positive, let's train our model using this threshold

lrn = makeLearner("classif.multinom", predict.type = "prob", predict.threshold = tune.res$th, trace = FALSE)
mod = train(lrn, attrition.task)


#Performance on training set after using parameter training
pred = predict(mod, task = attrition.task)

performance(pred, measures = list(attrition.costs, f1))

performance(pred, measures = list(attrition.costs, mmce))


#Performance on test set after using parameter training
pred = predict(mod, task = test.task)

performance(pred, measures = list(attrition.costs, f1))

performance(pred, measures = list(attrition.costs, mmce))

calculateConfusionMatrix(pred, relative = FALSE)

#Generating graphs to test performance

d = generateThreshVsPerfData(pred, measures = list(fpr, fnr, mmce))

plotThreshVsPerf(d)

#Getting partial dependence of features. Estimating how learned function is affected by one or more features
att = getTaskData(attrition.task)
pd = generatePartialDependenceData(mod, att)
plt = plotPartialDependence(pd)
head(plt$data)

plt
