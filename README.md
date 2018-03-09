# Predict-Employee-Attrition
Predicting Employeee Attrition on imbalanced HR dataset


Overview

Attrition is defined as the gradual reduction of the volume of employee in an organization, and not replacing their positions. This could occur due to employees resigning or retiring, and is seldom associated with layoffs, too. For a company, attrition rate being high is unfavourable for reasons since it leads to loss of talent, poor ROI, and resources spent on training employees. Also, lesser volume of employees leads to low throughput and performance of the company. Hence, organizations strive to reduce the attrition rate. It is important to analyse the factors that influence the attrition in the company, and thereby, curb these factors and reduce attrition rate. 
Objective of the Study

Understand the influence of the various factors or predictor variables on the predicted variable, the degree of influence and the relation between different attributes.

Implement a classification model for the dataset with good prediction results.


Data

For this study, we have chosen the IBM HR Analytics Employee Attrition and Performance Attrition dataset, which has been obtained from Kaggle. It is a fictional dataset created by IBM data scientists, which contains data related to employee performance measures and attrition. The dataset contains several predictor variables, having varying influence on the predicted variable Attrition which signifies whether an employee left the company or not. 

A breakdown of the variables and their types is as follows:
	Predicted Variable: Attrition (Yes or No)
	Predictor Variables: 24 continuous variables, and 8 categorical variables
	Total instances: 1470 rows

One of the challenges this dataset puts forward is the imbalance in the class distribution. Out of the total 1470 tuples in the dataset, only 237 tuples have their predicted class as “Yes”.
Pre-processing

The dataset contains three columns, namely Employee Count, Over 18 and Standard Hours, which have the same values throughout the data. We remove these features from our dataset, since they do not provide any value to our prediction. Apart from this, the dataset is uniform throughout, and we have no missing values or nulls for any of the columns.

Conclusion

The model the we propose seems to be performing reasonably well, for the given number of data points. The dimensionality of the data certainly affects the performance, and can be improved by collecting more data. The misclassification rate is about 6% better than that of random guessing, which is reasonably well for unbalanced classes. We also check for F1 scores. 
