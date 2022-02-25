“A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.” To find that logic is called “machine learning”.\
**Tom  Mitchell**

##### Types of Machine learning Algorithms
1- *Supervised Learning*: Input data(training data) and has a  label or result . ex: Spam/not-spam  (stock price)\
2- *Unsupervised Learning*: Input data is not labeled and does not have a known result. ex: Grouping students  by marks\
3- *Semi-Supervised Learning*: Input data is a mixture of labeled and unlabeled examples. ex: a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.\
4- *Reinforcement Learning*: A goal-oriented learning based on interaction with environment. Autonomous cars.

  
###### Supervised Machine Learning
- Regression: Linear Regression, Logistic Regression
- Instance-based Algorithms: k-Nearest Neighbor (KNN)
- Decision Tree Algorithms: CART
- Bayesian Algorithms: Naive Bayes
- Ensemble Algorithms: eXtreme Gradient Boosting
- Deep Learning Algorithms: Convolution Neural Network

###### Classification vs Regression
Classification predicting a label 
Regression predicting a quantity.

###### Classification Algorithms Examples:
Linear: Linear Regression, Logistic Regression
Nonlinear: Trees, k-Nearest Neighbors
Ensemble:
Bagging: Random Forest
Boosting: AdaBoost
Machine Learning Pipeline:
Define Problem


###### Data Visualization methods 
Data Selection
Feature Selection methods ..
Feature Engineering methods ..
Data Transormation methods ..
Spot Check Algorithm




###### The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting
- Correcting abnormal values and outliers  (age = 800 instead of 80)
- Completing missing information  (null) use  mean, median
- Creating new features for analysis  ( use existing features to create new features)
- Converting fields to the correct format for calculations 
 
 <p></p>
 
###### Sample Diabetes ML  
_Objective_: Decide whether someone has diabetes or not.
Pregnancies :- Number of times a woman has been pregnant
Glucose :- Plasma Glucose concentration of 2 hours in an oral glucose tolerance test
BloodPressure :- Diastollic Blood Pressure (mm hg)
SkinThickness :- Triceps skin fold thickness(mm)
Insulin :- 2 hour serum insulin(mu U/ml)
BMI :- Body Mass Index ((weight in kg/height in m)^2)
Age :- Age(years)
DiabetesPedigreeFunction :-scores likelihood of diabetes based on family history)
Result :- 0(doesn't have diabetes) or 1 (has diabetes) (Dependent value while others are independent)

The result variable value is either 1 or 0 indicating whether a person has diabetes(1) or not(0).


- Loading Dataset
- Information About Dataset columns and Summary
- Cleaning Data Eliminate Null values and zeros, Drop dublicates
- Data visualize  to see if data is balanced understand the reltionship between variables 
- Feature Selection
- Handling Outliers
- Split the Data Frame into X and y
- TRAIN TEST SPLIT
- Build the Classification Algorithm
- KNearestNeighbors
- Logistic Regression
- Naive Bayes
- DECISION TREE CLASSIFIER
- RandomForestClassifier
- AdaBoostClassifier
- Gradient Boosting Classifier
- XGBClassifier
- ExtraTreesClassifier

- Hyper Parameter Tuning using GridSearch CV
- Fit Best Model
- Predict on testing data using that model
- Performance Metrics :- Confusion Matrix, F1 Score, Precision Score, Recall Score


###### Outliers
Data Entry Errors,Data Entry Errors,Measurement Error,Natural Outlier. 
- Using Z score method (Find out how many standard deviations value away from the mean)
- ROBUST Z-SCORE (Called as Median absolute deviation method. It is similar to Z-score method with some changes in parameters)

 ###### Feature Scaling
  It is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing to handle highly varying magnitudes or values or units. If feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the values.
###### Standard Scaler:
 It is a very effective technique which re-scales a feature value so that it has distribution with 0 mean value and variance equals to 1.
Differences in the scales across input variables may increase the difficulty of the problem being modeled. An example of this is that large input values (e.g. a spread of hundreds or thousands of units) can result in a model that learns large weight values. A model with large weight values is often unstable, meaning that it may suffer from poor performance during learning and sensitivity to input values resulting in higher generalization error.
###### Quantile Transformer 
 Now we will use Quantile Transformer  This method transforms the features to follow a uniform or a normal distribution.
outliers are still present in this dataset but their impact has been reduced and we will check it byboxplot again


###### Tune hyperparameters 
After Grid Search, we got best parameters for all the models. Now tune hyperparameters see how to it performs.

- True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.

- True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.

- False Positives (FP) – When actual class is no and predicted class is yes.

- False Negatives (FN) – When actual class is yes but predicted class in no.

- Accuracy - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.

           Accuracy = TP+TN/TP+FP+FN+TN
- Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.

           Precision = TP/TP+FP
- Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.

           Recall = TP/TP+FN
- F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.

           F1 Score = 2(Recall Precision) / (Recall + Precision)
- Support - Support is the number of actual occurrences of the class in the specified dataset. Support doesn’t change between models but instead diagnoses the evaluation process.
 
