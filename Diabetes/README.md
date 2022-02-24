“A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.” To find that logic is called “machine learning”.

** Tom Mitchell **

- Types of Machine learning Algorithms
1- Supervised Learning: Input data(training data) and has a  label . Ex: Spam/not-spam  (stock price)
2- Unsupervised Learning: Input data is not labeled and does not have a known result. EX: Grouping customers by purchasing behavior
3- Semi-Supervised Learning: Input data is a mixture of labeled and unlabeled examples. EX: a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.

4- Reinforcement Learning: a goal-oriented learning based on interaction with environment. Autonomous cars.

-   
Supervised Machine Learning
Regression: Linear Regression, Logistic Regression

Instance-based Algorithms: k-Nearest Neighbor (KNN)

Decision Tree Algorithms: CART

Bayesian Algorithms: Naive Bayes

Ensemble Algorithms: eXtreme Gradient Boosting

Deep Learning Algorithms: Convolution Neural Network

Classification vs Regression
Classification predicting a label .vs. Regression predicting a quantity.

Classification Algorithms Examples:
Linear: Linear Regression, Logistic Regression
Nonlinear: Trees, k-Nearest Neighbors
Ensemble:
Bagging: Random Forest
Boosting: AdaBoost
Machine Learning Pipeline:
Define Problem

ML type of problem
Prepare Data

Data Visualization methos ...
Data Selection
Feature Selection methods ..
Feature Engineering methods ..
Data Transormation methods ..
Spot Check Algorithm

Test Harness ...
Perform Measure ...
Evaluate accuracy of different algorithms
Improve Results

Algorithms Turning methids
ensemble methods
Present Results

Save the model



The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting¶
In this stage, we will clean our data by 1) correcting aberrant values and outliers, 2) completing missing information, 3) creating new features for analysis, and 4) converting fields to the correct format for calculations and presentation.

Correcting: Reviewing the data, there does not appear to be any aberrant or non-acceptable data inputs. In addition, we see we may have potential outliers in age and fare. However, since they are reasonable values, we will wait until after we complete our exploratory analysis to determine if we should include or exclude from the dataset. It should be noted, that if they were unreasonable values, for example age = 800 instead of 80, then it's probably a safe decision to fix now. However, we want to use caution when we modify data from its original value, because it may be necessary to create an accurate model.
Completing: There are null values or missing data in the age, cabin, and embarked field. Missing values can be bad, because some algorithms don't know how-to handle null values and will fail. While others, like decision trees, can handle null values. Thus, it's important to fix before we start modeling, because we will compare and contrast several models. There are two common methods, either delete the record or populate the missing value using a reasonable input. It is not recommended to delete the record, especially a large percentage of records, unless it truly represents an incomplete record. Instead, it's best to impute missing values. A basic methodology for qualitative data is impute using mode. A basic methodology for quantitative data is impute using mean, median, or mean + randomized standard deviation. An intermediate methodology is to use the basic methodology based on specific criteria; like the average age by class or embark port by fare and SES. There are more complex methodologies, however before deploying, it should be compared to the base model to determine if complexity truly adds value. For this dataset, age will be imputed with the median, the cabin attribute will be dropped, and embark will be imputed with mode. Subsequent model iterations may modify this decision to determine if it improves the model’s accuracy.
Creating: Feature engineering is when we use existing features to create new features to determine if they provide new signals to predict our outcome. For this dataset, we will create a title feature to determine if it played a role in survival.
Converting: Last, but certainly not least, we'll deal with formatting. There are no date or currency formats, but datatype formats. Our categorical data imported as objects, which makes it difficult for mathematical calculations. For this dataset, we will conv

