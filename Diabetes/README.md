“A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.” To find that logic is called “machine learning”.\
**Tom  Mitchell**
 
 <p></p>
 
###### Sample Diabetes ML  
**Diabetes**:Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into energy. It occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Diabetes can cause heart attack, heart failure, stroke, kidney failure and coma.
_Objective_: Decide whether someone has diabetes or not.
- Pregnancies : Number of times a woman has been pregnant
- Glucose : Plasma Glucose concentration of 2 hours in an oral glucose tolerance test
- BloodPressure : Diastollic Blood Pressure (mm hg)
- SkinThickness : Triceps skin fold thickness(mm)
- Insulin : 2 hour serum insulin(mu U/ml)
- BodyMassIndex : Body Mass Index ((weight in kg/height in m)^2)
- Age : (years)
- DiabeteFamilyHistory :scores of diabetes based on family history)
- Result : 0(doesn't have diabetes) or 1 (has diabetes) (Dependent value while others are independent)

The result variable value is either 1 or 0 indicating whether a person has diabetes(1) or not(0).


- 1-Import Libraries and Load Dataset
- 2- Information about the Dataset
- 3- Cleaning the Data (Eliminate Null values and zeros, Drop duplicates)
- 4- Data visualize  For a better understanding the reltionship between variables and data distribution
- 5- Feature Selection
- 6- Handling Outliers
- 7- Split the Data Frame into X and y
- 8- Train Test Split
- Build the Classification Algorithm
- KNearestNeighbors
- Logistic Regression
- Naive Bayes
- Decision Tree Classifier
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







###### Confusion Matrix
The confusion matrix is a matrix used to determine the performance of the classification models for a given set of test data. It can only be determined if the true values for test data are known. It evaluates the performance of the classification models, shows the errors made by the classifiers, calculates the different parameters for the model, such as accuracy, precision
The true positive rate, also known as sensitivity or recall in machine learning.

<p align="center">
  <img  src="https://github.com/okansungur/AI/blob/main/Diabetes/ConfMatrix.png"><br/>
  Confusion Matrix
</p>



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
 
