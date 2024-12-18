“A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.” To find that logic is called “machine learning”.\
**Tom  Mitchell**

##### Types of Machine learning Algorithms
1- *Supervised Learning*: Input data(training data) and has a  label or result . ex: Spam/not-spam. Classification, Regression,Ensemble Methods,Neural Networks,Probabilistic Models, Hyperparameter Optimization, Evaluation and Metrics, Feature Engineering, Regularization, Transfer Learning

2- *Unsupervised Learning*: Input data is not labeled and does not have a known result. ex: Grouping students  by marks.Key techniques in unsupervised learning:Clustering,Dimensionality Reduction, Density Estimation,Association Rule Learning,Matrix Factorization,Generative Models,Manifold Learning,Self-Organizing Maps (SOMs)

3- *Semi-Supervised Learning*: (Speech Recognition) Input data is a mixture of labeled and unlabeled examples. ex: a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.

4- *Reinforcement Learning*: A goal-oriented learning based on interaction with environment. Autonomous cars.Robotics, Games, Mouse Neurons learn to play Pong doom, incubator.

  
###### Supervised Machine Learning
- Regression: Linear Regression, Logistic Regression
- Instance-based Algorithms: k-Nearest Neighbor (KNN)
- Decision Tree Algorithms: CART
- Bayesian Algorithms: Naive Bayes
- Ensemble Algorithms: eXtreme Gradient Boosting
- Deep Learning Algorithms: Convolution Neural Network

###### Classification vs Regression
- Classification predicting a label 
- Regression predicting a quantity.

###### Classification Algorithms Examples: Predicts a discrete label or category.
- Linear: Linear Regression, Logistic Regression
- Nonlinear: Trees, k-Nearest Neighbors
- Ensemble:
- Bagging: Random Forest
- Boosting: AdaBoost
- Decision Trees: Rule-based model for splitting data into classes.
- Random Forest: An ensemble of decision trees to improve accuracy and reduce overfitting.
- Support Vector Machines (SVM): Finds the hyperplane that best separates classes.
- Naive Bayes: Probabilistic model based on Bayes' theorem.
- K-Nearest Neighbors (KNN): Classifies based on the majority class of nearest neighbors.
- Neural Networks: Deep learning models for complex classification tasks.

 ###### Regression Predicts a continuous output.
Common Algorithms:
- Linear Regression: Models a linear relationship between features and the target.
- Ridge and Lasso Regression: Add regularization to linear regression to prevent overfitting.
- Polynomial Regression: Models nonlinear relationships using polynomial features.
- Decision Trees and Random Forest (Regression variants).
- Support Vector Regression (SVR): Extension of SVM for regression.
- Neural Networks: Handle complex relationships for continuous outputs.

 ###### Ensemble Methods
Combine multiple models to improve accuracy and robustness.
Key Techniques:
- Bagging: Trains multiple models on different subsets of data.
- Example: Random Forest.
- Boosting: Sequentially trains models, focusing on errors of the previous ones.
- Example: Gradient Boosting, XGBoost, LightGBM, AdaBoost.
- Stacking: Combines predictions from multiple models using a meta-model.
- 
###### Neural Networks
- Feedforward Neural Networks (FNNs): Used for both classification and regression.
- Convolutional Neural Networks (CNNs): Specialized for image data.
- Recurrent Neural Networks (RNNs): Handle sequential data like time series or text.
- Transformer Models: Advanced architecture for NLP and time series (e.g., BERT, GPT).

###### Probabilistic Models
Predict outcomes based on probability distributions.
Key Techniques:
- Bayesian Linear Regression.
- Hidden Markov Models (HMMs): For time-series data.
- Gaussian Processes: For regression with uncertainty estimation.

###### Hyperparameter Optimization
Techniques to fine-tune model parameters.
- Common Methods:
- Grid Search: Tries all combinations of parameters.
- Random Search: Samples random combinations of parameters.
- Bayesian Optimization: Uses probabilistic modeling for efficient search.

###### Evaluation and Metrics
Techniques to evaluate model performance.
For Classification:

- Accuracy, Precision, Recall, F1-Score, ROC-AUC.
For Regression:

- Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared.

###### Feature Engineering
- Feature Selection: Identifies important features (e.g., Recursive Feature Elimination, Lasso).
- Feature Transformation: Transforms features for better performance (e.g., scaling, encoding, PCA).
- Feature Extraction: Derives new features (e.g., embedding layers in neural networks).


###### Regularization
- Techniques to prevent overfitting:
- L1 Regularization (Lasso): Encourages sparsity in features.
- L2 Regularization (Ridge): Penalizes large coefficients.
- Dropout: Randomly drops neurons in neural networks.


######  Transfer Learning
Leverages pre-trained models to solve similar tasks with less data.
Common in deep learning for NLP and image recognition tasks.

###### -----------------------------------------------------------------------------------------------------

###### Clustering
Groups data points into clusters based on similarity.
Key algorithms:
- K-Means: Partition-based clustering.
- Hierarchical Clustering: Builds a hierarchy of clusters.
- DBSCAN: Density-based clustering, handles noise well.
- Gaussian Mixture Models (GMMs): Uses probabilistic modeling.
- Mean-Shift: Identifies high-density regions.

###### Dimensionality Reduction
Reduces the number of features while preserving important information.
Useful for visualization, noise reduction, and speeding up computations.
Key methods:
- Principal Component Analysis (PCA): Projects data onto principal components that explain the most variance.
- t-SNE (t-Distributed Stochastic Neighbor Embedding): Visualizes high-dimensional data in 2D/3D.
- UMAP (Uniform Manifold Approximation and Projection): Similar to t-SNE but faster and scalable.
- Autoencoders: Neural networks for encoding and reconstructing data.

###### Density Estimation
Estimates the probability distribution of data.
Applications include anomaly detection and data generation.
Key techniques:
- Kernel Density Estimation (KDE): Non-parametric density estimation.
- Gaussian Mixture Models (GMMs): Models data as a mixture of Gaussian distributions.

###### Association Rule Learning
Discovers relationships or patterns among features in data.
Commonly used in market basket analysis.
Key algorithms:
- Apriori: Identifies frequent itemsets and generates association rules.
- Eclat: Uses a vertical data format for frequent itemset mining.
- FP-Growth (Frequent Pattern Growth): Scales better for large datasets.

###### Matrix Factorization
Decomposes a matrix into lower-dimensional representations.
Applications: recommendation systems, image compression.
Key methods:
- Singular Value Decomposition (SVD).
- Non-Negative Matrix Factorization (NMF).


###### Generative Models
Learns to generate new data similar to the input data.
Key techniques:
- GANs (Generative Adversarial Networks): Uses two networks (generator and discriminator) to generate realistic data.
- Variational Autoencoders (VAEs): Probabilistic approach to data generation.



###### Anomaly Detection
Identifies data points that deviate significantly from the norm.
Key methods:
- Isolation Forests: Splits data recursively to isolate anomalies.
- One-Class SVM: Separates normal data from outliers.
- Density-Based Methods: Detect anomalies in low-density regions.


###### Manifold Learning
Captures complex, nonlinear structures in data.
Useful for visualization and dimensionality reduction.
Key techniques:
- ISOMAP: Preserves geodesic distances on a manifold.
- LLE (Locally Linear Embedding): Preserves local neighborhood structure.


###### Self-Organizing Maps (SOMs)
A type of neural network that maps high-dimensional data into a low-dimensional grid.
Commonly used for visualization and clustering.


###### Biclustering
Simultaneously clusters rows and columns of a data matrix.
Useful in bioinformatics (e.g., gene-expression analysis).



###### Data Visualization methods 
- Data Selection
- Feature Selection methods (Pearson Correlation. This is a filter-based method.Chi-Squared. This is another filter-based method.Recursive Feature Elimination. This is a wrapper based method.Lasso: Select From Model.
Tree-based: Select From Model. This is an Embedded method.)
- Feature Engineering methods ..(Create new feature)
- Data Transormation methods ..(Binning of Data)
- Spot Check Algorithm




###### The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting
- Correcting abnormal values and outliers  (age = 800 instead of 80)
- Completing missing information  (null) use  mean, median
- Creating new features for analysis  ( use existing features to create new features)
- Converting fields to the correct format for calculations 
 
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
 
