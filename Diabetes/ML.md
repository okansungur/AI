##### Types of Machine learning Algorithms
1- *Supervised Learning*: Input data(training data) and has a  label or result . ex: Spam/not-spam. Classification, Regression,Ensemble Methods,Neural Networks,Probabilistic Models, Hyperparameter Optimization, Evaluation and Metrics, Feature Engineering, Regularization, Transfer Learning

2- *Unsupervised Learning*: Input data is not labeled and does not have a known result. ex: Grouping students  by marks.Key techniques in unsupervised learning:Clustering,Dimensionality Reduction, Density Estimation,Association Rule Learning,Matrix Factorization,Generative Models,Manifold Learning,Self-Organizing Maps (SOMs)

3- *Semi-Supervised Learning*: (Speech Recognition) Input data is a mixture of labeled and unlabeled examples. ex: a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.

4- *Reinforcement Learning*: A goal-oriented learning based on interaction with environment. Self-driving cars, Robotics, Games,Healtcare, Mouse Neurons learn to play Pong doom, incubator.

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


###### Overfitting
Definition: Overfitting occurs when a model learns not only the underlying patterns in the training data but also the noise and random fluctuations, leading to poor performance on unseen data.

__Symptoms of Overfitting:__

-Very high accuracy on training data but low accuracy on validation/test data.
- Large gap between training and validation loss.
  
__Causes of Overfitting:__

- Model Complexity: The model has too many parameters or is too flexible (e.g., deep neural networks with too many layers).
- Insufficient Training Data: The model memorizes the training set instead of learning general patterns.
- Noise in the Data: The model tries to fit irrelevant details or outliers.
  
__How to Prevent Overfitting:__

- Reduce Model Complexity:
- Use simpler models (e.g., fewer layers or smaller decision trees).
- Regularization:
- L1 Regularization: Adds a penalty proportional to the absolute values of weights (encourages sparsity).
- L2 Regularization: Adds a penalty proportional to the square of weights (encourages smaller weights).
- 
__Increase Training Data:__

- Collect more samples or use data augmentation techniques.
  
 __Dropout:__
 
- Randomly "drop" units in a neural network during training to reduce reliance on specific neurons.
  
__Early Stopping:__

- Monitor the model's performance on validation data and stop training when performance stops improving.
  
__Cross-Validation:__

- Use techniques like k-fold cross-validation to ensure robust evaluation.
  
__Prune Decision Trees:__

Limit tree depth or prune unnecessary branches.





###### Underfitting
Definition: Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data.

__Symptoms of Underfitting:__

- Low accuracy on both training and validation/test data.
- Training and validation loss do not decrease significantly.
  
__Causes of Underfitting:__
- Insufficient Model Complexity:
The model lacks the capacity to represent the data (e.g., a linear model used for nonlinear data).
- Insufficient Training:
The model has not been trained for enough epochs or iterations.
- Poor Feature Representation:
Important features are missing or incorrectly engineered.

__How to Address Underfitting:__

- Increase Model Complexity:Use more complex models (e.g., deeper networks, more layers, or higher polynomial degrees).
- Train for More Epochs:Allow the model to learn longer by training for more iterations.
- Improve Feature Engineering:  Add more relevant features or use techniques like feature transformation.
- Reduce Regularization:If regularization is too strong, it may constrain the model excessively.
- Hyperparameter Tuning: Optimize parameters like learning rate, number of layers, or units in a neural network.




###### The 4 C's of Data Cleaning: Correcting, Completing, Creating, and Converting
- Correcting abnormal values and outliers  (age = 800 instead of 80)
- Completing missing information  (null) use  mean, median
- Creating new features for analysis  ( use existing features to create new features)
- Converting fields to the correct format for cal
