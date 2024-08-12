Credit Card Fraud Detection Using Machine Learning

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used for this project is highly imbalanced, with a very small percentage of transactions being fraudulent. The project involves data preprocessing, feature engineering, model training, and evaluation of the model's performance.

Usage:
-> Data Preprocessing: Load the dataset, handle missing values, and divide the dataset into features (X) and labels (Y).
-> Model Training: Split the data into training and testing sets, and train a Random Forest classifier.
-> Model Evaluation: Evaluate the trained model using various metrics.

Evaluation Metrics
The following metrics were used to evaluate the model's performance:

1) Accuracy: Measures the overall correctness of the model.
2) Precision: Measures the accuracy of the positive predictions (fraud).
3) Recall: Measures the model's ability to capture actual fraud cases.
4) F1-Score: The harmonic mean of precision and recall.
5) Matthews Correlation Coefficient (MCC): A balanced measure even if the classes are of very different sizes.
6) Confusion Matrix: To visualize the true positives, false positives, true negatives, and false negatives.

