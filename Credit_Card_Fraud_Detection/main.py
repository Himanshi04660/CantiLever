# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Load the dataset from the csv file using pandas
data = pd.read_csv("/content/creditcard.csv")

# Grab a peek at the data
data.head()

# Print the shape of the data
# data = data.sample(frac = 0.1, random_state = 48)
print(data.shape)
print(data.describe())

# Determine number of fraud cases in dataset
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

print("Amount details of the fraudulent transaction")
fraud.Amount.describe()

print("details of valid transaction")
valid.Amount.describe()

# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

# dividing the X and the Y from the dataset
X = data.drop(['Class'], axis = 1)
Y = data["Class"]
print(X.shape)
print(Y.shape)
# getting just the values for the sake of processing 
# (its a numpy array with no columns)
xData = X.values
yData = Y.values

# Using Scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(
        xData, yData, test_size = 0.2, random_state = 42)

# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
xTrain = imputer.fit_transform(xTrain)
xTest = imputer.transform(xTest)

# Random forest model creation and training
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Making predictions
yPred = rfc.predict(xTest)

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Remove NaN values from yTest and the corresponding predictions in yPred
mask = ~np.isnan(yTest)
yTest_clean = yTest[mask]
yPred_clean = yPred[mask]

# Evaluating the classifier
acc = accuracy_score(yTest_clean, yPred_clean)
print("The accuracy is {}".format(acc))

# Assuming binary classification; if multi-class, specify 'average' parameter
prec = precision_score(yTest_clean, yPred_clean, average='binary')
print("The precision is {}".format(prec))

rec = recall_score(yTest_clean, yPred_clean, average='binary')
print("The recall is {}".format(rec))

f1 = f1_score(yTest_clean, yPred_clean, average='binary')
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(yTest_clean, yPred_clean)
print("The Matthews correlation coefficient is {}".format(MCC))

# print the confusion matrix
print("Confusion Matrix:\n", confusion_matrix(yTest_clean, yPred_clean))
