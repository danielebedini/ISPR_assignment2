import pandas as pd
import numpy as np

# URL of the car evaluation dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

# Define column names based on dataset description
names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Load the dataset into a pandas DataFrame
data = pd.read_csv(url, names=names)

# Perform one-hot encoding on categorical features
data_encoded = pd.get_dummies(data, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

# Display the first few rows of the encoded dataset
print(data_encoded.head())

from sklearn.model_selection import train_test_split

# Extract features (X) and target variable (y)
X = data_encoded.drop(columns=['class'])
y = data_encoded['class']

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Display the first few rows of training and test sets
print("Training set:")
print(X_train)
print(y_train)
print("Test set:")
print(X_test)
print(y_test)

# Display the shapes of training and test sets
print("Training set shape - X:", X_train.shape, "y:", y_train.shape)
print("Test set shape - X:", X_test.shape, "y:", y_test.shape)

# Combine X and y to create whole training and test sets
encoded_train = np.append(X_train, y_train.values.reshape(-1, 1), axis=1)
encoded_test = np.append(X_test, y_test.values.reshape(-1, 1), axis=1)

print("Whole training set shape:", encoded_train.shape)
print("Whole test set shape:", encoded_test.shape)
