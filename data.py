# dataset stuff
from ucimlrepo import fetch_ucirepo 
import numpy as np
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 

print(X.head())
print(y.head())
# data but X and y are not separated
# whole_data = X.join(y)

# data but they are divided in training and testing sets

# one hot encoding
from sklearn.preprocessing import OneHotEncoder

#encoded_whole_data = OneHotEncoder().fit_transform(whole_data).toarray()
encoded_data = OneHotEncoder().fit_transform(X).toarray()
encoded_targets = OneHotEncoder().fit_transform(y).toarray()

#print(encoded_data)
#print(encoded_targets)
#print(encoded_whole_data)
  
# metadata 
#print(car_evaluation.metadata) 
  
# variable information 
#print(car_evaluation.variables) 
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(encoded_data, encoded_targets, test_size=0.2, random_state=42)

# TR set
whole_TR = np.append(X_train, y_train, axis=1)

# TS set
whole_TS = np.append(X_test, y_test, axis=1)

#print(y_test)
#print("*************************")
#print(X_test)

# label encoding
from sklearn.preprocessing import LabelEncoder

#label_encoded_whole_data = LabelEncoder().fit_transform(whole_data)
label_encoded_data = LabelEncoder().fit_transform(X)
label_encoded_targets = LabelEncoder().fit_transform(y)

print("Label Encoded Data:")
print(label_encoded_data)
print(label_encoded_targets)


