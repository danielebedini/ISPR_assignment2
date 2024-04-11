import pandas as pd

from pgmpy.models import BayesianNetwork
from pgmpy.estimators.PC import PC
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators.CITests import chi_square

from sklearn.metrics import accuracy_score

# import from dataset
from data import X_train, X_test, y_train, y_test, whole_TR, whole_TS
from scores import compute_structure_score

# create model
model = BayesianNetwork()

# PC algorithm
# It is a constraint-based structure learning algorithm that seeks to find the network structure that
# identifies (conditional) dependencies in data set using statistical independence tests and 
# estimates a DAG pattern that satisfies the identified dependencies. 
# The DAG pattern can then be completed to a faithful DAG, if possible.

# training data is X_train + y_train
# convert training data to pandas dataframe

pd_data = pd.DataFrame(whole_TR)

pc = PC(pd_data)
dag = pc.estimate(return_type='dag')
model.add_edges_from(dag.edges())

# fit model
model.fit(pd.DataFrame(X_train), estimator=BayesianEstimator)

# print model
#print("****************************************************************************************")
#print(X)
#print(y)
print("****************************************************************************************")
print(model)
print("****************************************************************************************")
print(dag.nodes())
print(dag.edges())

# Conditional Independence Tests for PC algorithm

# Chi square test
# It is a statistical test applied to the relationship between two categorical variables.
# The test is used to determine whether there is a significant association between the two variables.
# The test is based on the assumption that the variables are independent.
# If the test is significant (True), the variables are not independent. Otherwise (false), they are independent.

# variable names: buying, maint, doors, persons, lug_boot, safety

#pd_data = pd.DataFrame(X)
#test = chi_square('buying', 'maint',['doors', 'persons', 'lug_boot', 'safety'], pd_data, significance_level=0.05)

#print("****************************************************************************************")
#print(test)

# Score of the model, on the test set

'''
print("Structure Scores: ")
score = compute_structure_score(model, X_test)

# data without last column
# predict_data = predict_data.copy()
# predict_data.drop('E', axis=1, inplace=True)
# y_pred = model.predict(predict_data)
# y_pred

test_copy = pd.DataFrame(X_test.copy())
#test_copy.drop('class', axis=1, inplace=True)
y_pred = model.predict(test_copy)

# accuracy
accuracy = accuracy_score(y_test, y_pred) 
'''