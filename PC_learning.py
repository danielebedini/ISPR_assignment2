import pandas as pd

from pgmpy.models import BayesianNetwork
from pgmpy.estimators.PC import PC

# import from dataset
from try_plot import X, y, X_train, X_test, y_train, y_test
from scores import compute_structure_score, compute_correlation_score

SIG_LVL = 0.05

# create model
model = BayesianNetwork()

# PC algorithm
# It is a constraint-based structure learning algorithm that seeks to find the network structure that
# identifies (conditional) dependencies in data set using statistical independence tests and 
# estimates a DAG pattern that satisfies the identified dependencies. 
# The DAG pattern can then be completed to a faithful DAG, if possible.

# training data is X_train + y_train
# convert training data to pandas dataframe

tr_data = pd.concat([X_train, y_train], axis=1)
tr_unencoded = pd.concat([X, y], axis=1)
#pd_data = pd.DataFrame(tr_data)

pc = PC(tr_unencoded)
dag = pc.estimate(return_type='dag', significance_level=SIG_LVL)
model.add_edges_from(dag.edges())

# fit model
model.fit(tr_unencoded)

#model.predict(X_test)

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
score = compute_structure_score(model, test_set)

print("Correlation Scores: ")
score = compute_correlation_score(model, test_set)
'''