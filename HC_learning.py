import pandas as pd

from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator

from sklearn.metrics import accuracy_score

# import from dataset, also one hot encoded
from data2 import X_train, X_test, encoded_train, encoded_test
from scores import compute_structure_score, compute_correlation_score

# create model
model = BayesianNetwork()

# estimate model
pd_data = pd.DataFrame(encoded_train)
hc = HillClimbSearch(pd.DataFrame(pd_data))
best_model = hc.estimate()

# add edges
model.add_edges_from(best_model.edges())

# fit model
model.fit(pd_data, estimator=BayesianEstimator)


# print model
print("****************************************************************************************")
print(model)
print("****************************************************************************************")
print(model.nodes())
print(model.edges())

# accuracy
#pd_ts_data = pd.DataFrame(encoded_TS_data)
#y_pred = model.predict(pd.DataFrame(encoded_TS_features))

#print("Accuracy: ", accuracy_score(pd_ts_data, y_pred))

'''
score = compute_BDeuScore(model, encoded_TS_data)
print("BDeu Score: ", score)
print("****************************************************************************************")
score = compute_BicScore(model, encoded_TS_data)
print("Bic Score: ", score)
print("****************************************************************************************")
#score = compute_log_likelihood(model, X_test) # TODO: fix this
#print("Log likelihood Score: ", score)
'''

'''
print("****************************************************************************************")
print("Structure scores: ")
compute_structure_score(model, encoded_TS_data)

# correlation score
print("****************************************************************************************")
print("Correlation scores: ")
compute_correlation_score(model, encoded_TS_data)
'''

'''
# inference # TODO: check this
mle = MaximumLikelihoodEstimator(model, pd.DataFrame(encoded_TR_data))
cpds = mle.estimate_cpd()
model.add_cpds(cpds)
model.check_model()

# predict
model_inference = BayesianModelInference(model)
print("****************************************************************************************")
print("Predicting: ")
predicted = model_inference.predict(encoded_TS_data)
print(predicted)
print("****************************************************************************************")
'''
