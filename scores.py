import pandas as pd
from pgmpy.estimators import BDeuScore, BicScore, K2Score, MaximumLikelihoodEstimator, ExpectationMaximization
from pgmpy.metrics import log_likelihood_score, structure_score, correlation_score
from pgmpy.sampling import BayesianModelInference



# Structure Score: Compute the score of the model
# The score is computed using the Bayesian Information Criterion (BIC) score.
# The BIC score is a measure of the goodness of fit of a model to a dataset.

# BDeu Score
# The BDeu score is a Bayesian Dirichlet equivalent uniform (BDeu) score.
# The BDeu score is a Bayesian score that is used to estimate the probability of a dataset given a model.

def compute_BDeuScore(model, encoded_data):

    pd_data = pd.DataFrame(encoded_data)
    bdeu = BDeuScore(pd_data, equivalent_sample_size=5)
    score = bdeu.score(model)

    return score

# Bic Score
# The BIC score is a Bayesian Information Criterion (BIC) score.
# The BIC score is a measure of the goodness of fit of a model to a dataset.

def compute_BicScore(model, encoded_data):

    pd_data = pd.DataFrame(encoded_data)
    bic = BicScore(pd_data)
    score = bic.score(model)

    return score


# K2 score

def compute_K2Score(model, encoded_data):
    
    pd_data = pd.DataFrame(encoded_data)
    k2 = K2Score(pd_data)
    score = k2.score(model)

    return score

# log likelihood score

def compute_log_likelihood(model, data): # TODO: check this function

    #pd_data = pd.DataFrame(encoded_data)

    mle = MaximumLikelihoodEstimator(model, data)
    model = mle.estimate_cpd('buying') 

    score = log_likelihood_score(model, data)
    return score

# Structure Score: Compute the score of the model following different scoring methods
def compute_structure_score(model, data):
    score_types = ['k2', 'bdeu', 'bic' , 'bds']
    pd_data = pd.DataFrame(data)

    for score_type in score_types:
        score = structure_score(model, pd_data, scoring_method=score_type)
        print(f"{score_type} score: {score}")

def compute_correlation_score(model, data):
    tests = ['chi_square', 'g_sq', 'log_likelihood', 'freeman_tuckey', 'modified_log_likelihood', 'neyman', 'cressie_read']
    #scores = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    pd_data = pd.DataFrame(data)
    for test in tests:
        score = correlation_score(model, pd_data, test=test, significance_level=0.05, return_summary=False)
        print(f"{test} score (accuracy): {score}")
   

