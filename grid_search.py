from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore

from try_plot import X, y
from PC_learning import model

# Define parameter grid for PC algorithm (e.g., significance levels)
param_grid = {
    'significance_level': [0.01, 0.05, 0.1]
}

# Initialize GridSearchCV-like logic
best_model = None
best_score = float('-inf')

for params in ParameterGrid(param_grid):
    #model.reset()
    # Train PC model with specified parameters
    pc = PC(X)
    estimated_model = pc.estimate(significance_level=params['significance_level'])
    model_gs = BayesianNetwork()
    model_gs.add_edges_from(estimated_model.edges())
    
    # Evaluate model using custom scorer
    bic_score = BicScore(X)
    score = bic_score.score(model_gs)
    
    # Update best model and score based on evaluation
    if score > best_score:
        best_score = score
        best_model = model_gs

# Output best model and parameters
print("Best Model:")
print(best_model)
print("Best Score:", best_score)
