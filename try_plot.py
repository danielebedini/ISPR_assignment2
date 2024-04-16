import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data = pd.read_csv(url, names=column_names)

# Display the first few rows of the dataset
print(data.head())

# Separate features and target variable
X = data.drop('class', axis=1)  # Features
y = data['class']  # Target

# Perform one-hot encoding on features
X_encoded = pd.get_dummies(X)

# Display the first few rows of the encoded dataset
print(X_encoded.head())

# Separate features and target variable
X = data.drop('class', axis=1)  # Features
y = data['class']  # Target

tr_data_unencoded = pd.concat([X, y], axis=1)

print(X.head())
print(y.head())


# Perform one-hot encoding on features
#X_encoded = pd.get_dummies(X)

# Display the first few rows of the encoded dataset
#print(X_encoded.head())

#from pgmpy.estimators import PC
#from pgmpy.models import BayesianModel

# Concatenate encoded features with target variable
#data_encoded = pd.concat([X_encoded, y], axis=1)

#from sklearn.model_selection import train_test_split

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#tr_set_encoded = pd.concat([X_train, y_train], axis=1)

#test_set_encoded = pd.concat([X_test, y_test], axis=1)

'''
# Initialize the PC algorithm object
pc = PC(tr_set_encoded)

# Run the PC algorithm to learn the structure of the Bayesian Network
estimated_model = pc.estimate()

# Get the learned structure (edges) of the Bayesian Network
print(estimated_model.edges())





from scores import compute_correlation_score, compute_structure_score



# Compute the correlation score of the learned model
correlation_score = compute_correlation_score(estimated_model, test_set)
print(f"Correlation Score: {correlation_score}")

# Compute the structure score of the learned model
structure_score = compute_structure_score(estimated_model, test_set)
print(f"Structure Score: {structure_score}")
'''

def plot_graph(estimated_model):

    # Create a directed graph from the learned model
    edges = estimated_model.edges()
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Plot the directed graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Set layout for better visualization
    nx.draw(G, pos, with_labels=True, node_size=1200, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
    plt.title("Learned Bayesian Network Structure (PC Algorithm)")
    plt.show()