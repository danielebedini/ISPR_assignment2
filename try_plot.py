import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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

# Perform one-hot encoding on features
X_encoded = pd.get_dummies(X)

# Display the first few rows of the encoded dataset
print(X_encoded.head())

from pgmpy.estimators import PC
from pgmpy.models import BayesianModel

# Concatenate encoded features with target variable
data_encoded = pd.concat([X_encoded, y], axis=1)

# Initialize the PC algorithm object
pc = PC(data_encoded)

# Run the PC algorithm to learn the structure of the Bayesian Network
estimated_model = pc.estimate()

# Get the learned structure (edges) of the Bayesian Network
print(estimated_model.edges())

import networkx as nx
import matplotlib.pyplot as plt

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