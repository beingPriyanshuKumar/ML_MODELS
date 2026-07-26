import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch Version: {torch.__version__}")

# ==========================================
# Part 1 - Data Preprocessing
# ==========================================

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert numpy arrays to PyTorch Float Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # unsqueeze makes it a 2D column vector
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for Batching (equivalent to batch_size=32 in Keras)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# ==========================================
# Part 2 - Building the ANN
# ==========================================

# Define the network architecture
# We calculate the input size dynamically from our features
input_size = X_train.shape[1] 

ann = nn.Sequential(
    nn.Linear(input_size, 6),  # Input layer to first hidden layer
    nn.ReLU(),
    nn.Linear(6, 6),           # Second hidden layer
    nn.ReLU(),
    nn.Linear(6, 1),           # Output layer (Sigmoid is applied later or handled by loss)
    nn.Sigmoid()
)


# ==========================================
# Part 3 - Training the ANN
# ==========================================

# Compiling equivalents: Loss and Optimizer
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(ann.parameters(), lr=0.001)

# Training loop (equivalent to ann.fit)
epochs = 100
ann.train() # Set model to training mode

for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        outputs = ann(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 2. Backward pass and optimization
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights


# ==========================================
# Part 4 - Predictions and Evaluation
# ==========================================

ann.eval() # Set model to evaluation mode
with torch.no_grad(): # Turn off gradient tracking for predictions

    # Homework: Predicting the result of a single observation
    single_obs = sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
    single_obs_tensor = torch.tensor(single_obs, dtype=torch.float32)
    
    single_prediction = ann(single_obs_tensor)
    print("Single customer leaves prediction:")
    print((single_prediction > 0.5).item()) # .item() extracts the boolean value

    # Predicting the Test set results
    y_pred_tensor = ann(X_test_tensor)
    y_pred = (y_pred_tensor > 0.5).numpy() # Convert back to numpy for sklearn metrics

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")