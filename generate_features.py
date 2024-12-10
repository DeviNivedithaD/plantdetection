import numpy as np

# Dummy features for demonstration
# Replace this with your actual known leaf features
known_leaf_features = [
    np.random.rand(512),  # Example feature vector of length 512
    np.random.rand(512),
    np.random.rand(512)
]

# Save the known leaf features as a .npy file
np.save("known_leaf_features.npy", known_leaf_features)
print("known_leaf_features.npy file created successfully!")