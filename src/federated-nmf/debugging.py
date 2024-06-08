import numpy as np

# Initialize a random matrix A (e.g., sample data)
np.random.seed(0)
A = np.abs(np.random.rand(8, 10))  # 8 samples, 10 features

# Number of components (latent features)
num_components = 5

# Initialize W and H matrices with random non-negative values
W = np.abs(np.random.rand(A.shape[0], num_components))
H = np.abs(np.random.rand(num_components, A.shape[1]))

# Number of NMF iterations
num_iterations = 3

# NMF multiplicative update rules
def nmf_update(A, W, H, num_iter=100):
    for _ in range(num_iter):
        W *= (A @ H.T) / (W @ H @ H.T)
        H *= (W.T @ A) / (W.T @ W @ H)
    return W, H

# Run NMF iterations
for iteration in range(num_iterations):
    W, H = nmf_update(A, W, H)
    print(f"Iteration {iteration + 1}")
    print("W:\n", W)
    print("H:\n", H)

# Final W and H matrices after iterations
print("Final W:\n", W)
print("Final H:\n", H)
