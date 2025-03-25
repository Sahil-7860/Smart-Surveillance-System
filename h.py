import numpy as np

np.random.seed(42)  # DO NOT REMOVE THIS
def run_decomposition(n, L, U):
    # Set the random seed for reproducibility.
    np.random.seed(42)  # DO NOT REMOVE THIS

    # Step 1: Define the prescribed spectrum
    lambda_values = np.array([L + (i - 1) / (n - 1) * (U - L) for i in range(1, n + 1)])

    # Step 2: Generate two independent orthogonal matrices Q and R
    A = np.random.randn(n, n)  # Random matrix for Q
    B = np.random.randn(n, n)  # Random matrix for R
    Q, _ = np.linalg.qr(A)  # Q is orthogonal
    R, _ = np.linalg.qr(B)  # R is orthogonal

    # Step 3: Construct the non-symmetric matrix M
    M = Q @ np.diag(lambda_values) @ R.T

    # Step 4: Compute the eigenvalues of M
    eigenvalues = np.linalg.eig(M)[0]
    abs_eigenvalues = np.abs(eigenvalues)  # Take absolute values
    sorted_abs_eigenvalues = np.sort(abs_eigenvalues)[::-1]  # Sort in descending order

    # Step 5: Compute the singular values of M
    singular_values = np.linalg.svd(M, compute_uv=False)  # Already sorted in descending order

    # Step 6: Compute Δ and R
    delta = np.sum(np.abs(sorted_abs_eigenvalues - singular_values))
    result = np.rint(1e6 * delta).astype(int)  # Round to nearest integer

    return result
print(run_decomposition(50, 1, 100))