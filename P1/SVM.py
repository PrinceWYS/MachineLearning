# implete SVMs; solved in both the primal and the dual; You are allowed to use any off-the-shelf toolbox to solve the resulting Quadratic Programming problem. Do keep it in mind, though, you are not allowed to call existing SVM toolboxes/APIs.
import numpy as np

class SVM:
    def __init__(self, max_iter=1000, C=1.0, tol=1e-3, kernel='linear'):
        self.max_iter = max_iter    # Maximum number of iterations
        self.C = C                  # Regularization parameter
        self.tol = tol              # Tolerance
        self.kernel = kernel        # Kernel type
        self.alpha = None           # Lagrange multipliers
        self.b = None               # Bias
        self.support_vectors = None # Support vectors
        self.support_vector_labels = None   # Labels of support vectors
        self.kernel_cache = None    # Cache for kernel values
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize kernel cache
        self.kernel_cache = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                self.kernel_cache[i, j] = self.kernel_function(X[i], X[j])
                self.kernel_cache[j, i] = self.kernel_cache[i, j]
        
        # Initialize alpha
        self.alpha = np.zeros(n_samples)
        
        # SMO algorithm
        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            for i in range(n_samples):
                j = self.random_index(i, n_samples)
                if self.update_alpha(i, j, X, y):
                    break
            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break
        
        # Compute bias
        self.b = 0
        for i in range(n_samples):
            self.b += y[i]
            for j in range(n_samples):
                self.b -= self.alpha[j] * y[j] * self.kernel_cache[i, j]
        self.b /= n_samples
        
        # Get support vectors
        self.support_vectors = X[self.alpha > 1e-5]
        self.support_vector_labels = y[self.alpha > 1e-5]