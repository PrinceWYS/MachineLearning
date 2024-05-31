import numpy as np
from tqdm import tqdm
class SVM:
    def __init__(self, X, y, kernel='linear', C=1.0, tol=1e-3, max_iter=1000):
        self.X = np.array(X)
        self.y = np.array(y)
        self.kernel = kernel
        self.C = C       
        self.tol = tol
        self.max_iter = max_iter
    
        n_samples, n_features = len(X), len(X[0])
        self.n_samples = n_samples
        self.n_features = n_features
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        self.w = np.zeros(n_features)
        self.kernel_matrix = np.zeros((n_samples, n_samples))
        self.E = np.zeros(n_samples)
    
    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1.T, x2)
        elif self.kernel == 'rbf':
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))
        else:
            raise ValueError('Invalid kernel')
    
    def ktt(self, i):
        Ei = self._E(i)
        if (self.y[i] * Ei < -self.tol and self.alpha[i] < self.C) or (self.y[i] * Ei > self.tol and self.alpha[i] > 0):
            return True
        else:
            return False
    
    def inner_loop(self, i):
        Ei = self._E(i)
        max_diff = 0
        j = None
        E = np.nonzero(self.E)[0]
        if len(E) > 1:
            for k in E:
                if k == i:
                    continue
                Ek = self._E(k)
                diff = np.abs(Ei - Ek)
                if diff > max_diff:
                    max_diff = diff
                    j = k
        else:
            j = i
            while j == i:
                j = int(np.random.uniform(0, self.n_samples))
        
        return j

    def update(self, i, j):
        alpha_i_old = self.alpha[i]
        alpha_j_old = self.alpha[j]
        
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
            H = min(self.C, self.alpha[j] + self.alpha[i])

        if L == H:
            return 0
        eta = self.kernel_matrix[i, i] + self.kernel_matrix[j, j] - 2 * self.kernel_matrix[i, j]
        if eta <= 0:
            return 0
        
        def clip_bonder(alpha_new_unc, L, H):
            if alpha_new_unc < L:
                return L
            elif alpha_new_unc > H:
                return H
            else:
                return alpha_new_unc
        alpha_j_new_unc = alpha_j_old + self.y[j] * (self.E[i] - self.E[j]) / eta
        alpha_j_new = clip_bonder(alpha_j_new_unc, L, H)
        alpha_i_new = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - alpha_j_new)
        # update alpha
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        # update b
        b1 = self.b - self.E[i] - self.y[i] * (alpha_i_new - alpha_i_old) * self.kernel_matrix[i, i] - self.y[j] * (alpha_j_new - alpha_j_old) * self.kernel_matrix[i, j]
        b2 = self.b - self.E[j] - self.y[i] * (alpha_i_new - alpha_i_old) * self.kernel_matrix[i, j] - self.y[j] * (alpha_j_new - alpha_j_old) * self.kernel_matrix[j, j]
        
        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        # update E
        self.E[i] = self._E(i)
        self.E[j] = self._E(j)
        
        return 1
    
    def fit(self):
        # n_samples, n_features = len(self.X), len(self.X[0])
        n_samples, n_features = self.X.shape
        self.kernel_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                self.kernel_matrix[i, j] = self.kernel_function(self.X[i], self.X[j])
        # self.kernel_matrix = np.array([[self.kernel_function(self.X[i], self.X[j]) for j in range(self.n_samples)] for i in range(self.n_samples)])
        
        for _ in tqdm(range(self.max_iter)):
            alpha_prev = np.copy(self.alpha)
            for i in range(n_samples):
                if self.ktt(i):
                    j = self.inner_loop(i)
                    self.update(i, j)
        self.w = self._W(self.alpha, self.X, self.y)
        print(f'[INFO] Training finished. W: {self.w},\n b: {self.b},\n alpha: {self.alpha}')
        return self.w, self.b, self.alpha
      

    def predict(self, X, y=None):
        sum = 0
        # total = len(X)
        X = np.array(X)
        total = X.shape[0]
        pred_y = []
        
        for i in range(total):
            res = np.dot(self.w.T, X[i, :]) + self.b
            res = np.sign(res)
            pred_y.append(res)
            if y is not None:
                if res == y[i]:
                    sum += 1
        
        if y is not None:
            print(f'[INFO] Accuracy: {sum / total}')
            print(f'[INFO] label of test:\n{pred_y}')
        else:
            print(f'[INFO] Cannot calculate accuracy without y_true.')
                        
    def _g(self, i):
        return self.b + np.sum(self.alpha * self.y * self.kernel_matrix[i])

    def _E(self, i):
        return self._g(i) - self.y[i]

    def _W(self, alpha, X, y):
        return np.dot(X.T, alpha * y)