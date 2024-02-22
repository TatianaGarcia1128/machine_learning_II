import numpy as np
import fbpca

class RobustPCA:
    def __init__(self, n_components, mu=1.0, tol=1e-9, max_iter=100):
        self.n_components = n_components
        self.mu = mu
        self.tol = tol
        self.max_iter = max_iter
        self.components = None
        self.mean = None

    def converged(self, Z, d_norm):
        err = np.linalg.norm(Z, 'fro') / d_norm
        print('error: ', err)
        return err < self.tol

    def shrink(self, M, tau):
        S = np.abs(M) - tau
        return np.sign(M) * np.where(S>0, S, 0)

    def _svd(self, M, rank): return fbpca.pca(M, k=min(rank, np.min(M.shape)), raw=True)

    def norm_op(self, M): return self._svd(M, 1)[1][0]

    def svd_reconstruct(self, M, rank, min_sv):
        u, s, v = self._svd(M, rank)
        s -= min_sv
        nnz = (s > 0).sum()
        return u[:,:nnz] @ np.diag(s[:nnz]) @ v[:nnz], nnz
            
    def pcp(self, X, maxiter=10, k=10): # refactored
        m, n = X.shape
        trans = m<n
        if trans: X = X.T; m, n = X.shape
            
        lamda = 1/np.sqrt(m)
        op_norm = self.norm_op(X)
        Y = np.copy(X) / max(op_norm, np.linalg.norm( X, np.inf) / lamda)
        mu = k*1.25/op_norm; mu_bar = mu * 1e7; rho = k * 1.5
        
        d_norm = np.linalg.norm(X, 'fro')
        L = np.zeros_like(X); sv = 1
        
        examples = []
        
        for i in range(maxiter):
            print("rank sv:", sv)
            X2 = X + Y/mu
            
            # update estimate of Sparse Matrix by "shrinking/truncating": original - low-rank
            S = self.shrink(X2 - L, lamda/mu)
            
            # update estimate of Low-rank Matrix by doing truncated SVD of rank sv & reconstructing.
            # count of singular values > 1/mu is returned as svp
            L, svp = self.svd_reconstruct(X2 - S, sv, 1/mu)
            
            # If svp < sv, you are already calculating enough singular values.
            # If not, add 20% (in this case 240) to sv
            sv = svp + (1 if svp < sv else round(0.05*n))
            
            # residual
            Z = X - L - S
            Y += mu*Z; mu *= rho
            
            examples.extend([S[140,:], L[140,:]])
            
            if m > mu_bar: m = mu_bar
            if self.converged(Z, d_norm): break
        
        if trans: L=L.T; S=S.T
        return L, S, examples

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        print('X-center', X_centered, type(X_centered), X.shape)

        # Initialization
        L, S, Y = self.pcp(X=X_centered, maxiter=5, k=10)

        # Extract principal components from low-rank matrix
        U, Sigma, VT = np.linalg.svd(L)
        self.components = VT[:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def soft_threshold(self, X, thresh):
        return np.sign(X) * np.maximum(np.abs(X) - thresh, 0)