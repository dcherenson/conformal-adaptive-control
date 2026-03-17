import numpy as np

class ConcurrentLearningStack:
    def __init__(self, num_states, num_features, max_capacity=10, tolerance=1e-3):
        """
        Manages the history stack for Concurrent Learning.
        num_states: Dimension of X (p)
        num_features: Dimension of theta (m)
        max_capacity: Number of orthogonal points to store (P)
        tolerance: Minimum Frobenius norm difference required to record new data
        """
        self.max_capacity = max_capacity
        self.tolerance = tolerance
        
        # Pre-allocate history tensors
        # X_hist shape: (P, num_states, 1)
        # Z_hist shape: (P, num_states, num_features)
        self.X_hist = np.zeros((max_capacity, num_states, 1))
        self.Z_hist = np.zeros((max_capacity, num_states, num_features))
        
        self.current_size = 0
        self.head_index = 0  # Pointer for the FIFO cyclic overwrite
        self.last_added_Z = None

    def _check_richness(self, Z):
        """Evaluates if the new regressor Z is geometrically distinct."""
        if self.current_size == 0:
            return True
            
        # Frobenius norm measures the spatial difference between matrices
        diff = np.linalg.norm(Z - self.last_added_Z, ord='fro')
        return diff > self.tolerance

    def update(self, X, Z):
        """
        Attempt to add a new (X, Z) pair to the stack.
        Returns True if accepted by the richness filter, False otherwise.
        """
        if not self._check_richness(Z):
            return False  # Reject data (robot is just sitting still or moving linearly)

        # Reshape X if it was passed as a flat array to maintain tensor math
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.current_size < self.max_capacity:
            # Stack is not full yet; append to the end
            idx = self.current_size
            self.current_size += 1
        else:
            # Stack is full; overwrite the oldest data (FIFO)
            idx = self.head_index
            self.head_index = (self.head_index + 1) % self.max_capacity
            
        self.X_hist[idx] = X
        self.Z_hist[idx] = Z
        self.last_added_Z = np.copy(Z)
        
        return True

    def get_cl_gradient(self, theta_hat):
        """
        Calculates the concurrent learning parameter update term.
        Mathematically: sum(Z_j^T * (X_j - Z_j * theta_hat))
        """
        if self.current_size == 0:
            return np.zeros_like(theta_hat)
        
        # Slice only the valid data (important during the initial fill phase)
        Z_valid = self.Z_hist[:self.current_size]
        X_valid = self.X_hist[:self.current_size]
        
        # Vectorized batch prediction: (N, p, m) @ (m, 1) -> (N, p, 1)
        if theta_hat.ndim == 1:
            theta_hat = theta_hat.reshape(-1, 1)
        X_hat = Z_valid @ theta_hat
        
        # Batch error calculation: (N, p, 1)
        error = X_valid - X_hat
        
        # Batch gradient calculation: Z^T * error
        # Transpose the inner matrices of Z: (N, p, m) -> (N, m, p)
        Z_T = np.transpose(Z_valid, (0, 2, 1))
        
        # (N, m, p) @ (N, p, 1) -> (N, m, 1)
        gradients = Z_T @ error
        
        # Sum across the batch dimension to get the final (m, 1) update vector
        cl_grad = np.sum(gradients, axis=0)
        
        return cl_grad

    def get_information_matrix(self):
        """Returns the sum(Z^T * Z) matrix."""
        if self.current_size == 0:
            return np.zeros((self.Z_hist.shape[2], self.Z_hist.shape[2]))
        Z_valid = self.Z_hist[:self.current_size]
        Z_T = np.transpose(Z_valid, (0, 2, 1))
        return np.sum(Z_T @ Z_valid, axis=0)

    def check_persistent_excitation(self):
        """
        Calculates the minimum eigenvalue of the information matrix.
        Use this to verify your stack mathematically satisfies the PE condition.
        """
        Omega = self.get_information_matrix()
        if self.current_size == 0:
            return 0.0
        eigenvalues = np.linalg.eigvals(Omega)
        return np.min(np.real(eigenvalues))
