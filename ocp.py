import numpy as np

class OCPBase:
    """
    Base class for Online Conformal Prediction (OCP).
    Manages the estimated quantile q_k and the learning rate update step.
    """
    def __init__(self, alpha: float, eta_const: float, q_init: float = 0.0):
        """
        Args:
            alpha (float): User-specified mis-coverage rate in (0, 1).
            eta_const (float): Constant step size (eta_k = eta_const).
            q_init (float): Initial value for the estimated quantile (q_0).
        """
        if not (0 < alpha < 1):
            raise ValueError("Mis-coverage rate alpha must be between 0 and 1.")
            
        self.alpha = alpha
        self.eta_const = eta_const
        self.q_k = max(0.0, q_init) # q_k must be non-negative
        self.k = 1  # Step index for step size scaling (1-indexed to avoid div by zero)

    def update(self, S_k: float) -> float:
        """
        Updates the estimated quantile q_k using online gradient descent on the pinball loss.
        Eq (29) from the paper.
        
        Args:
            S_k (float): The current non-conformity score at time t_k.
            
        Returns:
            float: The newly updated quantile q_{k+1}.
        """
        # Constant step size
        eta_k = self.eta_const
        
        # Indicator function: 1 if S_k <= q_k, else 0
        indicator = 1.0 if S_k > self.q_k else 0.0
        
        # Update equation (29) modified for constant step size
        self.q_k = max(0.0, self.q_k + eta_k * (indicator - self.alpha))
        
        # Increment step index
        self.k += 1
        
        return self.q_k

    def get_quantile(self) -> float:
        """Returns the current estimated quantile q_k."""
        return self.q_k


class DriftScoreOCP(OCPBase):
    """
    Computes the non-conformity score for model drift using Observer and Adaptation outputs.
    Follows Eq (28) from the paper.
    """
    def compute_score(self, X: np.ndarray, Z: np.ndarray, theta_hat: np.ndarray) -> float:
        """
        Args:
            X (np.ndarray): Current state differential or estimation from Observer.
            Z (np.ndarray): Current regressor matrix from Observer.
            theta_hat (np.ndarray): Current estimated parameters from Adaptation.
            
        Returns:
            float: The non-conformity score S_k = ||X - Z \hat{\theta}||
        """
        # Ensure dimensions align for matrix multiplication
        if theta_hat.ndim == 1:
            theta_hat = theta_hat.reshape(-1, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # prediction: Z * \hat{\theta}
        prediction = Z @ theta_hat
        
        # error norm ||X - Z\hat{\theta}||
        S_k = np.linalg.norm(X - prediction)
        return S_k

class InnovationScoreOCP(OCPBase):
    """
    Computes the non-conformity score for measurement innovation (Sensor vs Observer).
    """
    def compute_score(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Args:
            y (np.ndarray): Actual sensor measurement.
            y_hat (np.ndarray): Predicted measurement from Observer (C * \hat{x}).
            
        Returns:
            float: The innovation non-conformity score S_k = ||y - \hat{y}||
        """
        S_k = np.linalg.norm(y - y_hat)
        return S_k
