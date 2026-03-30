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
    def get_dist_bound_from_quantile(self, q_k: float, T: float, L_d: float) -> float:
        # Calculate peak assuming it's an isolated triangle
        d_triangle = np.sqrt(2.0 * L_d * q_k)
        
        # Calculate the base width of that triangle
        triangle_base = 2.0 * d_triangle / L_d
        
        # Apply the piecewise truncation logic
        if triangle_base <= T:
            d_bar = d_triangle
        else:
            d_bar = (q_k / T) + 0.5 * L_d * T
        return d_bar
