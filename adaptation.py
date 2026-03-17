import numpy as np
from concurrent_learning import ConcurrentLearningStack

class Adaptation:
    """
    Represents the parameter learning system adapting theta_hat.
    """
    def __init__(self, num_states: int, num_features: int, max_capacity: int, tolerance: float):
        self.history_stack = ConcurrentLearningStack(
            num_states=num_states, 
            num_features=num_features, 
            max_capacity=max_capacity, 
            tolerance=tolerance
        )

    def update_stack(self, X: np.ndarray, Z: np.ndarray) -> bool:
        """Pushes new data to the CL stack."""
        return self.history_stack.update(X, Z)

    def get_cl_grad(self, theta_hat: np.ndarray) -> np.ndarray:
        """Computes the gradient to adapt theta_hat."""
        return self.history_stack.get_cl_gradient(theta_hat)

    def check_pe(self) -> float:
        """Returns the minimum eigenvalue of the information matrix."""
        return self.history_stack.check_persistent_excitation()

    def get_information_matrix(self) -> np.ndarray:
        """Returns the full information matrix."""
        return self.history_stack.get_information_matrix()
