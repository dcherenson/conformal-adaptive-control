import numpy as np
from plant import Plant

class Observer:
    """
    Represents the state observer estimating x_hat.
    """
    def __init__(self, plant: Plant, C: np.ndarray, L: np.ndarray):
        self.plant = plant
        self.C = C
        self.L = L

    def compute_xhat_dot(self, xhat: np.ndarray, u: np.ndarray, theta_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes the observer derivative."""
        return (self.plant.f(xhat) + 
                self.plant.g(xhat) @ u + 
                self.plant.F(xhat) @ theta_hat + 
                self.L @ (y - self.C @ xhat))

    def get_y_hat(self, xhat: np.ndarray) -> np.ndarray:
        return self.C @ xhat

    def compute_X(self, xhat_new: np.ndarray, xhat_old: np.ndarray, u_new: np.ndarray, u_old: np.ndarray, dt: float) -> np.ndarray:
        old = self.plant.f(xhat_old) + self.plant.g(xhat_old) @ u_old
        new = self.plant.f(xhat_new) + self.plant.g(xhat_new) @ u_new
        return xhat_new - xhat_old - 0.5 * dt * (old + new)

    def compute_Z(self, xhat_new: np.ndarray, xhat_old: np.ndarray, dt: float) -> np.ndarray:
        old = self.plant.F(xhat_old)
        new = self.plant.F(xhat_new)
        return 0.5 * dt * (old + new)
