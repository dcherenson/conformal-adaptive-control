import numpy as np

class Plant:
    """
    Represents the true physical system dynamics in 2D space.
    State x = [px, py, vx, vy]^T
    """
    def __init__(self):
        self.theta_true = np.array([0.5, 0.2])

    def f(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[2], x[3], 0.0, 0.0])

    def g(self, x: np.ndarray) -> np.ndarray:
        # Input u = [ux, uy]^T affects accelerations
        return np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])

    def F(self, x: np.ndarray) -> np.ndarray:
        # Nonlinear friction/damping on velocities vx, vy
        # Returns a 4x2 matrix mapped to theta_true = [0.5, 0.2]
        return np.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [-x[2], -np.tanh(x[2])],
            [-x[3], -np.tanh(x[3])]
        ])

    def Delta(self, x: np.ndarray, t: float) -> np.ndarray:
        """Time-varying disturbance on velocities."""
        return np.array([0.0, 0.0, 0.3 * np.sin(t), 0.3 * np.cos(t)])

    def step(self, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Computes the state derivative and integrates one step forward."""
        x_dot = self.f(x) + self.g(x) @ u + self.F(x) @ self.theta_true + self.Delta(x, t)
        return x + x_dot * dt
