import numpy as np
from scipy.integrate import solve_ivp

class Plant:
    """
    Quadcopter Point-Mass Simulation (3D).
    State x = [px, py, pz, vx, vy, vz]^T
    Control u = [ux, uy, uz]^T (Net Force vector)
    """
    def __init__(self):
        self.m = 1.0  # kg
        self.g = np.array([0, 0, -9.81])
        
        # We don't use theta_true for concurrent learning anymore, but keep for compatibility if needed elsewhere
        self.theta_true = np.array([0.5, 0.2])

    def wind_disturbance(self, t, p):
        # Average intensity 1.0 m/s^2, highly dynamic
        # Based on fans: varying sine waves.
        d_x = 1.0 * np.sin(0.5 * t) + 0.5 * np.sin(2.0 * t) + 0.2 * np.random.randn()
        d_y = 1.2 * np.cos(0.4 * t) + 0.6 * np.cos(1.8 * t) + 0.2 * np.random.randn()
        d_z = 0.5 * np.sin(0.3 * t) + 0.1 * np.random.randn()
        return np.array([d_x, d_y, d_z]) * self.m  # Force disturbance

    def f(self, x: np.ndarray) -> np.ndarray:
        # Expected nominal drift dynamics [dp, dv] where dv = g
        return np.array([x[3], x[4], x[5], 0.0, 0.0, -9.81])

    def g_mat(self, x: np.ndarray) -> np.ndarray:
        # Input u = [ux, uy, uz]^T affects accelerations
        # dv = u / m
        mat = np.zeros((6, 3))
        mat[3, 0] = 1.0 / self.m
        mat[4, 1] = 1.0 / self.m
        mat[5, 2] = 1.0 / self.m
        return mat

    def F(self, x: np.ndarray) -> np.ndarray:
        # Concurrent learning compatibility filler, unused with SSML
        return np.zeros((6, 2))

    def Delta(self, x: np.ndarray, t: float) -> np.ndarray:
        # Additive mismatch acceleration filler
        return np.concatenate((np.zeros(3), self.wind_disturbance(t, x[0:3]) / self.m))

    def dynamics(self, t, state, u):
        # state = [p_x, p_y, p_z, v_x, v_y, v_z]
        p = state[0:3]
        v = state[3:6]
        d = self.wind_disturbance(t, p)

        dp = v
        # m a = m g + u + d
        dv = self.g + (u + d) / self.m

        return np.concatenate((dp, dv))

    def step(self, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Computes the state derivative and integrates one step forward."""
        sol = solve_ivp(self.dynamics, [t, t + dt], x, args=(u,), method="RK45")
        return sol.y[:, -1]
