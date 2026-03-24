import numpy as np
from scipy.integrate import solve_ivp

class Plant:
    """
    Quadcopter Simulation (3D with Attitude).
    State x = [px, py, pz, vx, vy, vz, phi, theta]^T
    Control u = [p, q, T]^T (Roll rate, Pitch rate, Net Thrust)
    """
    def __init__(self):
        self.m = 1.0  # kg
        self.g = np.array([0, 0, -9.81])
        self.spatial_mode = False  # Toggle for online spatial gradient shift
        
        # We don't use theta_true for concurrent learning anymore, but keep for compatibility if needed elsewhere
        self.theta_true = np.array([0.5, 0.2])
        
        # Lipschitz Constants
        self.L_nom_x = 1.0  # From velocity block [0 I; 0 0]
        self.L_nom_u = 1.0 / self.m
        self.L_true_x = 1.0
        self.L_true_u = 1.0 / self.m

    def wind_disturbance(self, t, p):
        # Time-varying component
        d_x = 1.0 * np.sin(0.5 * t) + 0.5 * np.sin(2.0 * t) + 0.2 * np.random.randn()
        d_y = 1.2 * np.cos(0.4 * t) + 0.6 * np.cos(1.8 * t) + 0.2 * np.random.randn()
        d_z = 0.5 * np.sin(0.3 * t) + 0.1 * np.random.randn()
        
        # Spatial component (linear gradient based on position)
        if self.spatial_mode:
            d_x += 1.0 * p[0]
            d_y += 1.0 * p[1]
            d_z += 0.5 * p[2]
            
        return np.array([d_x, d_y, d_z]) * self.m  # Force disturbance

    def f(self, x: np.ndarray) -> np.ndarray:
        # Expected nominal drift dynamics [dp, dv, dAngles] where dv = g
        return np.array([x[3], x[4], x[5], 0.0, 0.0, -9.81, 0.0, 0.0])

    def g_mat(self, x: np.ndarray) -> np.ndarray:
        # Input u = [p, q, T]^T
        # dv = R(phi, theta) * [0, 0, T]^T / m
        # dAngles = [p, q]
        phi = x[6]
        theta = x[7]
        
        mat = np.zeros((8, 3))
        # Acceleration from Thrust
        mat[3, 2] = np.sin(theta) / self.m
        mat[4, 2] = -np.cos(theta) * np.sin(phi) / self.m
        mat[5, 2] = np.cos(theta) * np.cos(phi) / self.m
        
        # Angular rates from control
        mat[6, 0] = 1.0  # p -> dot{phi}
        mat[7, 1] = 1.0  # q -> dot{theta}
        return mat

    def F(self, x: np.ndarray) -> np.ndarray:
        # Concurrent learning compatibility filler, unused with SSML
        return np.zeros((8, 2))

    def Delta(self, x: np.ndarray, t: float) -> np.ndarray:
        # Additive mismatch acceleration filler
        return np.concatenate((np.zeros(3), self.wind_disturbance(t, x[0:3]) / self.m))

    def dynamics(self, t, state, u):
        # state = [px, py, pz, vx, vy, vz, phi, theta]
        p = state[0:3]
        v = state[3:6]
        angles = state[6:8]
        phi, theta = angles
        
        d = self.wind_disturbance(t, p)

        dp = v
        # m a = m g + R * T + d
        dv = self.g + d / self.m
        # Control u = [p, q, T]
        p_rate = u[0]
        q_rate = u[1]
        thrust = u[2]
        
        dv[0] += thrust / self.m * np.sin(theta)
        dv[1] += thrust / self.m * (-np.cos(theta) * np.sin(phi))
        dv[2] += thrust / self.m * (np.cos(theta) * np.cos(phi))
        
        dAngles = np.array([p_rate, q_rate])

        return np.concatenate((dp, dv, dAngles))

    def step(self, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Computes the state derivative and integrates one step forward."""
        sol = solve_ivp(self.dynamics, [t, t + dt], x, args=(u,), method="RK45")
        return sol.y[:, -1]
