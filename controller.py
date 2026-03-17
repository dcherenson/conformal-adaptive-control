import numpy as np
from scipy.optimize import minimize
from qpsolvers import solve_qp
from plant import Plant

class SafetyCriticalController:
    """
    Computes the control input u using a Control Barrier Function QP
    constrained by the OCP-derived Uncertainty Margin.
    """
    def __init__(self, plant: Plant, x_obs: np.ndarray, d_safe: float, lambda_cbf: float = 1.0, 
                 epsilon_bar: float = 0.1, T_horizon: float = 1.0, C_m_min_sv: float = 1.0, u_max: float = 10.0,
                 alpha_hocbf: float = 1.0):
        self.plant = plant
        self.x_obs = x_obs
        self.d_safe = d_safe
        self.lambda_cbf = lambda_cbf
        self.u_max = u_max
        self.alpha_hocbf = alpha_hocbf
        
        # System noise and measurement properties from Lemmas 1 & 2
        self.epsilon_bar = epsilon_bar
        self.T_horizon = T_horizon
        self.C_m_min_sv = C_m_min_sv

    def h0(self, x: np.ndarray) -> float:
        """Original distance boundary."""
        p = np.array([x[0], x[1]])
        obs = np.array([self.x_obs[0], self.x_obs[1]])
        return np.linalg.norm(p - obs)**2 - self.d_safe**2

    def h(self, x: np.ndarray) -> float:
        """High-Order Control Barrier Function (HOCBF) h1(x) = h0_dot + alpha * h0."""
        # h0_dot = 2 * (px - obs_x)*vx + 2 * (py - obs_y)*vy
        h0_val = self.h0(x)
        h0_dot = 2 * (x[0] - self.x_obs[0]) * x[2] + 2 * (x[1] - self.x_obs[1]) * x[3]
        return h0_dot + self.alpha_hocbf * h0_val

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the HOCBF h1 w.r.t [px, py, vx, vy]."""
        # h1(x) = 2(px-obs_x)vx + 2(py-obs_y)vy + alpha*((px-obs_x)^2 + (py-obs_y)^2 - d_safe^2)
        # dh1/dpx = 2*vx + 2*alpha*(px-obs_x)
        # dh1/dpy = 2*vy + 2*alpha*(py-obs_y)
        # dh1/dvx = 2*(px-obs_x)
        # dh1/dvy = 2*(py-obs_y)
        dh_dpx = 2 * x[2] + 2 * self.alpha_hocbf * (x[0] - self.x_obs[0])
        dh_dpy = 2 * x[3] + 2 * self.alpha_hocbf * (x[1] - self.x_obs[1])
        dh_dvx = 2 * (x[0] - self.x_obs[0])
        dh_dvy = 2 * (x[1] - self.x_obs[1])
        return np.array([dh_dpx, dh_dpy, dh_dvx, dh_dvy])

    def nominal_controller(self, xhat: np.ndarray, xd: np.ndarray) -> np.ndarray:
        """A simple PD controller guiding the system to xd = [px, py, vx, vy]."""
        Kp = 2.0
        Kd = 1.5
        u_x = Kp * (xd[0] - xhat[0]) - Kd * xhat[2]
        u_y = Kp * (xd[1] - xhat[1]) - Kd * xhat[3]
        return np.array([u_x, u_y])

    def compute_u(self, xhat: np.ndarray, theta_hat: np.ndarray, xd: np.ndarray, 
                  q_drift: float, q_inn: float) -> np.ndarray:
        
        # 1. Compute Nominal Control
        u_nom = self.nominal_controller(xhat, xd)

        # 2. Compute Data-Driven Bounds
        # Lemma 1: Measured state error bound
        x_bar_m = (q_inn + self.epsilon_bar) / self.C_m_min_sv
        
        # Lemma 2: Unmeasured state error and disturbance
        # Assuming partial observability or generalized bounds
        x_bar_u = q_drift
        E_k = q_drift / self.T_horizon
        
        # 3. Compute CBF components at estimate xhat
        h_val = self.h(xhat)
        grad_h_val = self.grad_h(xhat)
        
        # Expected drift dynamics without true parameters
        x_dot_drift = self.plant.f(xhat) + self.plant.F(xhat) @ theta_hat
        
        # Lie derivatives
        L_f_h = grad_h_val @ x_dot_drift
        L_g_h = grad_h_val @ self.plant.g(xhat)
        
        # Lipschitz constants for distance squared function
        # L_h approximately 2||x - obs||
        L_h = np.linalg.norm(grad_h_val)
        # Gradient is exactly 2x, Lipschitz constant of gradient partitions is 2
        L_grad_m = 2.0
        L_grad_u = 2.0
        
        # 4. Integrate Dedicated QP Solver (Refactoring SLSQP)
        # We linearize the disturbance norm ||f(x) + g(x)u|| using u_nom
        # to ensure the constraint is purely linear in u.
        
        # Linearized Margin: M_lin = term1 + term2_lin + term3
        term1 = (np.linalg.norm(grad_h_val) + L_grad_m*x_bar_m + L_grad_u*x_bar_u) * E_k
        term2_lin = (L_grad_m*x_bar_m + L_grad_u*x_bar_u) * np.linalg.norm(x_dot_drift + self.plant.g(xhat) @ u_nom)
        term3 = self.lambda_cbf * L_h * (x_bar_m + x_bar_u)
        margin_lin = term1 + term2_lin + term3

        # QP Formulation: min 0.5 * u.T * P * u + q.T * u
        # s.t. G * u <= h
        
        # Objective ||u - u_nom||^2 = 0.5 * u.T * (2*I) * u + (-2*u_nom).T * u
        P = 2.0 * np.eye(2)
        q = -2.0 * u_nom

        # Constraint Lgh*u >= -lambda*h - Lfh + margin_lin
        # => -Lgh * u <= lambda*h + Lfh - margin_lin
        G = -L_g_h.reshape(1, 2)
        h_qp = np.array([self.lambda_cbf * h_val + L_f_h - margin_lin])

        # Solve using OSQP via qpsolvers
        u_qp = solve_qp(P, q, G, h_qp, solver="osqp")

        if u_qp is not None:
            # Enforce actuator saturation strictly
            u_norm = np.linalg.norm(u_qp)
            if u_norm > self.u_max:
                u_qp = (u_qp / u_norm) * self.u_max
            return u_qp
        else:
            # Raise exception on infeasibility as requested to stop the program
            raise RuntimeError(f"QP Infeasible at xhat={xhat}, margin_lin={margin_lin:.4f}")


class RobustTubeMPC:
    """
    Implements the Dual-Conformal Robust Tube MPC (Theorem 4).
    Dynamically bounds uncertainty tubes using OCP parameters and tightens 
    predictive geometry over the horizon H.
    """
    def __init__(self, plant: Plant, x_obs: np.ndarray, d_safe: float, H: int = 5,
                 dt: float = 0.1, epsilon_bar: float = 0.1, C_m_min_sv: float = 1.0, 
                 T_horizon: float = 1.0):
        self.plant = plant
        self.x_obs = x_obs
        self.d_safe = d_safe
        self.H = H
        self.dt = dt
        
        # OCP Mapping parameters
        self.epsilon_bar = epsilon_bar
        self.C_m_min_sv = C_m_min_sv
        self.T_horizon = T_horizon
        
        # Ancillary controller gain K for the pointmass (Double Integrator LQR proxy)
        # u = v + K(xhat - z)
        # K needs to be stabilizing for A_cl = A + BK. We use a simple PD structure.
        self.Kp = 4.0
        self.Kd = 2.0
        # A_cl mapped continuous bounds: A_cl limits tracking error
        self.u_max = 10.0 # Bounding input capability
        self.A_cl_op_norm = 0.90 # Approximation of strict contractive decay for error bounds

    def compute_u(self, xhat: np.ndarray, theta_hat: np.ndarray, xd: np.ndarray, 
                  q_drift: float, q_inn: float) -> np.ndarray:
        """
        Solves the Optimal Control Problem (OCP) sequence v_{0..H-1} and returns u_0.
        """
        # 0. Sanity Check for Divergence
        if np.any(np.isnan(xhat)) or np.any(np.isnan(theta_hat)):
            return np.zeros(2)
            
        # 1. Compute Data-Driven Uncertainty Bounds
        x_bar_m = (q_inn + self.epsilon_bar) / self.C_m_min_sv
        x_bar_u = q_drift
        initial_error_bound = x_bar_m + x_bar_u
        disturbance_bound = q_drift / self.T_horizon
        
        # Calculate the cumulative Minkowski cross-section constants upfront
        # radius_{j|k} = ||A_cl||^j * ||xhat_k - z_0|| + ||A_cl||^j*E_x + sum_0^{j-1} ||A_cl||^i * E_k
        # Since we lock z_0 = xhat_k for simplicty on the first step, ||xhat_k - z_0|| = 0.
        tube_radii = np.zeros(self.H + 1)
        for j in range(self.H + 1):
            tube_radii[j] = initial_error_bound * (self.A_cl_op_norm**j)
            if j > 0:
                dist_sum = sum((self.A_cl_op_norm**i) * disturbance_bound for i in range(j))
                tube_radii[j] += dist_sum
            # Cap the radius to prevent optimization blowup under extreme uncertainty
            tube_radii[j] = min(tube_radii[j], 10.0)
                
        # 2. Formulate the finite horizon SLSQP Nonlinear problem
        # We need to optimize the virtual control sequence v_0...v_{H-1}
        # Dimensions: H vectors of shape (2,) -> flat array of size 2*H
        
        def objective(V_flat):
            V = V_flat.reshape(self.H, 2)
            cost = 0.0
            
            # Forward propagate virtual dynamics z
            # We assume initial virtual state z_0 equals the current observer state
            z = np.copy(xhat)
            for j in range(self.H):
                # Quadratic stage cost ||z_j - x_ref||^2_Q + ||v_j||^2_R
                cost += 50.0 * np.sum((z[:2] - xd[:2])**2) + 1.0 * np.sum((z[2:] - xd[2:])**2) + 0.05 * np.sum(V[j]**2)
                
                # Next virtual state z_{j+1} using true estimated physics
                # z_dot = f(z) + g(z)v + F(z)theta_hat
                z_dot = self.plant.f(z) + self.plant.g(z) @ V[j] + self.plant.F(z) @ theta_hat
                z = z + z_dot * self.dt
            
            # Terminal Cost ||z_H - x_ref||^2_P
            cost += 50.0 * np.sum((z[:2] - xd[:2])**2) + 5.0 * np.sum((z[2:] - xd[2:])**2)
            return cost

        def constraints(V_flat):
            V = V_flat.reshape(self.H, 2)
            # Inequality constraints (must be >= 0)
            cons = []
            
            z = np.copy(xhat)
            for j in range(self.H):
                # Apply the Pontryagin strict tracking subtraction
                # z_j \in S \ominus \Omega_{j|k}
                # This translates to: ||p_z - p_obs|| >= d_safe + radius_j
                p_z = np.array([z[0], z[1]])
                p_obs = np.array([self.x_obs[0], self.x_obs[1]])
                dist = np.linalg.norm(p_z - p_obs)
                
                # Constraint: dist - d_safe - tube_radii[j] >= 0
                cons.append(dist - self.d_safe - tube_radii[j])
                
                # Step z
                z_dot = self.plant.f(z) + self.plant.g(z) @ V[j] + self.plant.F(z) @ theta_hat
                z = z + z_dot * self.dt
                
            return np.array(cons)

        # 3. Solve the optimization
        from scipy.optimize import minimize
        V_init = np.zeros(2 * self.H) # Guess zero control
        cons_dict = {'type': 'ineq', 'fun': constraints}
        bnds = [(-self.u_max, self.u_max)] * (2 * self.H)
        
        res = minimize(objective, V_init, method='SLSQP', bounds=bnds, constraints=cons_dict, options={'ftol': 1e-4, 'maxiter': 50})
        
        # 4. Apply resulting control law: u = v_0* + K(xhat - z_0)
        # Since we enforced z_0 = xhat, the proportional tracking error is 0. u = v_0*
        if res.success:
            V_opt = res.x.reshape(self.H, 2)
            u_out = V_opt[0]
            # Double check for NaN before returning
            if np.any(np.isnan(u_out)):
                return np.zeros(2)
            # Clip to actuator limits
            u_norm = np.linalg.norm(u_out)
            if u_norm > self.u_max:
                u_out = (u_out / u_norm) * self.u_max
            return u_out
        else:
            # Fallback if infeasible: apply max braking
            p_x = xhat[0]
            p_y = xhat[1]
            obs_x = self.x_obs[0]
            obs_y = self.x_obs[1]
            vec = np.array([p_x - obs_x, p_y - obs_y])
            if np.linalg.norm(vec) > 1e-3:
                vec = vec / np.linalg.norm(vec)
            return 5.0 * vec
