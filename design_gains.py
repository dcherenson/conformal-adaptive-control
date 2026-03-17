import numpy as np
import cvxpy as cp

def solve_lmi_gains(A, C, alpha=2.0):
    """
    Solves A^T P + P A - C^T Y^T - Y C <= -2 * alpha * P
    L = P^-1 Y
    """
    n = A.shape[0]
    m = C.shape[0]
    
    P = cp.Variable((n, n), PSD=True)
    Y = cp.Variable((n, m))
    
    # LMI for stability with decay rate alpha
    # (A + alpha*I)^T P + P (A + alpha*I) - C^T Y^T - Y C <= 0
    A_shifted = A + alpha * np.eye(n)
    lmi = A_shifted.T @ P + P @ A_shifted - C.T @ Y.T - Y @ C
    
    prob = cp.Problem(cp.Minimize(cp.trace(P)), [lmi << 0, P >> np.eye(n)])
    prob.solve(solver=cp.SCS)
    
    if prob.status not in ["feasible", "optimal"]:
        # Fallback to slower alpha if infeasible
        return solve_lmi_gains(A, C, alpha=alpha*0.5)
        
    P_val = P.value
    Y_val = Y.value
    L = np.linalg.inv(P_val) @ Y_val
    return L, P_val

def get_stable_adaptation_gain(Omega, dt=0.1):
    """
    Ensures spectral radius of discrete adaptation loop is < 1.
    For discrete CL: theta_k+1 = theta_k - Gamma * Omega_k * theta_k
    => stability requires I - Gamma @ Omega has eigenvalues in unit circle.
    Approx: Gamma < 2 * inv(Omega) / dt or similar.
    """
    # Simple heuristic for Gamma stability: 
    # eig(Gamma @ Omega) < 2 / dt (roughly)
    # Since Omega is positive definite, we can use a scalar gamma or a matrix.
    max_eig_omega = np.max(np.real(np.linalg.eigvals(Omega)))
    gamma_scalar = 1.0 / (max_eig_omega * dt + 1e-3)
    return gamma_scalar * np.eye(Omega.shape[0])

if __name__ == "__main__":
    # Nominal linearized plant for Double Integrator with theta_true[0]=0.5 friction
    A = np.array([
        [0, 0, 1,    0],
        [0, 0, 0,    1],
        [0, 0, -0.5, 0],
        [0, 0, 0,    -0.5]
    ])
    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    L_opt, P_opt = solve_lmi_gains(A, C, alpha=1.0)
    print("Optimal L:\n", L_opt)
    
    # Mock Omega for PE check
    Omega_mock = 5.0 * np.eye(2)
    Gamma_opt = get_stable_adaptation_gain(Omega_mock)
    print("Stable Gamma Scalar:\n", Gamma_opt[0,0])
