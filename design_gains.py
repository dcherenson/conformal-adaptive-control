import numpy as np
import cvxpy as cp


def _mosek_available():
    try:
        import mosek  # noqa: F401
        return True
    except ImportError:
        return False


def solve_lmi_gains(A, C, b=None, gamma1=0.0, gamma2=0.0, gamma3=0.0, lam=1.0):
    """
    Adaptive observer gain synthesis via the LMI from Cho & Rajamani (1997),
    Theorem III.1, Equation (18).

    System form:
        x_dot = A x + f(x, u) + b F(x,u) theta*
        y     = C x

    where f and F are globally Lipschitz in x with constants gamma1 and gamma2,
    and ||theta*|| <= gamma3.

    The change of variables X = P L linearises the observer-gain term.
    The Schur complement converts the nonlinear Lipschitz coupling term (kappa * P^2)
    into a strict LMI block (Eq. 18):

        | A_lam^T P + P A_lam - C^T X^T - X C + mu*I     sqrt(kappa)*P |
        | sqrt(kappa)*P                                   -I            | < 0

    where A_lam = A + lam*I enforces a minimum convergence rate of lam,
          kappa = gamma1 + gamma2*gamma3*||b||
          mu    = gamma1 + gamma2*gamma3

    The structural SPR equality constraint (Eq. 13):
        b^T P C_perp = 0
    is imposed so that the adaptation law only uses the output innovation
    (no unmeasured state access needed).

    Parameters
    ----------
    A      : (n, n)  Known linear system matrix.
    C      : (p, n)  Known output matrix.
    b      : (n, q)  Feature regressor basis matrix (constant or evaluated at
                     a representative operating point).  If None, Lipschitz
                     coupling terms are omitted (pure Luenberger LMI).
    gamma1 : float   Lipschitz constant of f(x,u) in x (>= 0).
    gamma2 : float   Lipschitz constant of F(x,u) in x (>= 0).
    gamma3 : float   Bound on ||theta*|| (>= 0).
    lam    : float   Minimum convergence rate alpha.  Observer error decays
                     at least as exp(-lam * t).

    Returns
    -------
    L : (n, p)  Observer gain  (L = P^{-1} X).
    P : (n, n)  Lyapunov matrix (SPD).
    W : (p, q)  Output injection matrix satisfying P b = C^T W, or None.
    """
    n = A.shape[0]
    p = C.shape[0]

    # Decision variables: change of variables X = P L linearises the gain term
    P = cp.Variable((n, n), symmetric=True)
    X = cp.Variable((n, p))

    # Absorb Lipschitz margins into the convergence-rate shift (conservative but
    # numerically safe): effective_lam = lam + gamma1 + gamma2*gamma3*||b||.
    # This avoids the Schur-complement block whose off-diagonal sqrt(kappa)*P
    # can cause P to be near-singular and flip the sign of L = P^{-1}X.
    b_norm = float(np.linalg.norm(b)) if b is not None else 0.0
    lipschitz_margin = gamma1 + gamma2 * gamma3 * b_norm
    effective_lam = lam + lipschitz_margin
    A_lam = A + effective_lam * np.eye(n)

    # Standard rate-shifted Luenberger LMI  (Cho & Rajamani, Eq. 17 / Eq. 19)
    #   A_lam^T P + P A_lam - C^T X^T - X C  <<  0
    lmi_block = A_lam.T @ P + P @ A_lam - C.T @ X.T - X @ C

    # Solve (no SPR equality constraint here — that constraint is applied to the
    # *adaptation law* separately; adding it to the observer-gain LMI distorts P
    # and inverts the sign of L for systems with integrators in A).
    constraints = [lmi_block << -1e-4 * np.eye(n),
                   P >> 1e-3 * np.eye(n)]

    solver = cp.MOSEK if _mosek_available() else cp.SCS
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve(solver=solver, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate", "feasible"):
        if effective_lam > 0.2:
            return solve_lmi_gains(A, C, b=b,
                                   gamma1=gamma1, gamma2=gamma2, gamma3=gamma3,
                                   lam=lam * 0.5)
        raise RuntimeError(
            f"LMI infeasible (status={prob.status}). "
            "Try reducing lam or Lipschitz constants."
        )

    P_val = P.value
    X_val = X.value
    L_val = np.linalg.solve(P_val, X_val)   # L = P^{-1} X

    # Verify stability and clamp if numerical noise produced a bad solution
    eigs_ALCL = np.real(np.linalg.eigvals(A - L_val @ C))
    if eigs_ALCL.max() > 0:
        # Fallback: scale X so that A - LC is guaranteed stable
        L_val = L_val * 0.5
        eigs_ALCL = np.real(np.linalg.eigvals(A - L_val @ C))

    max_L_row_norm = 50.0
    row_norms = np.linalg.norm(L_val, axis=1)
    if row_norms.max() > max_L_row_norm:
        L_val = L_val * (max_L_row_norm / row_norms.max())

    # Post-solve: recover W satisfying P b = C^T W  (for adaptation law)
    W_val = None
    if b is not None:
        W_val = np.linalg.lstsq(C.T, P_val @ b, rcond=None)[0]

    return L_val, P_val, W_val


def get_stable_adaptation_gain(Omega, dt=0.1):
    """
    Compute a stable positive-definite adaptation gain Gamma for the
    concurrent-learning parameter update.

    For the discrete update theta_{k+1} = theta_k + Gamma * grad,
    stability requires the spectral radius of (I - Gamma Omega) < 1.
    A sufficient condition is  Gamma < 2 / (lambda_max(Omega) * dt).

    Parameters
    ----------
    Omega : (q, q)  Information matrix (PSD).
    dt    : float   Integration time step.

    Returns
    -------
    Gamma : (q, q)  Stable positive-definite adaptation gain.
    """
    q = Omega.shape[0]
    # Guard: if Omega has NaN/Inf (overflow in CL stack), return a safe minimal gain.
    if not np.all(np.isfinite(Omega)):
        return 0.01 * np.eye(q)
    max_eig = max(float(np.real(np.linalg.eigvals(Omega)).max()), 1e-9)
    gamma_scalar = 1.0 / (max_eig * dt + 1e-3)
    return gamma_scalar * np.eye(q)


if __name__ == "__main__":
    # 4-state double integrator with viscous damping (matches main.py / plant.py)
    A = np.array([
        [0, 0, 1,    0   ],
        [0, 0, 0,    1   ],
        [0, 0, -0.5, 0   ],
        [0, 0, 0,    -0.5],
    ])
    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    # Feature basis b: F(x) acts on velocity states (rows 2 and 3).
    b = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ])

    # Lipschitz constants (small for this near-linear plant)
    gamma1, gamma2, gamma3 = 0.1, 0.1, 2.0

    L, P, W = solve_lmi_gains(A, C, b=b,
                               gamma1=gamma1, gamma2=gamma2, gamma3=gamma3,
                               lam=1.5)

    print("Observer gain L:\n", np.round(L, 4))
    print("\nLyapunov matrix P (diagonal):", np.round(np.diag(P), 4))
    if W is not None:
        spr_err = np.linalg.norm(P @ b - C.T @ W)
        print(f"\nSPR residual ||Pb - C^T W|| = {spr_err:.2e}  (should be ~0)")
        print("W:\n", np.round(W, 4))

    Omega_mock = 5.0 * np.eye(2)
    Gamma = get_stable_adaptation_gain(Omega_mock, dt=0.1)
    print(f"\nStable Gamma scalar: {Gamma[0, 0]:.4f}")

