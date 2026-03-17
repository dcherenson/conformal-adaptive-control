import numpy as np
from ocp import DriftScoreOCP, InnovationScoreOCP

def test_drift_score_ocp():
    print("Testing DriftScoreOCP...")
    
    # Initialize with alpha=0.1 (10% mis-coverage rate), eta_const=0.5, q_init=1.0
    ocp_drift = DriftScoreOCP(alpha=0.1, eta_const=0.5, q_init=1.0)
    
    assert np.isclose(ocp_drift.get_quantile(), 1.0)
    
    # Mock data dimensions based on main.py state=2, features=2
    # p = 2, m = 2
    X = np.array([[2.0], [3.0]]) # shape (2, 1)
    Z = np.array([[1.0, 0.0], [0.0, 1.0]]) # shape (2, 2)
    theta_hat = np.array([1.5, 2.5]) # shape (2,), gets reshaped internally
    
    # || [2.0, 3.0]^T - [1.5, 2.5]^T || = || [0.5, 0.5]^T || = sqrt(0.5^2 + 0.5^2) = sqrt(0.5) = 0.707
    S_1 = ocp_drift.compute_score(X, Z, theta_hat)
    print(f"Drift Score S_1: {S_1:.4f}")
    assert np.isclose(S_1, np.sqrt(0.5))
    
    # Update q_k
    # S_1 (0.707) <= q_0 (1.0), so indicator = 1
    # eta_1 = 0.5 / sqrt(1) = 0.5
    # q_1 = max(0, 1.0 + 0.5 * (0.1 - 1.0)) = max(0, 1.0 - 0.45) = 0.55
    q_1 = ocp_drift.update(S_1)
    print(f"Quantile q_1: {q_1:.4f}")
    assert np.isclose(q_1, 0.55)
    
    # Step 2: High error scenario
    X2 = np.array([[5.0], [5.0]])
    S_2 = ocp_drift.compute_score(X2, Z, theta_hat)
    print(f"Drift Score S_2: {S_2:.4f}")
    
    # S_2 (~4.3) > q_1 (0.55), so indicator = 0
    # eta_2 = 0.5 / sqrt(2) = 0.3535
    # q_2 = max(0, 0.55 + 0.3535 * (0.1 - 0)) = 0.55 + 0.03535 = 0.58535
    q_2 = ocp_drift.update(S_2)
    print(f"Quantile q_2: {q_2:.4f}")
    assert np.isclose(q_2, 0.55 + (0.5 / np.sqrt(2)) * 0.1)

def test_innovation_score_ocp():
    print("\\nTesting InnovationScoreOCP...")
    ocp_inn = InnovationScoreOCP(alpha=0.05, eta_const=0.1, q_init=0.0)
    
    y = np.array([1.0, 2.0])
    y_hat = np.array([1.1, 1.9])
    
    # || [1.0, 2.0] - [1.1, 1.9] || = || [-0.1, 0.1] || = sqrt(0.01 + 0.01) = sqrt(0.02) = 0.1414
    S_1 = ocp_inn.compute_score(y, y_hat)
    print(f"Innovation Score S_1: {S_1:.4f}")
    assert np.isclose(S_1, np.sqrt(0.02))
    
    # Update q_k
    # S_1 (0.1414) > q_0 (0.0), so indicator = 0
    # eta_1 = 0.1 / sqrt(1) = 0.1
    # q_1 = max(0, 0.0 + 0.1 * (0.05 - 0)) = 0.005
    q_1 = ocp_inn.update(S_1)
    print(f"Quantile q_1: {q_1:.4f}")
    assert np.isclose(q_1, 0.005)

if __name__ == "__main__":
    test_drift_score_ocp()
    test_innovation_score_ocp()
    print("\\nAll OCP tests passed successfully!")
