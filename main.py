import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.switch_backend('Agg')

# Architecture Modules
from plant import Plant
from ocp import DriftScoreOCP
from controller import DynamicTubeMPC

# SSML Modules
from ssml import get_or_train_model, flatten_params, assign_params, compute_jacobian, spectral_normalization_clip, get_reference

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Simulation Variables
    t = 0.0
    dt = 0.05   # slightly coarser than ssml.py to ensure 3D MPC completes in reasonable time (H=8 becomes 0.4s horizon)
    t_end = 15.0  # Run for 15 seconds to see tracking and obstacle avoidance

    # Initialize Architecture Components
    sys_plant = Plant()
    
    # Using DTMPC with full-state feedback and 3D Quadcopter
    sys_controller = DynamicTubeMPC(plant=sys_plant, x_obs=np.array([0.0, 0.0, 1.0]), d_safe=0.5, H=8, dt=dt)
    
    # OCP for Drift bounds
    ocp_drift = DriftScoreOCP(alpha=0.1, eta_const=0.1, q_init=0.1)

    # SSML Network Initialization
    model = get_or_train_model()
    theta_0_flat = flatten_params(model).clone().detach()
    theta_flat = flatten_params(model).clone().detach()
    gamma_lr = 0.01
    lambd = 0.1

    # State variables (6 elements)
    x = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0]) # start at origin
    u = np.array([0.0, 0.0, 9.81 * sys_plant.m])

    # Tracking arrays
    x_plotting = [np.copy(x)]
    t_plotting = [t]
    
    # OCP Tracking
    drift_scores = [0.0]
    drift_quantiles = [ocp_drift.get_quantile()]
    dist_bound_plotting = [ocp_drift.get_quantile() / sys_controller.T_horizon]

    # Target plotting
    xd_plotting = []

    print("Running SSML DTMPC 3D Flight Simulation...")

    # Simulation Loop
    while t <= t_end:
        # High Level Objective Output from SSML (Figure-8)
        p_d, v_d, a_d = get_reference(t)
        xd = np.concatenate([p_d, v_d])
        xd_plotting.append(np.copy(xd))

        # Controller computes 3D control force
        u_old = np.copy(u)
        u = sys_controller.compute_u(x, xd, drift_quantiles[-1], model_nn=model)

        x_old = np.copy(x)

        # Plant integrates one step forward with true wind disturbance
        x = sys_plant.step(x_old, u, t, dt)

        # Measure acceleration finite difference
        v_old = x_old[3:6]
        v_new = x[3:6]
        dv_measured = (v_new - v_old) / dt
        
        # Nominal acceleration (without wind/mismatch)
        dv_nominal = np.array([0, 0, -9.81]) + u_old / sys_plant.m
        
        # True acceleration mismatch
        true_mismatch = dv_measured - dv_nominal

        # Neural Network prediction
        x_in = np.concatenate((x_old, u_old))
        with torch.no_grad():
            f_nn_full = model(torch.tensor(x_in, dtype=torch.float32)).numpy()
        pred_mismatch = f_nn_full[3:6]
        
        # Error between true and prediction
        error_acc = true_mismatch - pred_mismatch
        
        # OCP Updates
        S_drift = np.linalg.norm(error_acc)
        q_drift = ocp_drift.update(S_drift)

        # Adaptation updates learns theta online
        # Lift 3D error to 6D output error for SSML net: [0, 0, 0, ex, ey, ez]
        output_error = np.concatenate((np.zeros(3), error_acc))
        J = compute_jacobian(model, x_in).detach().numpy()
        
        theta_dot = gamma_lr * np.dot(J.T, output_error) - lambd * (
            theta_flat.detach().numpy() - theta_0_flat.numpy()
        )
        theta_flat = theta_flat + torch.tensor(theta_dot, dtype=torch.float32) * dt
        
        assign_params(model, theta_flat)
        spectral_normalization_clip(model)

        # Append variables for graphing
        t += dt
        x_plotting.append(np.copy(x))
        t_plotting.append(t)
        drift_scores.append(S_drift)
        drift_quantiles.append(q_drift)
        dist_bound_plotting.append(q_drift / sys_controller.T_horizon)

    # Append terminal target since while loop offsets it by 1
    p_d, v_d, a_d = get_reference(t)
    xd_plotting.append(np.concatenate([p_d, v_d]))

    # Convert lists to arrays for editing plots
    x_history = np.array(x_plotting)
    t_history = np.array(t_plotting)
    xd_history = np.array(xd_plotting)
    dist_bound_history = np.array(dist_bound_plotting)

    # Subplots
    fig, (ax1, ax5) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot x vs time
    ax1.plot(t_history, x_history[:, 0], 'b-', label='x[0] px', linewidth=2)
    ax1.plot(t_history, x_history[:, 1], 'r-', label='x[1] py', linewidth=2)
    ax1.plot(t_history, x_history[:, 2], 'g-', label='x[2] pz', linewidth=2)
    ax1.plot(t_history, xd_history[:, 0], 'b--', alpha=0.5)
    ax1.plot(t_history, xd_history[:, 1], 'r--', alpha=0.5)
    ax1.plot(t_history, xd_history[:, 2], 'g--', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Plant 3D State Tracking vs Time')
    ax1.legend()
    ax1.grid(True)

    # Plot Drift Score vs Quantile
    ax5.plot(t_history, drift_scores, 'c-', label='Drift Score $S_{drift,k}$', linewidth=2)
    ax5.plot(t_history, drift_quantiles, 'k--', label='Drift Quantile $q_{drift,k}$', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Value')
    ax5.set_title('Neural Acceleration Mismatch Score vs Quantile Threshold')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()
    plt.savefig('architecture_state_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to architecture_state_comparison.png")

    # --- 3D Position vs Time Plot ---
    fig_pos = plt.figure(figsize=(8, 6))
    ax_pos = fig_pos.add_subplot(111, projection='3d')
    ax_pos.plot(x_history[:, 0], x_history[:, 1], x_history[:, 2], label='Trajectory', color='blue')
    ax_pos.plot(xd_history[:, 0], xd_history[:, 1], xd_history[:, 2], label='Target (Fig-8)', color='orange', linestyle='--')
    
    # Draw spherical obstacle
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    obs_x = sys_controller.x_obs[0] + sys_controller.d_safe * np.cos(u_sphere) * np.sin(v_sphere)
    obs_y = sys_controller.x_obs[1] + sys_controller.d_safe * np.sin(u_sphere) * np.sin(v_sphere)
    obs_z = sys_controller.x_obs[2] + sys_controller.d_safe * np.cos(v_sphere)
    ax_pos.plot_surface(obs_x, obs_y, obs_z, color='red', alpha=0.3)
    
    ax_pos.set_xlabel('X Position')
    ax_pos.set_ylabel('Y Position')
    ax_pos.set_zlabel('Z Position')
    ax_pos.set_title('3D Trajectory Tracking & Obstacle Avoidance')
    ax_pos.legend()
    fig_pos.tight_layout()
    fig_pos.savefig('pos_vs_time.png', dpi=150, bbox_inches='tight')
    print("Plot saved to pos_vs_time.png")

    # --- OCP Bounds Plot ---
    fig_bounds, ax_b1 = plt.subplots(1, 1, figsize=(10, 4))
    ax_b1.plot(t_history, drift_scores, 'g-', label=r'True Mismatch Norm', linewidth=2)
    ax_b1.plot(t_history, dist_bound_history, 'k--', label=r'OCP Disturbance Bound $E_k$', linewidth=2)
    ax_b1.set_xlabel('Time (s)')
    ax_b1.set_ylabel('Acceleration Mismatch Norm')
    ax_b1.set_title('OCP-Derived Estimated Disturbance Bound vs True Disturbance')
    ax_b1.legend()
    ax_b1.grid(True)
    fig_bounds.tight_layout()
    fig_bounds.savefig('ocp_bounds_vs_true.png', dpi=150, bbox_inches='tight')
    print("Plot saved to ocp_bounds_vs_true.png")
    
    # Validation Check
    distances = np.linalg.norm(x_history[:,0:3] - sys_controller.x_obs, axis=1)
    print(f"Minimum distance to obstacle center: {np.min(distances):.3f} (Constraint: {sys_controller.d_safe:.3f})")

if __name__ == '__main__':
    main()
