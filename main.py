import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
plt.switch_backend('Agg')

# Architecture Modules
from plant import Plant
from ocp import DriftScoreOCP
from controller import DynamicTubeMPC

# SSML Modules
from ssml import get_or_train_model, flatten_params, assign_params, compute_jacobian, compute_ssml_input_lipschitz, spectral_normalization_clip, get_reference

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Simulation Variables
    t = 0.0
    dt = 0.05   # slightly coarser than ssml.py to ensure 3D MPC completes in reasonable time (H=8 becomes 0.4s horizon)
    t_end = 5.0  # Point-to-point navigation time

    # Initialize Architecture Components
    sys_plant = Plant(spatial_mode=True)
    
    # Define Obstacles (Large+small pair creating narrow gap, small further along)
    obstacles = [
        {'pos': np.array([3.0, -1.2, 1.0]), 'r': 0.8},  # Large obstacle, below path
        {'pos': np.array([3.0,  0.7, 1.0]), 'r': 0.3},  # Small obstacle above path (gap ~0.6m)
        {'pos': np.array([6.5,  0.0, 1.0]), 'r': 0.5},  # Small obstacle further along path
    ]
    x_goal = np.array([8.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Using DTMPC with full-state feedback and 3D Quadcopter
    sys_controller = DynamicTubeMPC(plant=sys_plant, obstacles=obstacles, H=8, dt=dt)
    
    # OCP for Drift bounds
    ocp_drift = DriftScoreOCP(alpha=0.1, eta_const=0.1, q_init=0.1)

    # SSML Network Initialization
    model = get_or_train_model()
    sys_plant.spatial_mode = True  # Enable spatial distribution shift during runtime
    theta_0_flat = flatten_params(model).clone().detach()
    theta_flat = flatten_params(model).clone().detach()
    # Adaptation Parameters

    gamma_lr = 0.2
    lambd = 0.1

    # State variables (8 elements: pos, vel, roll, pitch)
    x = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # start at offset
    u = np.array([0.0, 0.0, 9.81 * sys_plant.m])
    u_old = np.copy(u)

    # Tracking arrays
    x_plotting = [np.copy(x)]
    t_plotting = [t]
    Ld_plotting = [0.0]
    z_pred_plotting = [np.tile(x, (sys_controller.H, 1))]
    phi_pred_plotting = [np.tile(sys_controller.Phi, sys_controller.H)]

    
    theta_history_norm = [0.0]
    
    # Rolling Windows (10 time steps long)
    import collections
    T_window = 5
    past_states = collections.deque(maxlen=T_window)
    past_v = collections.deque(maxlen=T_window)
    
    # OCP Tracking
    drift_scores = [0.0]
    drift_quantiles = [0.1]
    dist_bound_plotting = [0.1] # Initial guess for plotter

    # Target plotting
    xd_plotting = []

    tube_plotting = [sys_controller.Phi]
    theta_plotting = [theta_flat.detach().numpy().copy()]
    residual_plotting = [0.0]  # Initial residual is 0.0 before evaluating data


    print("Running SSML DTMPC 3D Flight Simulation...")

    # Simulation Loop
    while t <= t_end:
        t_start = time.time()
        # Point-to-Point Goal
        xd = x_goal
        # Controller computes 3D control force and prediction horizon
        u_old = np.copy(u)
        u, z_pred, phi_pred, success = sys_controller.compute_u(x, xd, drift_quantiles[-1], model_nn=model)
        if not success:
            print(f"Solver failed at t={t:.2f}s! Ending simulation early.")
            break
        z_pred_plotting.append(z_pred)
        phi_pred_plotting.append(phi_pred)
        xd_plotting.append(np.copy(xd))

        x_old = np.copy(x)

        # Plant integrates one step forward with true wind disturbance
        x = sys_plant.step(x_old, u, t, dt)

        # True dynamics derivative: f_true(x, u) + d
        x_dot_true = sys_plant.dynamics(t, x_old, u_old)

        # Neural Network prediction
        x_in = np.concatenate((x_old[3:6], x_old[6:8]))
        with torch.no_grad():
            f_nn_acc = model(torch.tensor(x_in, dtype=torch.float32)).numpy()
        
        # Predicted dynamics model: f_nom(x, u) + F_nn(x, u, theta_hat)
        x_dot_pred = sys_plant.f(x_old) + sys_plant.g_mat(x_old) @ u_old + np.concatenate([np.zeros(3), f_nn_acc, np.zeros(2)])
        
        # True continuous residual between predicted dynamics and true dynamics
        error_full = x_dot_true - x_dot_pred
        
        # Error acceleration extracted for adaptation
        error_acc = error_full[3:6]
        
        # Compute predicted full-state derivative v_current = x_dot_pred
        v_current = x_dot_pred

        # Update Rolling Windows
        past_states.append(np.copy(x_old))
        past_v.append(np.copy(v_current))

        # OCP Updates: Integral Drift Score over the rolling window (O(N) vectorized)
        L = len(past_states)
        S_drift = 0.0
        if L > 0:
            x_buffer = np.array(past_states)
            v_buffer = np.array(past_v)
            
            # 1. Reverse the historical buffers (time goes backward from k-1 to k-N)
            v_rev = np.flip(v_buffer, axis=0)
            x_rev = np.flip(x_buffer, axis=0)
            
            # 2. Compute the backward integrals using a cumulative sum
            integrals = np.cumsum(v_rev, axis=0) * dt
            
            # 3. Compute the full state prediction error for all sub-intervals
            prediction_errors = x - x_rev - integrals
            
            # 4. Take the L2 norm of the error vector at each time step
            error_norms = np.linalg.norm(prediction_errors, axis=1)
            
            # 5. Extract the supremum score
            S_drift = np.max(error_norms)
            
        q_drift = ocp_drift.update(S_drift)

        # Lipschitz Estimation for discrepancy d(t)
        # L_d = (Ltrue_x + Lnom_x + Lnn_x)*||xdot|| + (Ltrue_u + Lnom_u + Lnn_u)*||udot||
        Lnn_x, Lnn_u = compute_ssml_input_lipschitz(model, x_in)
        norm_xdot = np.linalg.norm(x_dot_true)
        norm_udot = np.linalg.norm((u - u_old) / dt)
        
        L_d = (sys_plant.L_true_x + sys_plant.L_nom_x + Lnn_x) * norm_xdot + \
              (sys_plant.L_true_u + sys_plant.L_nom_u + Lnn_u) * norm_udot
        L_d = max(L_d, 0.1) # Minimum L_d to avoid zero bounds
        
        # Calculate new disturbance bound using triangle piece-wise logic
        # T is the window length (10 timesteps)
        dist_bound = ocp_drift.get_dist_bound_from_quantile(q_drift, T=T_window*dt, L_d=L_d)

        # Adaptation updates learns theta online
        J = compute_jacobian(model, x_in).detach().numpy()
        
        theta_dot = gamma_lr * np.dot(J.T, error_acc) - lambd * (
            theta_flat.detach().numpy() - theta_0_flat.numpy()
        )
        theta_flat = theta_flat + torch.tensor(theta_dot, dtype=torch.float32) * dt
        
        assign_params(model, theta_flat)
        spectral_normalization_clip(model)

        param_change = np.linalg.norm(theta_flat.detach().numpy() - theta_0_flat.numpy())
        theta_history_norm.append(param_change)

        # Append variables for graphing
        t += dt
        u_old = np.copy(u)
        x_plotting.append(np.copy(x))
        t_plotting.append(t)
        drift_scores.append(S_drift)
        drift_quantiles.append(q_drift)
        dist_bound_plotting.append(dist_bound)
        tube_plotting.append(sys_controller.Phi)
        theta_plotting.append(theta_flat.detach().numpy().copy())
        residual_plotting.append(np.linalg.norm(error_acc))
        Ld_plotting.append(L_d)

        # Collision check
        stop_flag = False
        for obs in obstacles:
            if np.linalg.norm(x[:3] - obs['pos']) < obs['r']:
                print(f"COLLISION at t={t:.2f}s! Penetrated obstacle at {obs['pos']}")
                stop_flag = True
                break
        # Stop when close enough to goal
        goal_dist = np.linalg.norm(x[:3] - x_goal[:3])
        if goal_dist < 0.1:
            print(f"Goal reached at t={t:.2f}s! Distance: {goal_dist:.3f}m")
            stop_flag = True
        if stop_flag:
            break

        # every 10 time steps, print out how long it took the run the loop
        t_end_timer = time.time()
        if int(t/dt) % 10 == 0:
            print(f"t: {t:.2f}, 10 steps took {t_end_timer - t_start:.4f} seconds")
            # print(f"t: {t:.2f}, x: {x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, v: {x[3]:.2f}, {x[4]:.2f}, {x[5]:.2f}, u: {u[0]:.2f}, {u[1]:.2f}, {u[2]:.2f}, S_drift: {S_drift:.2f}, q_drift: {q_drift:.2f}, dist_bound: {dist_bound_plotting[-1]:.2f}")

    # Append terminal target since while loop offsets it by 1
    xd_plotting.append(np.copy(x_goal))

    # Convert lists to arrays for editing plots
    x_history = np.array(x_plotting)
    t_history = np.array(t_plotting)
    xd_history = np.array(xd_plotting)
    z_pred_hist = np.array(z_pred_plotting)
    phi_pred_hist = np.array(phi_pred_plotting)
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

    # 3D position plot
    fig_pos = plt.figure(figsize=(8, 6))
    ax_pos = fig_pos.add_subplot(111, projection='3d')
    ax_pos.plot(x_history[:, 0], x_history[:, 1], x_history[:, 2], label='Trajectory', color='blue')
    ax_pos.scatter(x_goal[0], x_goal[1], x_goal[2], color='gold', marker='*', s=200, label='Goal')
    
    # Draw spherical obstacles
    for obs in obstacles:
        u_sphere, v_sphere = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
        ox = obs['pos'][0] + obs['r'] * np.cos(u_sphere) * np.sin(v_sphere)
        oy = obs['pos'][1] + obs['r'] * np.sin(u_sphere) * np.sin(v_sphere)
        oz = obs['pos'][2] + obs['r'] * np.cos(v_sphere)
        ax_pos.plot_surface(ox, oy, oz, color='red', alpha=0.2)
    
    ax_pos.set_xlabel('X Position')
    ax_pos.set_ylabel('Y Position')
    ax_pos.set_zlabel('Z Position')
    ax_pos.set_title('3D Trajectory Tracking & Obstacle Avoidance')
    ax_pos.legend()
    
    # Set equal aspect ratio for 3D plot
    x_lim = ax_pos.get_xlim()
    y_lim = ax_pos.get_ylim()
    z_lim = ax_pos.get_zlim()
    max_range = np.array([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]).max() / 2.0
    mid_x = (x_lim[1]+x_lim[0]) * 0.5
    mid_y = (y_lim[1]+y_lim[0]) * 0.5
    mid_z = (z_lim[1]+z_lim[0]) * 0.5
    ax_pos.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_pos.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_pos.set_zlim(mid_z - max_range, mid_z + max_range)

    fig_pos.tight_layout()
    fig_pos.savefig('pos_vs_time.png', dpi=150, bbox_inches='tight')
    print("Plot saved to pos_vs_time.png")

    # --- Unified OCP Bounds & Residuals Plot ---
    residual_history = np.array(residual_plotting)
    drift_history = np.array(drift_scores)
    fig_bounds, ax_b1 = plt.subplots(1, 1, figsize=(10, 5))
    ax_b1.plot(t_history, residual_history, 'r-', label=r'Instantaneous Residual $\|\tilde F + \delta\|$', alpha=0.6)
    ax_b1.plot(t_history, dist_bound_history, 'k--', label=r'OCP Disturbance Bound $E_k$', linewidth=2)
    ax_b1.set_xlabel('Time (s)')
    ax_b1.set_ylabel('Dynamics Mismatch (m/s^2)')
    ax_b1.set_title('Unified OCP Bounds, Drift Scores, and Dynamics Residuals')
    ax_b1.legend()
    ax_b1.grid(True)
    fig_bounds.tight_layout()
    fig_bounds.savefig('ocp_bounds_vs_true.png', dpi=150, bbox_inches='tight')
    print("Plot saved to ocp_bounds_vs_true.png")
    
    # --- Tube Size Plot ---
    tube_history = np.array(tube_plotting)
    fig_tube, ax_tube = plt.subplots(1, 1, figsize=(10, 4))
    ax_tube.plot(t_history, tube_history, 'm-', label=r'Tube Size $\Phi$', linewidth=2)
    ax_tube.set_xlabel('Time (s)')
    ax_tube.set_ylabel(r'Boundary $\Phi$ (m)')
    ax_tube.set_title('Dynamic Tube Size Over Time')
    ax_tube.legend()
    ax_tube.grid(True)
    fig_tube.tight_layout()
    fig_tube.savefig('tube_size_vs_time.png', dpi=150, bbox_inches='tight')
    print("Plot saved to tube_size_vs_time.png")

    # --- Neural Network Parameters Plot ---
    theta_history = np.array(theta_plotting)
    fig_theta, ax_theta = plt.subplots(1, 1, figsize=(10, 6))
    # Downsample lines if there are too many, or just plot them with low alpha
    ax_theta.plot(t_history, theta_history, alpha=0.1, linewidth=1)
    ax_theta.set_xlabel('Time (s)')
    ax_theta.set_ylabel(r'NN Weight Values $\theta$')
    ax_theta.set_title('Neural Network Parameters Evolution')
    ax_theta.grid(True)
    fig_theta.tight_layout()
    fig_theta.savefig('nn_params_vs_time.png', dpi=150, bbox_inches='tight')
    print("Plot saved to nn_params_vs_time.png")

    # --- Parameter Deviation Plot ---
    fig_norm, ax_norm = plt.subplots(1, 1, figsize=(10, 4))
    ax_norm.plot(t_history, theta_history_norm, 'b-', label=r'$\|\theta - \theta_0\|$')
    ax_norm.set_xlabel('Time (s)')
    ax_norm.set_ylabel('Parameter Deviation')
    ax_norm.set_title('Neural Network Parameter Adaptation Progress')
    ax_norm.legend()
    ax_norm.grid(True)
    fig_norm.savefig('nn_param_deviation.png', dpi=150, bbox_inches='tight')
    print("Plot saved to nn_param_deviation.png")
    

    
    # --- Top-Down Tube Plot ---
    fig_top, ax_top = plt.subplots(1, 1, figsize=(8, 8))
    # Plot Trajectory
    ax_top.plot(xd_history[:, 0], xd_history[:, 1], 'k--', label='Reference Path', alpha=0.5)
    ax_top.plot(x_history[:, 0], x_history[:, 1], 'b-', label='Quadcopter Trajectory', linewidth=2)
    
    # Plot obstacles
    for obs in obstacles:
        obs_circle = plt.Circle((obs['pos'][0], obs['pos'][1]), obs['r'], color='r', alpha=0.2)
        ax_top.add_patch(obs_circle)
    ax_top.scatter(x_goal[0], x_goal[1], color='gold', marker='*', s=150, label='Goal')

    # Plot Tube Boundary (Downsampled)
    for i in range(0, len(t_history), 5):  # Every 5 steps
        circ = plt.Circle((x_history[i, 0], x_history[i, 1]), tube_history[i], color='b', fill=False, alpha=0.2)
        ax_top.add_patch(circ)
        
    ax_top.set_xlabel('X Position (m)')
    ax_top.set_ylabel('Y Position (m)')
    ax_top.set_title('Top-Down View: Trajectory and Dynamic Tube Corridor')
    ax_top.legend()
    ax_top.grid(True)
    ax_top.set_aspect('equal')
    fig_top.tight_layout()
    fig_top.savefig('top_down_tube.png', dpi=150, bbox_inches='tight')
    print("Plot saved to top_down_tube.png")

    # --- Lipschitz Constant Plot ---
    Ld_history = np.array(Ld_plotting)
    fig_ld, ax_ld = plt.subplots(1, 1, figsize=(10, 4))
    ax_ld.plot(t_history, Ld_history, 'g-', label=r'Estimated $L_d$', linewidth=2)
    ax_ld.set_xlabel('Time (s)')
    ax_ld.set_ylabel('Lipschitz Constant')
    ax_ld.set_title('Dynamics Discrepancy Lipschitz Constant Evolution')
    ax_ld.legend()
    ax_ld.grid(True)
    fig_ld.tight_layout()
    fig_ld.savefig('lipschitz_vs_time.png', dpi=150, bbox_inches='tight')
    print("Plot saved to lipschitz_vs_time.png")
    
    # Validation Check (distance to nearest obstacle)
    min_obs_dist = 100.0
    for obs in obstacles:
        dists = np.linalg.norm(x_history[:, 0:3] - obs['pos'], axis=1) - obs['r']
        min_obs_dist = min(min_obs_dist, np.min(dists))
    print(f"Minimum distance to any obstacle surface: {min_obs_dist:.3f}")

    # --- Animation ---
    print("Generating Animation...")
    fig_anim = plt.figure(figsize=(8, 6))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    
    # Plot target trajectory
    line_ref, = ax_anim.plot(xd_history[:, 0], xd_history[:, 1], xd_history[:, 2], label='Reference', color='orange', linestyle='--')
    line_true, = ax_anim.plot([], [], [], 'b-', label='Quadcopter Path', linewidth=2)
    scatter_true = ax_anim.scatter([], [], [], color='blue', s=50)
    
    line_pred, = ax_anim.plot([], [], [], 'g--', label='MPC Prediction', linewidth=1)
    scatter_pred = ax_anim.scatter([], [], [], color='cyan', alpha=0.3, label='Tube Envelope')
    

    
    ax_anim.scatter(x_goal[0], x_goal[1], x_goal[2], color='gold', marker='*', s=100)
    
    def init():
        line_true.set_data([], [])
        line_true.set_3d_properties([])
        scatter_true._offsets3d = ([], [], [])
        line_pred.set_data([], [])
        line_pred.set_3d_properties([])
        scatter_pred._offsets3d = ([], [], [])
        return scatter_true, line_true, line_ref, line_pred, scatter_pred
        
    def update_graph(k):
        scatter_true._offsets3d = (x_history[k:k+1, 0], x_history[k:k+1, 1], x_history[k:k+1, 2])
        line_true.set_data(x_history[:k+1, 0], x_history[:k+1, 1])
        line_true.set_3d_properties(x_history[:k+1, 2])
        
        line_ref.set_data(xd_history[:k+1, 0], xd_history[:k+1, 1])
        line_ref.set_3d_properties(xd_history[:k+1, 2])
        
        # Update predicting horizon
        line_pred.set_data(z_pred_hist[k, :, 0], z_pred_hist[k, :, 1])
        line_pred.set_3d_properties(z_pred_hist[k, :, 2])
        
        scatter_pred._offsets3d = (z_pred_hist[k, :, 0], z_pred_hist[k, :, 1], z_pred_hist[k, :, 2])
        # scale scatter size dynamically from tube bound phi
        scatter_pred.set_sizes((phi_pred_hist[k] * 100) ** 2)

        sys_plant.Phi = tube_history[k] # update for title if needed
        ax_anim.set_title(rf'DTMPC 3D Flight | Time: {t_history[k]:.1f}s | $\Phi$: {tube_history[k]:.2f}m')
        return scatter_true, line_true, line_ref, line_pred, scatter_pred

    # Draw spherical obstacles
    for obs in obstacles:
        u_sphere_anim, v_sphere_anim = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
        ox = obs['pos'][0] + obs['r'] * np.cos(u_sphere_anim) * np.sin(v_sphere_anim)
        oy = obs['pos'][1] + obs['r'] * np.sin(u_sphere_anim) * np.sin(v_sphere_anim)
        oz = obs['pos'][2] + obs['r'] * np.cos(v_sphere_anim)
        ax_anim.plot_surface(ox, oy, oz, color='red', alpha=1.0)
        
    # Corridor-style rectangle with equal physical scale (1m = 1m on all axes)
    x_lo, x_hi = np.min(x_history[:,0])-0.5, np.max(x_history[:,0])+0.5
    y_lo, y_hi = -2.5, 2.5
    z_lo, z_hi = 0.0, 2.0
    ax_anim.set_xlim([x_lo, x_hi])
    ax_anim.set_ylim([y_lo, y_hi])
    ax_anim.set_zlim([z_lo, z_hi])
    # Equal physical scale: box is stretched proportionally to the data ranges
    ax_anim.set_box_aspect([x_hi-x_lo, y_hi-y_lo, z_hi-z_lo])
    ax_anim.set_xlabel('X')
    ax_anim.set_ylabel('Y')
    ax_anim.set_zlabel('Z')
    ax_anim.legend()
    
    ani = animation.FuncAnimation(fig_anim, update_graph, init_func=init, frames=len(x_history), interval=50, blit=False)
    ani.save('scenario.gif', writer='pillow', fps=20)
    print("Animation saved to scenario.gif")

if __name__ == '__main__':
    main()
