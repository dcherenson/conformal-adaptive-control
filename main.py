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
    np.random.seed(1337)
    torch.manual_seed(1337)

    # Simulation Variables
    t = 0.0
    dt_sim = 0.01
    dt_mpc = 0.05
    n_substeps = int(dt_mpc / dt_sim) # 5
    t_end = 10.0  # Point-to-point navigation time

    # Initialize Architecture Components
    sys_plant = Plant(spatial_mode=True)
    
    # Define Obstacles (Large+small pair creating narrow gap, small further along)
    obstacles = [
        {'pos': np.array([3.0, 0.4, 1.0]), 'r': 0.7},  # Large obstacle, below path
        {'pos': np.array([3.0,  -0.9, 1.0]), 'r': 0.3},  # Small obstacle above path (gap ~0.6m)
        {'pos': np.array([5.5,  0.0, 1.0]), 'r': 0.4},  # Small obstacle further along path
    ]
    x_goal = np.array([7.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_radius = 0.3  # Goal region: sphere of this radius around x_goal (m)

    # Using DTMPC with full-state feedback and 3D Quadcopter
    mpc_horizon = 10
    sys_controller = DynamicTubeMPC(plant=sys_plant, obstacles=obstacles, H=mpc_horizon, dt=dt_mpc)
    
    # OCP for Drift bounds
    ocp_integral = DriftScoreOCP(alpha=0.1, eta_const=0.1, q_init=0.1)
    ddot_bound = 3.0

    # SSML Network Initialization
    model = get_or_train_model()
    theta_0_flat = flatten_params(model).clone().detach()
    theta_flat = flatten_params(model).clone().detach()
    # Adaptation Parameters

    gamma_lr = 0.5
    lambd = 0.1

    # State variables (8 elements: pos, vel, roll, pitch)
    x = np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # start at offset
    u = np.array([0.0, 0.0, 9.81 * sys_plant.m])
    u_old = np.copy(u)

    # Tracking arrays
    x_plotting = [np.copy(x)]
    t_plotting = [t]
    Ld_plotting = [0.0]
    z_pred_plotting = [np.tile(x, (sys_controller.H, 1))]
    phi_pred_plotting = [np.tile(sys_controller.Phi, sys_controller.H)]

    
    theta_history_norm = [0.0]
    
    # Rolling Windows (Maintain same physical time window as before: 0.5s)
    import collections
    T_window = int(mpc_horizon * dt_mpc / dt_sim) # 50 steps at 100 Hz
    past_states = collections.deque(maxlen=T_window)
    past_f_nom = collections.deque(maxlen=T_window)
    past_v = collections.deque(maxlen=T_window)
    past_theta_dot_norm = collections.deque(maxlen=T_window)
    
    # OCP Tracking
    drift_scores_I = [0.0]
    drift_quantiles_I = [ocp_integral.get_quantile()]
    dist_bound_plotting = [ocp_integral.get_quantile()]

    # Target plotting
    xd_plotting = []

    tube_plotting = [sys_controller.Phi]
    theta_plotting = [theta_flat.detach().numpy().copy()]
    residual_plotting = [0.0]  # Initial residual is 0.0 before evaluating data

    # Coverage Tracking
    correct_bounds_count = 0
    total_steps_count = 0


    print("Running SSML DTMPC 3D Flight Simulation...")
    mpc_step_counter = 0
    theta_dot_norm_val = 0.0 # Initial adaptation rate

    # Simulation Loop
    xd = x_goal
    t_sim_block_start = time.time()
    while t <= t_end:
        t_start = time.time()
        # 1. Update MPC at 20 Hz
        if int(t/dt_sim + 0.5) % n_substeps == 0:
            mpc_step_counter += 1
            # Controller computes 3D control force and prediction horizon
            u_old_mpc = np.copy(u) # store for possible reference
            t_solve_start = time.time()
            u, z_pred, phi_pred, success = sys_controller.compute_u(x, xd, dist_bound_plotting[-1], model_nn=model)
            solve_time = time.time() - t_solve_start
            
            if mpc_step_counter % 10 == 0:
                 t_now = time.time()
                 sim_block_time = t_now - t_sim_block_start
                 print(f"t: {t:.2f}s | MPC Solve: {solve_time:.4f}s | Last 50 steps: {sim_block_time:.4f}s")
                 t_sim_block_start = t_now

            if not success:
                print(f"Solver failed at t={t:.2f}s! Ending simulation early.")
                break
            z_pred_plotting.append(z_pred)
            phi_pred_plotting.append(phi_pred)
            xd_plotting.append(np.copy(xd))
        else:
            # Maintain previous MPC outputs for plotting to keep lengths consistent
            z_pred_plotting.append(z_pred_plotting[-1])
            phi_pred_plotting.append(phi_pred_plotting[-1])
            xd_plotting.append(xd_plotting[-1])

        x_old = np.copy(x)

        # 2. Plant integrates one step forward at 100 Hz
        x = sys_plant.step(x_old, u, t, dt_sim)

        # Neural Network prediction
        x_in = np.concatenate((x_old[3:6], x_old[6:8]))
        with torch.no_grad():
            f_nn_acc = model(torch.tensor(x_in, dtype=torch.float32)).numpy()
        
        # Predicted dynamics model: f_nom(x, u) + F_nn(x, u, theta_hat)
        x_dot_pred = sys_plant.f(x_old) + sys_plant.g_mat(x_old) @ u_old + np.concatenate([np.zeros(3), f_nn_acc, np.zeros(2)])

        # True dynamics derivative (ground truth for monitoring)
        x_dot_true = sys_plant.dynamics(t, x_old, u_old)
        error_acc_true = (x_dot_true - x_dot_pred)[3:6]
        
        # Finite difference acceleration mismatch for adaptation and OCP (observable)
        v_dot_est = (x[3:6] - x_old[3:6]) / dt_sim
        error_acc = v_dot_est - x_dot_pred[3:6]
        
        # Compute predicted full-state derivative v_current = x_dot_pred
        v_current = x_dot_pred

        # Update Rolling Windows
        past_states.append(np.copy(x_old))
        past_f_nom.append(np.copy(x_dot_pred))
        past_v.append(np.copy(v_current))
        past_theta_dot_norm.append(theta_dot_norm_val)

        L = len(past_states)
        S_I = 0.0 # Integral Score (Size)
        norm_theta_dot = 0.0 # Supremum of theta_dot norm over interval

        
        if L > 1:
            x_buffer = np.array(past_states)
            # past_f_nom is the history of f_nom(x, u, theta_hat)
            f_nom_buffer = np.array(past_f_nom) 
            
            x_k = x_buffer[-1] # Current state
            
            # ==========================================================
            # 1. INTEGRAL OCP (Tracking the Size of the Disturbance)
            # ==========================================================
            # 1a. Reverse the historical buffers
            f_nom_rev = np.flip(f_nom_buffer, axis=0)
            x_rev = np.flip(x_buffer, axis=0)
            
            # 1b. Compute the backward integrals using a cumulative sum
            integrals_rev = np.cumsum(f_nom_rev, axis=0) * dt_sim
            
            # 1c. Compute the full state prediction error for all sub-intervals ending at t_k
            prediction_errors = x_k - x_rev - integrals_rev
            
            # 1d. Take the L2 norm and extract the supremum score
            error_norms = np.linalg.norm(prediction_errors, axis=1)
            S_I = np.max(error_norms)
            
            # 1e. Extract supremum of theta_dot norm over the SAME sub-interval
            idx_max = np.argmax(error_norms)
            h_best = idx_max # Length of interval (0-indexed index in rev is length)
            # prediction_errors[idx_max] corresponds to x_rev[idx_max] = x_buffer[L-1-idx_max]
            # The interval is x_buffer[L-1-idx_max] to x_buffer[L-1]
            theta_dot_buffer = np.array(past_theta_dot_norm)
            if h_best > 0:
                norm_theta_dot = np.max(theta_dot_buffer[-h_best:])
            else:
                norm_theta_dot = 0.0
            
            # Update the Integral Quantile
            q_I = ocp_integral.update(S_I)
            dist_bound = np.sqrt(2.0 * ddot_bound * q_I)
        else:
            dist_bound = np.sqrt(2.0 * ddot_bound * ocp_integral.get_quantile())

        # Compute Lipschitz constant L_d for plotting
        L_d, _ = compute_ssml_input_lipschitz(model, x_in)

        # Jacobian J = d f_nn / d theta, shape [OUTPUT_DIM x num_params]
        J = compute_jacobian(model, x_in).detach().numpy()

        # Pass dist_bound directly to the MPC tube generation

        # Track empirical coverage using truth
        total_steps_count += 1
        if np.linalg.norm(error_acc_true) <= dist_bound:
            correct_bounds_count += 1

        # Adaptation updates learns theta online (at 100 Hz)
        theta_dot = gamma_lr * np.dot(J.T, error_acc) - lambd * (
            theta_flat.detach().numpy() - theta_0_flat.numpy()
        )
        theta_flat = theta_flat + torch.tensor(theta_dot, dtype=torch.float32) * dt_sim
        theta_dot_norm_val = np.linalg.norm(theta_dot)
        
        assign_params(model, theta_flat)
        spectral_normalization_clip(model)

        param_change = np.linalg.norm(theta_flat.detach().numpy() - theta_0_flat.numpy())
        theta_history_norm.append(param_change)

        # Append variables for graphing (at 100 Hz)
        t += dt_sim
        # u_old = np.copy(u) # already handled at loop top if needed
        x_plotting.append(np.copy(x))
        t_plotting.append(t)
        drift_scores_I.append(S_I)
        drift_quantiles_I.append(ocp_integral.get_quantile())
        dist_bound_plotting.append(dist_bound)
        tube_plotting.append(sys_controller.Phi)
        theta_plotting.append(theta_flat.detach().numpy().copy())
        residual_plotting.append(np.linalg.norm(error_acc_true))
        Ld_plotting.append(L_d)

        # Collision check
        stop_flag = False
        for obs in obstacles:
            if np.linalg.norm(x[:3] - obs['pos']) < obs['r']:
                print(f"COLLISION at t={t:.2f}s! Penetrated obstacle at {obs['pos']}")
                stop_flag = True
                break
        # Stop when inside the goal region
        goal_dist = np.linalg.norm(x[:3] - x_goal[:3])
        if goal_dist < goal_radius:
            print(f"Goal region reached at t={t:.2f}s! Distance: {goal_dist:.3f}m (radius: {goal_radius}m)")
            stop_flag = True
        if stop_flag:
            break


    # Append terminal target since while loop offsets it by 1
    xd_plotting.append(np.copy(x_goal))

    # Convert lists to arrays for editing plots
    x_history = np.array(x_plotting)
    t_history = np.array(t_plotting)
    xd_history = np.array(xd_plotting)
    z_pred_hist = np.array(z_pred_plotting)
    phi_pred_hist = np.array(phi_pred_plotting)
    dist_bound_history = np.array(dist_bound_plotting)
    drift_scores_I = np.array(drift_scores_I)
    drift_quantiles_I = np.array(drift_quantiles_I)

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
    ax5.plot(t_history, drift_scores_I, 'c-', label='Integral Score $S_{I,k}$', linewidth=2, alpha=0.5)
    ax5.plot(t_history, drift_quantiles_I, 'c--', label='Integral Quantile $q_{I,k}$', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Value')
    ax5.set_title('Neural Acceleration OCP (Integral Score)')
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
    fig_bounds, ax_b1 = plt.subplots(1, 1, figsize=(10, 5))
    ax_b1.plot(t_history, residual_history, 'r-', label=r'Instantaneous Residual $\|\tilde F + \delta\|$', alpha=0.6)
    ax_b1.plot(t_history, dist_bound_history, 'k--', label=r'OCP Disturbance Bound $E_k$', linewidth=2)
    ax_b1.set_xlabel('Time (s)')
    ax_b1.set_ylabel('Dynamics Mismatch (m/s^2)')
    ax_b1.set_title('Unified OCP Bounds and Dynamics Residuals')
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
    ax_top.plot(xd_history[:, 0], xd_history[:, 1], 'k--', alpha=0.5)
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

    # Empirical Coverage Report
    if total_steps_count > 0:
        coverage = correct_bounds_count / total_steps_count
        print(f"\n--- Empirical Coverage Report ---")
        print(f"Disturbance was correctly bounded {correct_bounds_count} out of {total_steps_count} times.")
        print(f"Empirical Coverage: {coverage:.2%} (Target: {1-ocp_integral.alpha:.1%})")
        print(f"----------------------------------\n")

    # --- Animation ---
    print("Generating Animation...")
    fig_anim = plt.figure(figsize=(8, 6))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    
    # Plot target trajectory
    line_ref, = ax_anim.plot(xd_history[:, 0], xd_history[:, 1], xd_history[:, 2], color='orange', linestyle='--')
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
    
    ani = animation.FuncAnimation(fig_anim, update_graph, init_func=init, frames=range(0, len(x_history), 5), interval=50, blit=False)
    ani.save('scenario.gif', writer='pillow', fps=20)
    print("Animation saved to scenario.gif")

    # --- Top-Down 2D Animation ---
    print("Generating top-down animation...")
    fig_td = plt.figure(figsize=(8, 6))
    ax_td = fig_td.add_subplot(111)
    
    # Plot target trajectory
    line_ref_td, = ax_td.plot(xd_history[:, 0], xd_history[:, 1], color='orange', linestyle='--')
    line_true_td, = ax_td.plot([], [], 'b-', label='Quadcopter Path', linewidth=2)
    scatter_true_td = ax_td.scatter([], [], color='blue', s=50)
    
    line_pred_td, = ax_td.plot([], [], 'g--', label='MPC Prediction', linewidth=1)
    scatter_pred_td = ax_td.scatter([], [], color='cyan', alpha=0.3, label='Tube Envelope')
    
    ax_td.scatter(x_goal[0], x_goal[1], color='gold', marker='*', s=100)
    
    def init_td():
        line_true_td.set_data([], [])
        scatter_true_td.set_offsets(np.empty((0, 2)))
        line_pred_td.set_data([], [])
        scatter_pred_td.set_offsets(np.empty((0, 2)))
        return scatter_true_td, line_true_td, line_ref_td, line_pred_td, scatter_pred_td
        
    def update_graph_td(k):
        scatter_true_td.set_offsets(np.column_stack((x_history[k:k+1, 0], x_history[k:k+1, 1])))
        line_true_td.set_data(x_history[:k+1, 0], x_history[:k+1, 1])
        
        line_ref_td.set_data(xd_history[:k+1, 0], xd_history[:k+1, 1])
        
        # Update predicting horizon
        line_pred_td.set_data(z_pred_hist[k, :, 0], z_pred_hist[k, :, 1])
        
        scatter_pred_td.set_offsets(np.column_stack((z_pred_hist[k, :, 0], z_pred_hist[k, :, 1])))
        # scale scatter size dynamically from tube bound phi
        scatter_pred_td.set_sizes((phi_pred_hist[k] * 100) ** 2)

        ax_td.set_title(rf'DTMPC Top-Down Flight | Time: {t_history[k]:.1f}s | $\Phi$: {tube_history[k]:.2f}m')
        return scatter_true_td, line_true_td, line_ref_td, line_pred_td, scatter_pred_td

    # Draw spherical obstacles (as 2D circles)
    for obs in obstacles:
        circ = plt.Circle((obs['pos'][0], obs['pos'][1]), obs['r'], color='red', alpha=1.0)
        ax_td.add_patch(circ)
        
    x_lo, x_hi = np.min(x_history[:,0])-0.5, np.max(x_history[:,0])+0.5
    y_lo, y_hi = -2.5, 2.5
    ax_td.set_xlim([x_lo, x_hi])
    ax_td.set_ylim([y_lo, y_hi])
    ax_td.set_aspect('equal')
    ax_td.set_xlabel('X')
    ax_td.set_ylabel('Y')
    ax_td.legend()
    
    ani_td = animation.FuncAnimation(fig_td, update_graph_td, init_func=init_td, frames=range(0, len(x_history), 5), interval=50, blit=False)
    ani_td.save('scenario_topdown.gif', writer='pillow', fps=20)
    print("Top-down animation saved to scenario_topdown.gif")

if __name__ == '__main__':
    main()
