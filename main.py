import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
plt.switch_backend('Agg')

# Architecture Modules
from plant import Plant
from sensor import Sensor
from observer import Observer
from adaptation import Adaptation
from ocp import DriftScoreOCP, InnovationScoreOCP
from controller import SafetyCriticalController, RobustTubeMPC
from high_level_objective import HighLevelObjective
from design_gains import solve_lmi_gains, get_stable_adaptation_gain

def main():
    np.random.seed(42)

    # Simulation Variables
    t = 0.0
    dt = 0.1

    # Initialize Architecture Components
    sys_plant = Plant()
    
    # Observation matrix: Only position is measured
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    sys_sensor = Sensor(C=C, noise_std=0.1)
    
    # Observer gain L must be 4x2
    # Designing L via LMI for a desired decay rate alpha
    A_lin = np.array([
        [0, 0, 1,    0],
        [0, 0, 0,    1],
        [0, 0, -0.5, 0],
        [0, 0, 0,    -0.5]
    ])
    L, _ = solve_lmi_gains(A_lin, C, alpha=1.5)
    sys_observer = Observer(plant=sys_plant, C=C, L=L)
    
    sys_adaptation = Adaptation(num_states=4, num_features=2, max_capacity=10, tolerance=1e-3)
    
    ocp_drift = DriftScoreOCP(alpha=0.1, eta_const=0.1, q_init=0.1)
    ocp_inn = InnovationScoreOCP(alpha=0.1, eta_const=0.1, q_init=0.1)
    
    # Toggle which controller to use:
    sys_controller = SafetyCriticalController(plant=sys_plant, x_obs=np.array([1.0, 1.1]), d_safe=0.5)
    # sys_controller = RobustTubeMPC(plant=sys_plant, x_obs=np.array([1.0, 1.1]), d_safe=0.5, H=8)
    
    sys_objective = HighLevelObjective()

    # State variables (4 elements)
    x = np.array([0.0, 0.0, 0.0, 0.0])
    xhat = np.array([0.0, 0.0, 0.0, 0.0])
    theta_hat = np.array([0.0, 0.0])
    u = np.array([0.0, 0.0])

    # Tracking arrays
    x_plotting = [np.copy(x)]
    xhat_plotting = [np.copy(xhat)]
    t_plotting = [t]
    stack_eigenvalues = [sys_adaptation.check_pe()]
    
    # OCP Tracking
    drift_scores = [0.0]
    drift_quantiles = [ocp_drift.get_quantile()]
    inn_scores = [0.0]
    inn_quantiles = [ocp_inn.get_quantile()]

    # Parameter and Error Tracking
    Gamma = 5.0 * np.eye(2) # Adaptation learning rate
    theta_plotting = [np.copy(theta_hat)]
    theta_err_plotting = [np.copy(theta_hat - sys_plant.theta_true)]
    residual_err_plotting = [np.linalg.norm(sys_plant.F(x) @ (theta_hat - sys_plant.theta_true) - sys_plant.Delta(x, t))]
    state_err_plotting = [np.linalg.norm(xhat - x)]
    
    x_bar_m_init = (ocp_inn.get_quantile() + sys_controller.epsilon_bar) / sys_controller.C_m_min_sv
    x_bar_u_init = ocp_drift.get_quantile()
    state_err_bound_plotting = [x_bar_m_init + x_bar_u_init]
    dist_bound_plotting = [ocp_drift.get_quantile() / sys_controller.T_horizon]

    # Simulation Loop
    while t <= 15.0:
        # High Level Objective Output
        xd = sys_objective.get_reference(t)

        # Sensor Output
        y = sys_sensor.measure(x)

        # State Observer updates
        # To avoid lag, Observer runs its derivative
        # The true controller would use xhat, OCP scores, and xd. For now, u=0.
        u_old = np.copy(u)
        # u = sys_controller.compute_u(xhat, theta_hat, xd, drift_quantiles[-1], inn_quantiles[-1])
        u = sys_controller.nominal_controller(xhat, xd)

        xhat_dot = sys_observer.compute_xhat_dot(xhat, u, theta_hat, y)
        xhat_old = np.copy(xhat)
        xhat += xhat_dot * dt

        # Update CL Stack and Gamma via LMI Stability Bound
        X_mat = sys_observer.compute_X(xhat, xhat_old, u, u_old, dt)
        Z_mat = sys_observer.compute_Z(xhat, xhat_old, dt)
        sys_adaptation.update_stack(X_mat, Z_mat)
        
        # Dynamic Gamma synthesis
        Omega = sys_adaptation.get_information_matrix()
        Gamma = get_stable_adaptation_gain(Omega, dt=dt)
        
        # Adaptation updates learns theta
        cl_grad = sys_adaptation.get_cl_grad(theta_hat)
        theta_hat_dot = Gamma @ cl_grad.flatten()
        theta_hat += theta_hat_dot * dt
        
        # Plant integrates one step forward
        x = sys_plant.step(x, u, t, dt)
        
        # OCP Updates
        y_hat = sys_observer.get_y_hat(xhat)
        
        S_inn = ocp_inn.compute_score(y, y_hat)
        q_inn = ocp_inn.update(S_inn)
        
        S_drift = ocp_drift.compute_score(X_mat, Z_mat, theta_hat)
        q_drift = ocp_drift.update(S_drift)

        # Append variables for graphing
        t += dt
        x_plotting.append(np.copy(x))
        xhat_plotting.append(np.copy(xhat))
        t_plotting.append(t)
        stack_eigenvalues.append(sys_adaptation.check_pe())
        drift_scores.append(S_drift)
        drift_quantiles.append(q_drift)
        inn_scores.append(S_inn)
        inn_quantiles.append(q_inn)
        
        theta_plotting.append(np.copy(theta_hat))
        theta_err = theta_hat - sys_plant.theta_true
        theta_err_plotting.append(np.copy(theta_err))
        res_err = np.linalg.norm(sys_plant.F(x) @ theta_err - sys_plant.Delta(x, t))
        residual_err_plotting.append(res_err)
        state_err_plotting.append(np.linalg.norm(xhat - x))
        
        x_bar_m = (q_inn + sys_controller.epsilon_bar) / sys_controller.C_m_min_sv
        x_bar_u = q_drift
        state_err_bound_plotting.append(x_bar_m + x_bar_u)
        dist_bound_plotting.append(q_drift / sys_controller.T_horizon)

    print("Final history stack valid X_hist:\\n", sys_adaptation.history_stack.X_hist[:sys_adaptation.history_stack.current_size])

    # Convert lists to arrays for editing plots
    x_history = np.array(x_plotting)
    xhat_history = np.array(xhat_plotting)
    t_history = np.array(t_plotting)
    eigenvalue_history = np.array(stack_eigenvalues)
    theta_history = np.array(theta_plotting)
    theta_err_history = np.array(theta_err_plotting)
    res_err_history = np.array(residual_err_plotting)
    state_err_history = np.array(state_err_plotting)
    state_err_bound_history = np.array(state_err_bound_plotting)
    dist_bound_history = np.array(dist_bound_plotting)

    # Subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 20))

    # Plot x vs time
    ax1.plot(t_history, x_history[:, 0], 'b-', label='x[0]', linewidth=2)
    ax1.plot(t_history, x_history[:, 1], 'r-', label='x[1]', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('x')
    ax1.set_title('Plant True State x vs Time')
    ax1.legend()
    ax1.grid(True)

    # Plot xhat vs time
    ax2.plot(t_history, xhat_history[:, 0], 'b--', label='xhat[0]', linewidth=2)
    ax2.plot(t_history, xhat_history[:, 1], 'r--', label='xhat[1]', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('xhat')
    ax2.set_title('Observer Estimated State xhat vs Time')
    ax2.legend()
    ax2.grid(True)

    # Plot minimum eigenvalue vs time
    ax3.plot(t_history, eigenvalue_history, 'g-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Min Eigenvalue')
    ax3.set_title('Adaptation: Minimum Eigenvalue vs Time')
    ax3.grid(True)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot Innovation Score vs Quantile
    ax4.plot(t_history, inn_scores, 'm-', label='Innovation Score $S_{inn,k}$', linewidth=2)
    ax4.plot(t_history, inn_quantiles, 'k--', label='Innovation Quantile $q_{inn,k}$', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Value')
    ax4.set_title('Sensor Innovation Score vs Quantile Threshold')
    ax4.legend()
    ax4.grid(True)
    
    # Plot Drift Score vs Quantile
    ax5.plot(t_history, drift_scores, 'c-', label='Drift Score $S_{drift,k}$', linewidth=2)
    ax5.plot(t_history, drift_quantiles, 'k--', label='Drift Quantile $q_{drift,k}$', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Value')
    ax5.set_title('Observer Drift Score vs Quantile Threshold')
    ax5.legend()
    ax5.grid(True)

    plt.tight_layout()
    plt.savefig('architecture_state_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to architecture_state_comparison.png")

    # --- Validation Checks ---
    distances_to_obs = np.sqrt((x_history[:,0] - sys_controller.x_obs[0])**2 + (x_history[:,1] - sys_controller.x_obs[1])**2)
    min_dist = np.min(distances_to_obs)
    print(f"Minimum distance to obstacle center: {min_dist:.3f} (Constraint: {sys_controller.d_safe:.3f})")
    print(f"Final position: [{x_history[-1,0]:.3f}, {x_history[-1,1]:.3f}] (Target: {xd[0]:.3f}, {xd[1]:.3f})")

    # --- 2D Position vs Time Plot ---
    fig_pos, ax_pos = plt.subplots(figsize=(8, 4))
    ax_pos.plot(t_history, x_history[:, 0], label='px(t)', color='blue')
    ax_pos.plot(t_history, x_history[:, 1], label='py(t)', color='orange')
    ax_pos.axhline(y=xd[0], color='blue', linestyle='--', alpha=0.5, label='Target px')
    ax_pos.axhline(y=xd[1], color='orange', linestyle='--', alpha=0.5, label='Target py')
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position')
    ax_pos.set_title('2D Position vs Time (Target Tracking)')
    ax_pos.legend()
    ax_pos.grid(True)
    fig_pos.tight_layout()
    fig_pos.savefig('pos_vs_time.png', dpi=150, bbox_inches='tight')
    print("Plot saved to pos_vs_time.png")

    # --- Parameter Estimation Plot ---
    fig_param, (ax_p1, ax_p2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax_p1.plot(t_history, theta_history[:, 0], label=r'$\hat{\theta}_1$')
    ax_p1.plot(t_history, theta_history[:, 1], label=r'$\hat{\theta}_2$')
    ax_p1.axhline(sys_plant.theta_true[0], color='r', linestyle='--', label=r'$\theta_{1,true}$')
    ax_p1.axhline(sys_plant.theta_true[1], color='g', linestyle='--', label=r'$\theta_{2,true}$')
    ax_p1.set_xlabel('Time (s)')
    ax_p1.set_title(r'Parameter Estimates $\hat{\theta}$ vs Time')
    ax_p1.legend()
    ax_p1.grid(True)
    
    ax_p2.plot(t_history, theta_err_history[:, 0], label=r'$\tilde{\theta}_1$')
    ax_p2.plot(t_history, theta_err_history[:, 1], label=r'$\tilde{\theta}_2$')
    ax_p2.set_xlabel('Time (s)')
    ax_p2.set_title(r'Parameter Estimation Error $\tilde{\theta}$ vs Time')
    ax_p2.legend()
    ax_p2.grid(True)
    
    fig_param.tight_layout()
    fig_param.savefig('parameter_estimation.png', dpi=150, bbox_inches='tight')
    print("Plot saved to parameter_estimation.png")

    # --- Residual and State Error Plot ---
    fig_err, (ax_e1, ax_e2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax_e1.plot(t_history, res_err_history, 'g-', linewidth=2)
    ax_e1.plot(t_history, res_err_history, 'r-')
    ax_e1.set_xlabel('Time (s)')
    ax_e1.set_ylabel(r'|| F(x)$\tilde{\theta}$ - $\Delta(x,t)$ ||')
    ax_e1.set_title('True Residual / Approximation Error over Time')
    ax_e1.grid(True)
    
    ax_e2.plot(t_history, state_err_history, 'b-')
    ax_e2.set_xlabel('Time (s)')
    ax_e2.set_ylabel(r'|| $\hat{x}$ - x ||')
    ax_e2.set_title('State Estimation Error Norm vs Time')
    ax_e2.grid(True)
    
    fig_err.tight_layout()
    fig_err.savefig('residuals_and_state_error.png', dpi=150, bbox_inches='tight')
    print("Plot saved to residuals_and_state_error.png")
    
    # --- OCP Bounds vs True Error Plot ---
    fig_bounds, (ax_b1, ax_b2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax_b1.plot(t_history, res_err_history, 'g-', label=r'True Disturbance ||F(x)$\tilde{\theta}$ - $\Delta(x,t)$||', linewidth=2)
    ax_b1.plot(t_history, dist_bound_history, 'k--', label=r'OCP Disturbance Bound $E_k$', linewidth=2)
    ax_b1.set_xlabel('Time (s)')
    ax_b1.set_ylabel('Disturbance Norm')
    ax_b1.set_title('OCP-Derived Estimated Disturbance Bound vs True Disturbance')
    ax_b1.legend()
    ax_b1.grid(True)
    
    ax_b2.plot(t_history, state_err_history, 'm-', label=r'True State Est Error ||$\hat{x}$ - x||', linewidth=2)
    ax_b2.plot(t_history, state_err_bound_history, 'k--', label=r'OCP State Est Bound $\bar{x}_m + \bar{x}_u$', linewidth=2)
    ax_b2.set_xlabel('Time (s)')
    ax_b2.set_ylabel('State Error Norm')
    ax_b2.set_title('OCP-Derived State Estimation Bound vs True State Estimation Error')
    ax_b2.legend()
    ax_b2.grid(True)
    
    fig_bounds.tight_layout()
    fig_bounds.savefig('ocp_bounds_vs_true.png', dpi=150, bbox_inches='tight')
    print("Plot saved to ocp_bounds_vs_true.png")

    # --- 2D Animation Generation ---
    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
    ax_anim.set_xlim(-0.5, 3.0)
    ax_anim.set_ylim(-0.5, 3.0)
    ax_anim.set_aspect('equal')
    ax_anim.set_title("2D Safe Navigation using CBF and OCP")
    ax_anim.set_xlabel("X Position")
    ax_anim.set_ylabel("Y Position")
    ax_anim.grid(True)

    # Draw Obstacle
    obs_circle = plt.Circle((sys_controller.x_obs[0], sys_controller.x_obs[1]), sys_controller.d_safe, 
                            color='red', alpha=0.3, label='Obstacle + Margin')
    ax_anim.add_patch(obs_circle)
    ax_anim.plot(sys_controller.x_obs[0], sys_controller.x_obs[1], 'rx', markersize=10, label='Obstacle Center')

    # Draw Goal
    ax_anim.plot(xd[0], xd[1], 'g*', markersize=15, label='Target Objective')

    # Moving Elements
    robot_line, = ax_anim.plot([], [], 'b-', alpha=0.6, label='Robot Trajectory')
    robot_dot, = ax_anim.plot([], [], 'bo', markersize=8)
    
    ax_anim.legend(loc='upper left')

    def init_anim():
        robot_line.set_data([], [])
        robot_dot.set_data([], [])
        return robot_line, robot_dot

    def update_anim(frame):
        # frame is the index in the history array
        xs = x_history[:frame, 0]
        ys = x_history[:frame, 1]
        
        robot_line.set_data(xs, ys)
        
        if frame > 0:
            robot_dot.set_data([xs[-1]], [ys[-1]])
            
        return robot_line, robot_dot

    # Set up animation
    # Only animate a subset of frames if the simulation is very long, but 5.0s @ dt=0.1 is only 50 frames.
    ani = animation.FuncAnimation(fig_anim, update_anim, frames=len(t_history),
                                  init_func=init_anim, blit=True, interval=100) # 100ms per frame
                                  
    writer = animation.PillowWriter(fps=10, metadata=dict(artist='Antigravity'), bitrate=1800)
    ani.save('scenario.gif', writer=writer)
    print("Animation saved to scenario.gif")

if __name__ == '__main__':
    main()
