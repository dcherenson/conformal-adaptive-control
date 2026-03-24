import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

WEIGHTS_PATH = "ssml_weights.pt"

# Define the SSML-AC Network Architecture
# Use observer state and control as input: x_in = [p_hat(3), v_hat(3), u(3)]
INPUT_DIM = 9
HIDDEN_DIM = 50
# Output estimates additive dynamics mismatch for full observer dynamics:
# [delta_p_dot(3), delta_v_dot(3)]
OUTPUT_DIM = 6


class SSMLNet(nn.Module):
    def __init__(self):
        super(SSMLNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc4 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

    def get_jacobian(self, x):
        # Computes Jacobian of output w.r.t network parameters
        # For simplicity, we can use torch.autograd.functional.jacobian or flatten params
        # Actually a faster way for online adaptation is to use autograd over flattened params.
        pass


def spectral_normalization_clip(model, v_max=10.0):
    for param in model.parameters():
        norm = torch.linalg.norm(param)
        if norm > v_max:
            param.data = param.data * (v_max / norm)


# Quadcopter Dynamics (Simplified to Point-Mass for Position Tracking)
class QuadcopterSim:
    def __init__(self):
        self.m = 1.0  # kg
        self.g = np.array([0, 0, -9.81])

    def wind_disturbance(self, t, p):
        # Average intensity 1.0 m/s^2, highly dynamic
        # Based on fans: varying sine waves.
        d_x = 1.0 * np.sin(0.5 * t) + 0.5 * np.sin(2.0 * t) + 0.2 * np.random.randn()
        d_y = 1.2 * np.cos(0.4 * t) + 0.6 * np.cos(1.8 * t) + 0.2 * np.random.randn()
        d_z = 0.5 * np.sin(0.3 * t) + 0.1 * np.random.randn()
        return np.array([d_x, d_y, d_z]) * self.m  # Force disturbance

    def dynamics(self, t, state, u):
        # state = [p_x, p_y, p_z, v_x, v_y, v_z]
        p = state[0:3]
        v = state[3:6]
        d = self.wind_disturbance(t, p)

        dp = v
        # m a = m g + u + d
        dv = self.g + (u + d) / self.m

        return np.concatenate((dp, dv))


# Generate Figure-8 Reference Trajectory
def get_reference(t):
    # 1.2m long, 1.0m wide figure-8
    # Period: T_period = 20s (60s total = 3 loops)
    w = 2 * np.pi / 20.0

    p_d = np.array([1.2 / 2 * np.sin(w * t), 1.0 / 2 * np.sin(2 * w * t), 1.0])
    v_d = np.array([1.2 / 2 * w * np.cos(w * t), 1.0 / 2 * 2 * w * np.cos(2 * w * t), 0.0])
    a_d = np.array([-1.2 / 2 * w**2 * np.sin(w * t), -1.0 / 2 * (2 * w) ** 2 * np.sin(2 * w * t), 0.0])

    return p_d, v_d, a_d


def nominal_pd_control(t, state, m, Kp, Kd):
    p = state[0:3]
    v = state[3:6]
    p_d, v_d, a_d = get_reference(t)

    e = p - p_d
    edot = v - v_d

    # u = m * a_d - m*g - Kp e - Kd edot
    u = m * a_d - m * np.array([0, 0, -9.81]) - np.dot(Kp, e) - np.dot(Kd, edot)
    return u


def collect_offline_data():
    print("Collecting Offline Data...")
    sim = QuadcopterSim()
    dt = 0.02  # 50 Hz
    t_end = 60.0  # 3 loops

    Kp = np.diag([5.0, 5.0, 5.0])
    Kd = np.diag([2.0, 2.0, 2.0])

    times = np.arange(0, t_end, dt)
    state = np.zeros(6)

    data_x = []
    data_y = []

    for t in times:
        u = nominal_pd_control(t, state, sim.m, Kp, Kd)
        d_true = sim.wind_disturbance(t, state[0:3])

        # input x = [p, v, u]
        x_in = np.concatenate((state[0:3], state[3:6], u))

        # Target is additive mismatch on full state derivative:
        # x_dot_true - x_dot_nominal = [0, d/m]
        y_target = np.concatenate((np.zeros(3), d_true / sim.m))

        data_x.append(x_in)
        data_y.append(y_target)

        # Step simulation
        sol = solve_ivp(sim.dynamics, [t, t + dt], state, args=(u,), method="RK45")
        state = sol.y[:, -1]

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


def train_ssml(data_x, data_y):
    print("Training SSML...")
    model = SSMLNet()

    Ha = 25  # Adaptation horizon
    Ht = 25  # Training horizon

    alpha = 0.002
    beta = 0.001
    lambda_dir = 0.5
    lambda_norm = 0.05

    epochs = 500
    N = len(data_x)

    optimizer = optim.SGD(model.parameters(), lr=beta)

    # Convert to tensors
    X_tensor = torch.tensor(data_x, dtype=torch.float32)
    Y_tensor = torch.tensor(data_y, dtype=torch.float32)

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        # Sample random tasks (trajectory clips)
        num_tasks = 10
        total_meta_loss = 0

        for _ in range(num_tasks):
            # Pick random start for clip
            start_idx = np.random.randint(0, N - Ha - Ht - 1)

            # Adaptation set
            X_a = X_tensor[start_idx : start_idx + Ha]
            Y_a = Y_tensor[start_idx : start_idx + Ha]

            # Training set
            X_t = X_tensor[start_idx + Ha : start_idx + Ha + Ht]
            Y_t = Y_tensor[start_idx + Ha : start_idx + Ha + Ht]

            # 1. Inner Loop: Gradient descent on Ba
            # Clone model to simulate adaptation
            fast_weights = [p.clone() for p in model.parameters()]

            pred_a = model.relu(torch.nn.functional.linear(X_a, fast_weights[0], fast_weights[1]))
            pred_a = model.relu(torch.nn.functional.linear(pred_a, fast_weights[2], fast_weights[3]))
            pred_a = model.relu(torch.nn.functional.linear(pred_a, fast_weights[4], fast_weights[5]))
            pred_a = torch.nn.functional.linear(pred_a, fast_weights[6], fast_weights[7])

            loss_a = torch.sum((pred_a - Y_a) ** 2)
            grads = torch.autograd.grad(loss_a, fast_weights, create_graph=True)
            fast_weights = [fw - alpha * g for fw, g in zip(fast_weights, grads)]

            # 2. Outer Loop: Prediction on Bt
            pred_t = model.relu(torch.nn.functional.linear(X_t, fast_weights[0], fast_weights[1]))
            pred_t = model.relu(torch.nn.functional.linear(pred_t, fast_weights[2], fast_weights[3]))
            pred_t = model.relu(torch.nn.functional.linear(pred_t, fast_weights[4], fast_weights[5]))
            pred_t = torch.nn.functional.linear(pred_t, fast_weights[6], fast_weights[7])
            loss_t_adapt = torch.sum((pred_t - Y_t) ** 2)

            # Direct prediction loss
            pred_t_dir = model(X_t)
            loss_t_dir = torch.sum((pred_t_dir - Y_t) ** 2)

            meta_loss = loss_t_adapt + lambda_dir * loss_t_dir
            total_meta_loss += meta_loss

        norm_penalty = sum([torch.norm(p) ** 2 for p in model.parameters()])
        total_meta_loss = total_meta_loss / num_tasks + lambda_norm * norm_penalty

        total_meta_loss.backward()
        optimizer.step()

        # Apply layer spectral normalization (approximated by weight clipping)
        spectral_normalization_clip(model)

        epoch_loss += total_meta_loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss}")

    print("Pre-training Complete.")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")
    return model


def get_or_train_model():
    """Load pre-trained weights if they exist, otherwise collect data and train."""
    model = SSMLNet()
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
        print(f"Loaded pre-trained weights from {WEIGHTS_PATH}")
    else:
        data_x, data_y = collect_offline_data()
        model = train_ssml(data_x, data_y)
    return model


def flatten_params(model):
    return torch.cat([p.flatten() for p in model.parameters()])


def assign_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx : idx + numel].view_as(p))
        idx += numel


def compute_jacobian(model, x_in):
    # Compute Jacobian J = df/dtheta
    model.eval()
    x_tensor = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0)

    # Jacobian of network output w.r.t flattened parameters
    J_list = []
    for i in range(OUTPUT_DIM):
        model.zero_grad()
        out = model(x_tensor)[0, i]
        out.backward(retain_graph=True)
        grad_flat = torch.cat([p.grad.flatten() for p in model.parameters()])
        J_list.append(grad_flat)

    return torch.stack(J_list)


def run_online_ssml_ac(pretrained_model, baseline=False):
    print(f"Running {'Baseline (PID)' if baseline else 'SSML-AC'} Flight...")
    sim = QuadcopterSim()
    dt = 0.02
    t_end = 60.0

    Kp = np.diag([10.0, 10.0, 10.0])
    Kd = np.diag([4.0, 4.0, 4.0])

    # Adaptation gains from DNN observer section in paper:
    # theta_dot = Gamma * J^T * C^T (y - C x_hat) - lambda * (theta - theta_0)
    gamma = 0.01
    lambd = 0.1

    # Position-only measurement matrix y = C x + noise, x=[p; v]
    C = np.hstack((np.eye(3), np.zeros((3, 3))))

    # Observer gain L = [L_p; L_v]
    Lp = 8.0 * np.eye(3)
    Lv = 20.0 * np.eye(3)
    L = np.vstack((Lp, Lv))

    times = np.arange(0, t_end, dt)
    state = np.zeros(6)
    x_hat = np.zeros(6)

    model = SSMLNet()
    if not baseline:
        model.load_state_dict(pretrained_model.state_dict())

    theta_0_flat = flatten_params(model).clone().detach()
    theta_flat = flatten_params(model).clone().detach()

    trajectory_pos = []
    trajectory_ref = []
    trajectory_xhat = []   # observer position estimate
    trajectory_xtrue = []  # true full state (for vel estimation error too)

    for t in times:
        p = state[0:3]
        p_hat = x_hat[0:3]
        v_hat = x_hat[3:6]

        # Position-only noisy measurement
        y = p + np.random.randn(3) * 0.05

        p_d, v_d, a_d = get_reference(t)
        e_hat = p_hat - p_d
        edot_hat = v_hat - v_d

        # Nominal PD control based on observer state estimate only
        u_base = sim.m * a_d - sim.m * sim.g - np.dot(Kp, e_hat) - np.dot(Kd, edot_hat)

        if baseline:
            u = u_base
            f_nn_full = np.zeros(OUTPUT_DIM)
        else:
            x_in = np.concatenate((x_hat, u_base))
            with torch.no_grad():
                f_nn_full = model(torch.tensor(x_in, dtype=torch.float32)).numpy()

            # Control compensation uses estimated acceleration mismatch only
            f_nn_acc = f_nn_full[3:6]
            u = u_base - sim.m * f_nn_acc

            # Jacobian J = d f_nn / d theta, shape [OUTPUT_DIM x num_params]
            J = compute_jacobian(model, x_in).detach().numpy()

            # Adaptation law from paper (DNN section):
            # theta_dot = Gamma * J^T * C^T (y - C x_hat) - lambda * (theta - theta0)
            innovation = y - C @ x_hat
            output_error_lifted = C.T @ innovation
            theta_dot = gamma * np.dot(J.T, output_error_lifted) - lambd * (
                theta_flat.detach().numpy() - theta_0_flat.numpy()
            )
            theta_flat = theta_flat + torch.tensor(theta_dot, dtype=torch.float32) * dt

            assign_params(model, theta_flat)
            spectral_normalization_clip(model)

        # Observer propagation uses only output innovation and model mismatch estimate
        innovation = y - C @ x_hat
        x_hat_dot_nom = np.concatenate((x_hat[3:6], sim.g + u / sim.m))
        x_hat_dot = x_hat_dot_nom + f_nn_full + L @ innovation
        x_hat = x_hat + x_hat_dot * dt

        trajectory_pos.append(p)
        trajectory_ref.append(p_d)
        trajectory_xhat.append(x_hat.copy())
        trajectory_xtrue.append(state.copy())

        sol = solve_ivp(sim.dynamics, [t, t + dt], state, args=(u,), method="RK45")
        state = sol.y[:, -1]

    trajectory_pos = np.array(trajectory_pos)
    trajectory_ref = np.array(trajectory_ref)
    trajectory_xhat = np.array(trajectory_xhat)
    trajectory_xtrue = np.array(trajectory_xtrue)

    rmse = np.sqrt(np.mean((trajectory_pos - trajectory_ref) ** 2))
    print(f"RMSE: {rmse*100:.2f} cm")
    return trajectory_pos, trajectory_ref, trajectory_xhat, trajectory_xtrue, rmse


def plot_results(traj_pid, traj_ssml, traj_ref,
                 xhat_pid, xtrue_pid, xhat_ssml, xtrue_ssml):
    times = np.arange(0, 60.0, 0.02)
    labels = ["X", "Y", "Z"]

    # ── Figure 1: 3-D trajectories + tracking error ──────────────────────────
    fig1 = plt.figure(figsize=(13, 5))

    ax1 = fig1.add_subplot(121, projection="3d")
    ax1.plot(traj_ref[:, 0], traj_ref[:, 1], traj_ref[:, 2], "k--", label="Reference")
    ax1.plot(traj_pid[:, 0], traj_pid[:, 1], traj_pid[:, 2], "r-", label="PID", alpha=0.7)
    ax1.plot(traj_ssml[:, 0], traj_ssml[:, 1], traj_ssml[:, 2], "b-", label="SSML-AC", alpha=0.7)
    ax1.set_title("Figure-8 Tracking under Wind Disturbance")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.legend()

    ax2 = fig1.add_subplot(122)
    err_pid = np.linalg.norm(traj_pid - traj_ref, axis=1)
    err_ssml = np.linalg.norm(traj_ssml - traj_ref, axis=1)
    ax2.plot(times, err_pid * 100, "r-", label="PID", alpha=0.8)
    ax2.plot(times, err_ssml * 100, "b-", label="SSML-AC", alpha=0.8)
    ax2.set_title("Tracking Error")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position Error [cm]")
    ax2.legend()
    fig1.tight_layout()
    fig1.savefig("ssml_tracking.png", dpi=150)
    print("Saved: ssml_tracking.png")

    # ── Figure 2: State estimation error (position) ───────────────────────────
    fig2, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    fig2.suptitle("State Estimation Error: x̂ vs x_true", fontsize=13)

    for i, lbl in enumerate(labels):
        # PID: observer vs true position
        ax = axes[i, 0]
        ax.plot(times, xtrue_pid[:, i],    "k-",  lw=1,   label="True")
        ax.plot(times, xhat_pid[:, i],     "r--", lw=1,   label="Observer")
        ax.fill_between(times,
                         xtrue_pid[:, i] - (xhat_pid[:, i] - xtrue_pid[:, i]),
                         xtrue_pid[:, i],
                         alpha=0.15, color="red")
        ax.set_ylabel(f"Pos {lbl} [m]")
        if i == 0:
            ax.set_title("PID Baseline")
            ax.legend(fontsize=8)

        # SSML-AC: observer vs true position
        ax = axes[i, 1]
        ax.plot(times, xtrue_ssml[:, i],   "k-",  lw=1,   label="True")
        ax.plot(times, xhat_ssml[:, i],    "b--", lw=1,   label="Observer")
        ax.fill_between(times,
                         xtrue_ssml[:, i] - (xhat_ssml[:, i] - xtrue_ssml[:, i]),
                         xtrue_ssml[:, i],
                         alpha=0.15, color="blue")
        if i == 0:
            ax.set_title("SSML-AC")
            ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    fig2.tight_layout()
    fig2.savefig("ssml_estimation.png", dpi=150)
    print("Saved: ssml_estimation.png")

    # ── Figure 3: Estimation error norm ───────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    fig3.suptitle("State Estimation Error Norm (‖x̂ − x_true‖)", fontsize=13)

    for (ax, xhat, xtrue, color, title) in [
        (axes3[0], xhat_pid,  xtrue_pid,  "r", "PID Baseline"),
        (axes3[1], xhat_ssml, xtrue_ssml, "b", "SSML-AC"),
    ]:
        pos_err = np.linalg.norm(xhat[:, :3] - xtrue[:, :3], axis=1) * 100   # cm
        vel_err = np.linalg.norm(xhat[:, 3:] - xtrue[:, 3:], axis=1)          # m/s
        ax.plot(times, pos_err, color=color,    lw=1,  label="Position est. error [cm]")
        ax.plot(times, vel_err, color=color, ls="--", lw=1, alpha=0.7, label="Velocity est. error [m/s]")
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)

    fig3.tight_layout()
    fig3.savefig("ssml_estimation_norm.png", dpi=150)
    print("Saved: ssml_estimation_norm.png")


if __name__ == "__main__":
    pretrained_model = get_or_train_model()

    traj_pid,  traj_ref, xhat_pid,  xtrue_pid,  rmse_pid  = run_online_ssml_ac(pretrained_model, baseline=True)
    traj_ssml, _,        xhat_ssml, xtrue_ssml, rmse_ssml = run_online_ssml_ac(pretrained_model, baseline=False)

    print(f"\nPID  RMSE: {rmse_pid*100:.2f} cm")
    print(f"SSML RMSE: {rmse_ssml*100:.2f} cm")

    plot_results(traj_pid, traj_ssml, traj_ref,
                 xhat_pid, xtrue_pid, xhat_ssml, xtrue_ssml)
