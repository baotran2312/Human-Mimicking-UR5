# Manuscript Draft: Problem Formulation & Proposed Scheme (Revised)

This document contains the revised draft of the **Problem Formulation** and **Proposed Scheme** sections, addressing all theoretical and practical comments regarding singularities, two-sided delay bounds, LKF mathematical compatibility, task-space representation (quaternions), and practical implementation.

---

# III. Problem Formulation

Consider an integrated robotic system consisting of a 6-DoF manipulator (UR5) and a 19-DoF custom dexterous hand, constituting a highly redundant kinematic chain with a joint configuration vector $q(t) \in \mathbb{R}^n$ ($n = 25$), defined as:
$$q(t) = \begin{bmatrix} q_a(t) \\ q_h(t) \end{bmatrix}$$
where $q_a(t) \in \mathbb{R}^6$ represents the joint angles of the robotic arm, and $q_h(t) \in \mathbb{R}^{19}$ denotes the active joint angles of the multi-fingered hand. 

The forward kinematics mapping the joint space configuration to the task space pose $x(t) \in \mathcal{X}$ is given by:
$$x(t) = f(q(t))$$
where the task space $\mathcal{X} = \mathbb{R}^3 \times \mathcal{S}^3 \times \mathbb{R}^{n_f}$ is designed to avoid orientation representation singularities by utilizing unit quaternions for the wrist orientation, and relative Euclidean finger opposition distances for the hand. Specifically, $x(t) = [p_w^T(t), Q_w^T(t), d_f^T(t)]^T$, where $p_w(t) \in \mathbb{R}^3$ is the wrist position, $Q_w(t) = \{\eta, \epsilon\} \in \mathcal{S}^3$ is the unit quaternion of the wrist orientation, and $d_f(t) \in \mathbb{R}^{n_f}$ denotes the relative distances between designated fingertip pairs. 

Differentiating the forward kinematics yields the differential kinematics:
$$v(t) = J(q(t))\dot{q}(t)$$
where $v(t) = [\dot{p}_w^T(t), \omega_w^T(t), \dot{d}_f^T(t)]^T \in \mathbb{R}^m$ is the spatial task-space velocity vector ($\omega_w$ being the physical angular velocity of the wrist), and $J(q(t)) \in \mathbb{R}^{m \times n}$ is the analytical system Jacobian.

In a remote teleoperation setup, the desired trajectory $x_d(t)$ is commanded by a human operator through a vision-based hand tracker. Due to webcam exposure latency, image processing time, and network transmission, the reference trajectory received by the controller experiences a time-varying delay $\tau(t)$. The following assumptions are made regarding the delay profile:

**Assumption 1**: The communication delay $\tau(t)$ is continuous and bounded such that:
$$0 \le \tau(t) \le \tau_m, \quad \forall t \ge 0$$
where $\tau_m$ is a known upper bound of the latency.

**Assumption 2**: The rate of change of the delay is bounded on both sides to prevent control signal chattering during sudden network recovery (jitter):
$$|\dot{\tau}(t)| \le d < 1, \quad \forall t \ge 0$$

Under delayed teleoperation, the task-space tracking error $e(t) \in \mathbb{R}^m$ is defined as:
$$e(t) = \begin{bmatrix} e_p(t) \\ e_o(t) \\ e_f(t) \end{bmatrix} = \begin{bmatrix} p_{d}(t-\tau(t)) - p_w(t) \\ \eta_d(t-\tau(t))\epsilon(t) - \eta(t)\epsilon_d(t-\tau(t)) - \epsilon^\times(t)\epsilon_d(t-\tau(t)) \\ d_{d}(t-\tau(t)) - d_f(t) \end{bmatrix}$$
where $e_o(t) \in \mathbb{R}^3$ represents the quaternion-based orientation error, and $\epsilon^\times$ is the skew-symmetric matrix of $\epsilon$.

The primary control objective is to design a joint velocity controller $\dot{q}(t)$ that guarantees asymptotic convergence of the tracking error ($\lim_{t \to \infty} \|e(t)\| = 0$) under Assumptions 1 and 2, while utilizing the null-space to avoid joint limits and self-collisions.

---

# IV. Proposed Scheme

## A. Singularity-Robust CLIK Control Law
To prevent numerical instability near kinematic singularities (such as during full finger extension), we employ a Damped Least-Squares (DLS) Jacobian inverse. The proposed control law is formulated as:
$$\dot{q}(t) = J^{\dagger}_{DLS}(q(t)) \left[ v_d(t - \tau(t))(1 - \dot{\tau}(t)) + K_p e(t) + K_d e(t - \tau(t)) \right] + \left( I_n - J^{\dagger}_{DLS}(q(t))J(q(t)) \right) \dot{q}_0(t)$$
where:
*   $J^{\dagger}_{DLS}(q) = J^T(JJ^T + \lambda^2 I_m)^{-1} \in \mathbb{R}^{n \times m}$ is the DLS pseudo-inverse, with $\lambda > 0$ being the damping factor.
*   $K_p, K_d \in \mathbb{R}^{m \times m}$ are positive-definite diagonal gain matrices.
*   $K_d e(t - \tau(t))$ is the discrete delay-feedback compensation term.
*   $\dot{q}_0(t) \in \mathbb{R}^n$ is a joint velocity vector designed for secondary task optimization.

Substituting the control law into the differential kinematics yields the closed-loop tracking error dynamics (assuming $\lambda \to 0$ in non-singular regions):
$$\dot{e}(t) = -K_p e(t) - K_d e(t - \tau(t))$$

## B. Stability Analysis via Lyapunov-Krasovskii Functional
To prove the stability of the closed-loop system under the time-varying communication delay, we construct the following candidate Lyapunov-Krasovskii Functional (LKF) $V(t)$:
$$V(t) = e^T(t) P e(t) + \int_{t - \tau(t)}^{t} e^T(s) Q e(s) ds + \tau_m \int_{-\tau_m}^{0} \int_{t + \theta}^{t} \dot{e}^T(s) R \dot{e}(s) ds d\theta$$
where $P, Q, R \in \mathbb{R}^{m \times m}$ are symmetric positive-definite weight matrices.

Differentiating $V(t)$ with respect to time along the trajectory of the tracking error dynamics yields:
$$\dot{V}(t) = 2 e^T(t) P \left[ -K_p e(t) - K_d e(t - \tau(t)) \right] + e^T(t) Q e(t) - (1 - \dot{\tau}(t)) e^T(t - \tau(t)) Q e(t - \tau(t)) + \tau_m^2 \dot{e}^T(t) R \dot{e}(t) - \tau_m \int_{t - \tau_m}^{t} \dot{e}^T(s) R \dot{e}(s) ds$$
Using Assumption 2 and applying Leibniz-Newton theorem with Jensen's inequality to the integral term:
$$-\tau_m \int_{t - \tau_m}^{t} \dot{e}^T(s) R \dot{e}(s) ds \le -\left[ e(t) - e(t-\tau(t)) \right]^T R \left[ e(t) - e(t-\tau(t)) \right]$$
We define the augmented state vector $\xi(t) = [e^T(t), e^T(t-\tau(t))]^T \in \mathbb{R}^{2m}$. The derivative $\dot{V}(t)$ can be bounded as:
$$\dot{V}(t) \le -\xi^T(t) \Omega \xi(t)$$
where the symmetric matrix $\Omega \in \mathbb{R}^{2m \times 2m}$ is explicitly defined as:
$$\Omega = \begin{bmatrix} 2 P K_p - Q + R - \tau_m^2 K_p^T R K_p & P K_d - R - \tau_m^2 K_p^T R K_d \\ * & (1 - d) Q + R - \tau_m^2 K_d^T R K_d \end{bmatrix}$$
where $*$ denotes the symmetric blocks. By solving the Linear Matrix Inequality (LMI) $\Omega > 0$ using Schur complement, we guarantee that $\dot{V}(t)$ is negative-definite, proving that the tracking error $e(t)$ asymptotically converges to zero.

## C. Null-Space Projection for Joint Limit and Collision Avoidance
The secondary joint velocity vector $\dot{q}_0(t)$ is designed to push the robot joints away from physical boundaries and avoid self-collisions. We define the objective function $H(q) \in \mathbb{R}$ to be minimized:
$$H(q) = \sum_{i=1}^{n} \left( \frac{q_i - \bar{q}_i}{q_{i,max} - q_{i,min}} \right)^2 + \sum_{j < k} \frac{\sigma}{d_{jk}^2(q)}$$
where $q_{i,max}, q_{i,min}$ are the physical boundaries of the $i$-th joint, $\bar{q}_i = \frac{q_{i,max} + q_{i,min}}{2}$, $d_{jk}(q)$ is the distance between capsule representations of link $j$ and link $k$ of the robot arm-hand, and $\sigma > 0$ is a safety weight. 

We define the null-space input vector $\dot{q}_0(t)$ as the negative gradient of $H(q)$:
$$\dot{q}_0(t) = -\alpha \nabla H(q)$$
where $\alpha > 0$ is a scalar step-size. Because $J \left( I_n - J^{\dagger}_{DLS}J \right) \approx 0$, the secondary task operates entirely within the null space, ensuring that safety constraints do not perturb Cartesian tracking performance.

---

## D. Practical Implementation Details
In practice, the delay $\tau(t)$ is calculated in real-time using network packet timestamps: $\tau(t) = t_{recv} - t_{send}$. Because raw latency measurements contain high-frequency noise (jitter), the derivative $\dot{\tau}(t)$ cannot be computed directly via numerical differentiation. 

To resolve this issue, the controller implements a continuous-time Low-Pass Filter (LPF) to smooth the delay estimate:
$$\tau_{filt}(s) = \frac{1}{T_f s + 1} \tau(s)$$
where $T_f > 0$ is the filter time constant. The filtered derivative $\dot{\tau}_{filt}(t)$ is then computed analytically from the filter state, ensuring a smooth and bounded $(1 - \dot{\tau}_{filt}(t))$ scaling term in the control loop. This CLIK controller operates as a high-frequency (100 Hz) teleoperation bridge in the Isaac Sim simulator, allowing the human operator to generate stable, chatter-free demonstration data for downstream Diffusion Policy training.
