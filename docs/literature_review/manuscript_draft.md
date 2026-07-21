# Manuscript Draft: Problem Formulation & Proposed Scheme (Revised - Ver 5)

This document contains the revised draft of the **Problem Formulation** and **Proposed Scheme** sections, resolving the theoretical inconsistencies in stochastic modeling, jitter boundedness, null-space saturation limits, and dexterous hand posture synergy.

---

# III. Problem Formulation

Consider an integrated robotic system consisting of a 6-DoF manipulator (UR5) and a 19-DoF custom dexterous hand, constituting a highly redundant kinematic chain with a joint configuration vector $q(t) \in \mathbb{R}^n$ ($n = 25$), defined as:
$$q(t) = \begin{bmatrix} q_a(t) \\ q_h(t) \end{bmatrix}$$
where $q_a(t) \in \mathbb{R}^6$ represents the joint angles of the robotic arm, and $q_h(t) \in \mathbb{R}^{19}$ denotes the active joint angles of the multi-fingered hand. 

The forward kinematics mapping the joint space configuration to the task space pose $x(t) \in \mathcal{X}$ is given by:
$$x(t) = f(q(t))$$
where the task space $\mathcal{X} = \mathbb{R}^3 \times \mathcal{S}^3 \times \mathbb{R}^{n_f}$ is designed with $n_f = 5$ relative Euclidean finger opposition distances (specifically, thumb-index, thumb-middle, thumb-ring, thumb-pinky, and index-pinky opposition pairs) to prevent kinematic over-constraint ($m = 3 + 3 + 5 = 11$ task-space dimensions, which is well below $n = 25$ total joints). 

Differentiating the forward kinematics yields the differential kinematics:
$$v(t) = J(q(t))\dot{q}(t)$$
where $v(t) = [\dot{p}_w^T(t), \omega_w^T(t), \dot{d}_f^T(t)]^T \in \mathbb{R}^m$ is the spatial task-space velocity vector ($\omega_w$ being the physical angular velocity of the wrist), and $J(q(t)) \in \mathbb{R}^{m \times n}$ is the analytical system Jacobian.

In a remote teleoperation setup, the desired trajectory $x_d(t)$ is commanded by a human operator through a vision-based hand tracker. The controller receives $x_d(t)$ under an unknown but deterministic time-varying communication delay $\tau(t)$, satisfying the following conditions:

**Assumption 1**: The communication delay $\tau(t)$ is continuous and bounded such that:
$$0 \le \tau(t) \le \tau_m, \quad \forall t \ge 0$$
where $\tau_m$ is a known upper bound of the latency.

**Assumption 2**: The rate of change of the true delay is bounded on both sides to prevent unstable tracking:
$$|\dot{\tau}(t)| \le d < 1, \quad \forall t \ge 0$$

**Assumption 3**: The measured latency consists of a true delay $\tau_{true}(t)$ satisfying Assumptions 1 and 2, and a high-frequency bounded measurement noise $n(t)$ representing network packet jitter:
$$\tau(t) = \tau_{true}(t) + n(t), \quad \|n(t)\| \le n_{max}$$

Under delayed teleoperation, the task-space tracking error $e(t) \in \mathbb{R}^m$ is defined as:
$$e(t) = \begin{bmatrix} e_p(t) \\ e_o(t) \\ e_f(t) \end{bmatrix} = \begin{bmatrix} p_{d}(t-\tau(t)) - p_w(t) \\ \eta_d(t-\tau(t))\epsilon(t) - \eta(t)\epsilon_d(t-\tau(t)) - \epsilon^\times(t)\epsilon_d(t-\tau(t)) \\ d_{d}(t-\tau(t)) - d_f(t) \end{bmatrix}$$
where $e_o(t) \in \mathbb{R}^3$ represents the quaternion-based orientation error, and $\epsilon^\times$ is the skew-symmetric matrix of $\epsilon$.

The primary control objective is to design a joint velocity controller $\dot{q}(t)$ that guarantees Input-to-State Stability (ISS) of the tracking error under Assumptions 1, 2, and 3, while utilizing the null-space to safely avoid joint limits, self-collisions, and unnatural hand configurations.

---

# IV. Proposed Scheme

## A. Singularity-Robust CLIK Control Law
To prevent numerical instability near kinematic singularities, we employ a Damped Least-Squares (DLS) Jacobian inverse. The proposed control law is formulated as:
$$\dot{q}(t) = J^{\dagger}_{DLS}(q(t)) \left[ v_{d,filt}(t) + K_p e(t) + K_d e(t - \tau(t)) \right] + \left( I_n - J^{\dagger}_{DLS}(q(t))J(q(t)) \right) \dot{q}_0(t)$$
where:
*   $J^{\dagger}_{DLS}(q) = J^T(JJ^T + \lambda^2 I_m)^{-1} \in \mathbb{R}^{n \times m}$ is the DLS pseudo-inverse, with $\lambda > 0$ being the damping factor.
*   $K_p, K_d \in \mathbb{R}^{m \times m}$ are positive-definite diagonal gain matrices.
*   $K_d e(t - \tau(t))$ is the discrete delay-feedback compensation term.
*   $v_{d,filt}(t) = v_d(t - \tau(t))(1 - \dot{\tau}_{filt}(t))$ is the feedforward velocity based on the filtered delay derivative.
*   $\dot{q}_0(t) \in \mathbb{R}^n$ is the joint velocity vector designed for secondary task optimization.

Substituting the control law into the differential kinematics yields the closed-loop tracking error dynamics:
$$\dot{e}(t) = -E(e_o) \left[ K_p e(t) + K_d e(t - \tau(t)) \right] + d_t(t)$$
where $E(e_o) = \text{diag}\left(I_3, \frac{1}{2}(\eta_e I_3 + \epsilon_e^\times), I_{5}\right)$ is the orientation scaling matrix, and $d_t(t)$ is the aggregate perturbation term:
$$d_t(t) = d_s(t) + d_f(t) + d_l(t)$$
*   $d_s(t) = \left( J J^{\dagger}_{DLS} - I_m \right) \left[ v_{d,filt}(t) + K_p e(t) + K_d e(t - \tau(t)) \right]$ is the damping perturbation.
*   $d_f(t) = v_d(t - \tau(t)) \left( \dot{\tau}_{filt}(t) - \dot{\tau}(t) \right)$ is the feedforward error caused by the low-pass filter phase lag. Because $n(t)$ and $v_d$ are bounded, the lag perturbation $d_f(t)$ remains bounded.
*   $d_l(t) = -J(q)\left( I_n - J^{\dagger}_{DLS}(q)J(q) \right)\dot{q}_0(t)$ is the workspace leakage originating from the DLS null-space projection.

**Remark 1 (Region of Attraction)**: Under local representation of orientation error, the region of attraction is defined as $\Omega_o = \{ e_o \in \mathbb{R}^3 \mid \eta_e > 0 \}$, corresponding to rotation errors of less than $180^\circ$. Within this domain, $E(e_o)$ is non-singular and converges to $I_m$ near the equilibrium state ($\eta_e \to 1, \epsilon_e \to 0$). Under this local linearization, the error dynamics simplify to:
$$\dot{e}(t) \approx -K_p e(t) - K_d e(t - \tau(t)) + d_t(t)$$

---

## B. Stability Analysis via Lyapunov-Krasovskii Functional
To prove the Input-to-State Stability (ISS) of the closed-loop error system, we construct the following candidate Lyapunov-Krasovskii Functional (LKF) $V(t)$:
$$V(t) = e^T(t) P e(t) + \int_{t - \tau(t)}^{t} e^T(s) Q e(s) ds + \tau_m \int_{-\tau_m}^{0} \int_{t + \theta}^{t} \dot{e}^T(s) R \dot{e}(s) ds d\theta$$
where $P, Q, R \in \mathbb{R}^{m \times m}$ are diagonal positive-definite matrices (hence $P K_p = K_p P$, $P K_d = K_d P$).

Differentiating $V(t)$ along the trajectory of the perturbed error dynamics yields:
$$\dot{V}(t) = 2 e^T(t) P \left[ -K_p e(t) - K_d e(t - \tau(t)) + d_t(t) \right] + e^T(t) Q e(t) - (1 - \dot{\tau}(t)) e^T(t - \tau(t)) Q e(t - \tau(t)) + \tau_m^2 \dot{e}^T(t) R \dot{e}(t) - \tau_m \int_{t - \tau_m}^{t} \dot{e}^T(s) R \dot{e}(s) ds$$

Using Young's inequality, the bilinear perturbation term is bounded by:
$$2 e^T(t) P d_t(t) \le \epsilon_1 e^T(t) e(t) + \frac{1}{\epsilon_1} d_t^T(t) P^2 d_t(t)$$
where $\epsilon_1 > 0$.

Let $\phi_e(t) = -K_p e(t) - K_d e(t - \tau(t))$. The acceleration-related LKF term contains:
$$\dot{e}^T(t) R \dot{e}(t) = \left[ \phi_e(t) + d_t(t) \right]^T R \left[ \phi_e(t) + d_t(t) \right]$$
Applying Young's inequality to the cross-term $2 \phi_e^T(t) R d_t(t)$ yields:
$$\dot{e}^T(t) R \dot{e}(t) \le (1 + \epsilon_2) \phi_e^T(t) R \phi_e(t) + \left( 1 + \frac{1}{\epsilon_2} \right) d_t^T(t) R d_t(t)$$
where $\epsilon_2 > 0$.

To handle the delay-dependent integral term, we partition the integration interval into two segments: $[t - \tau_m, t - \tau(t)]$ and $[t - \tau(t), t]$. Applying Jensen's inequality to each segment yields:
$$-\tau_m \int_{t - \tau(t)}^{t} \dot{e}^T(s) R \dot{e}(s) ds \le -\frac{\tau_m}{\tau(t)} a^T(t) R a(t)$$
$$-\tau_m \int_{t - \tau_m}^{t-tau(t)} \dot{e}^T(s) R \dot{e}(s) ds \le -\frac{\tau_m}{\tau_m - \tau(t)} b^T(t) R b(t)$$
where $a(t) = e(t) - e(t-\tau(t))$ and $b(t) = e(t-\tau(t)) - e(t-\tau_m)$.

Let $\eta(t) = [e^T(t), e^T(t-\tau(t)), e^T(t-\tau_m)]^T \in \mathbb{R}^{3m}$ be the augmented state vector. Applying the **reciprocal convexity lemma** (Park et al., 2011), if there exists a matrix $S \in \mathbb{R}^{m \times m}$ such that $\begin{bmatrix} R & S \\ S^T & R \end{bmatrix} \ge 0$, the partitioned segments can be bounded. We obtain:
$$\dot{V}(t) \le -\eta^T(t) \Omega_{ISS} \eta(t) + \gamma \|d_t(t)\|^2$$
where the ISS-gain parameter $\gamma$ is explicitly formulated as:
$$\gamma = \frac{1}{\epsilon_1}\lambda_{max}(P^2) + \tau_m^2 \left( 1 + \frac{1}{\epsilon_2} \right) \lambda_{max}(R)$$
and the $3m \times 3m$ matrix $\Omega_{ISS}$ is defined as:
$$\Omega_{ISS} = \begin{bmatrix} 2 P K_p - Q - \epsilon_1 I_m & P K_d & 0 \\ * & (1-d)Q & 0 \\ * & * & 0 \end{bmatrix} + \begin{bmatrix} R & S-R & -S \\ * & 2R-S-S^T & S-R \\ * & * & R \end{bmatrix} - \tau_m^2 (1 + \epsilon_2) \begin{bmatrix} K_p^T R K_p & K_p^T R K_d & 0 \\ * & K_d^T R K_d & 0 \\ * & * & 0 \end{bmatrix}$$

By solving the LMI $\Omega_{ISS} > 0$ under the constraint $\begin{bmatrix} R & S \\ S^T & R \end{bmatrix} \ge 0$, the derivative satisfies $\dot{V}(t) \le -\lambda_{min}(\Omega_{ISS}) \|\eta(t)\|^2 + \gamma \|d_t(t)\|^2$, proving the Input-to-State Stability (ISS) of the tracking error.

---

## C. Null-Space Projection for Joint Limit, Collision, and Posture Avoidance
The secondary joint velocity vector $\dot{q}_0(t)$ is designed to push the robot joints away from physical boundaries, avoid self-collisions, and guide the redundant hand joints toward natural anthropomorphic configurations to resolve the $19 - 5 = 14$ DoFs under-constraint. We define the objective function $H(q) \in \mathbb{R}$ to be minimized:
$$H(q) = w_{lim} \sum_{i=1}^{n} \left( \frac{q_i - \bar{q}_i}{q_{i,max} - q_{i,min}} \right)^2 + H_{coll}(q) + w_{post} \sum_{j=7}^{25} \left( q_j - q_{j,nom} \right)^2$$
where:
*   $q_{i,max}, q_{i,min}$ are the physical boundaries of the $i$-th joint, and $\bar{q}_i = \frac{q_{i,max} + q_{i,min}}{2}$.
*   $q_{j,nom}$ denotes the nominal natural posture joint angle of the $j$-th hand joint, calibrated from the operator's relaxed hand position.
*   $w_{lim}$ and $w_{post}$ are positive weight parameters.

To prevent gradient explosion and chattering in discrete-time execution (100 Hz), the collision potential $H_{coll}(q)$ is active only within a finite influence distance $d_{influence} > 0$:
$$H_{coll}(q) = \sum_{a < b} \frac{\sigma}{2} \left( \frac{1}{d_{ab}(q)} - \frac{1}{d_{influence}} \right)^2 \quad \text{if } d_{ab}(q) \le d_{influence}, \text{ else } 0$$
where $d_{ab}(q)$ is the distance between capsule representations of link $a$ and link $b$. 

To ensure the perturbation vector $d_l(t)$ remains bounded in the LKF analysis, the gradient mapping is saturated to bound the norm of the potential vector:
$$\dot{q}_0(t) = \text{sat}_{v_{max}} \left( -\alpha \nabla H(q) \right)$$
where the saturation limits the magnitude of $\dot{q}_0(t)$ to prevent infinite joint speeds.

**Remark 2 (SVD-based Null-Space Leakage)**: Applying singular value decomposition (SVD) to the Jacobian $J = U \Sigma V^T$, the workspace leakage introduced by the DLS projection is given by $J \left( I_n - J^{\dagger}_{DLS}J \right) = U \text{diag}\left( \frac{\sigma_i \lambda^2}{\sigma_i^2 + \lambda^2} \right) V^T$. By AM-GM inequality, each diagonal entry is bounded by $\frac{\lambda}{2}$ for all $\sigma_i$. As the system approaches a deep singularity ($\sigma_i \to 0$), the leakage converges to zero. The maximum leakage is bounded globally by $\mathcal{O}(\lambda)$ at near-singular configurations ($\sigma_i \approx \lambda$), ensuring that the secondary tasks do not corrupt workspace accuracy.

---

## D. Practical Implementation Details
In practice, the delay $\tau(t)$ is calculated in real-time using network packet timestamps: $\tau(t) = t_{recv} - t_{send}$. Because raw latency measurements contain high-frequency noise (jitter), the derivative $\dot{\tau}(t)$ cannot be computed directly via numerical differentiation. 

To resolve this issue, the controller implements a continuous-time Low-Pass Filter (LPF):
$$\tau_{filt}(s) = \frac{1}{T_f s + 1} \tau(s)$$
where the filter time constant $T_f > 0$ is tuned empirically during teleoperation trials such that the filtered derivative satisfies $|\dot{\tau}_{filt}(t)| \le d < 1$. This CLIK controller operates as a high-frequency (100 Hz) teleoperation bridge in the Isaac Sim simulator, allowing the human operator to generate stable, chatter-free demonstration data for downstream Diffusion Policy training.
