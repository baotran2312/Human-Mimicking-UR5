# Lecture Slides: Delay-Robust CLIK Control for Unified 25-DoF Arm-Hand Systems

========== PAGE 1 ==========

=== SLIDE 1 ===
HEADER: Nonlinear Control | 1 | Title
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

[Big centered title: "Delay-Robust Closed-Loop Inverse Kinematics Control for Unified 25-DoF Arm-Hand Dexterous Teleoperation Systems"]

• Authors: Thien Bao Tran, The Tri Bui, Huu Tran Nhat Le, and Ha Quang Thinh Ngo
• HCM City University of Technology (HCMUT) & FPT University, Vietnam
• Focus: 25-DoF Arm-Hand Control, Communication Delay, Singularity Avoidance, ISS Stability.


=== SLIDE 2 ===
HEADER: Nonlinear Control | 2 | §1. Introduction
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§1. Introduction: Vision-Based Teleoperation & Challenges

• Vision-Based Teleoperation:
  + Key for remote manipulation using 25-DoF arm-hand systems
  + Learning policies (\cite{ref_diffpolicy3d}, \cite{ref_flowdiff}) require smooth human demonstration datasets
  + Markerless trackers (\cite{ref_anyteleop}, \cite{ref_dexsim2real}) use egocentric camera views \cite{ref_egocentric3d, ref_deltadorsal}
• Latency & Packet Jitter Challenges:
  + Over wireless networks, time-varying communication delays $\tau(t)$ are inevitable \cite{ref_tase_tele}
  + Jitter creates non-differentiable delay trajectories, breaking standard feedforward velocity terms
  + Kinematics of 25-DoF arm-hand are highly redundant, prone to self-collisions and singularities


=== SLIDE 3 ===
HEADER: Nonlinear Control | 3 | §2. Related Work
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§2. Related Work: State-of-the-Art (SOTA) Comparison

[COLOR: Table I: Comparison of Robotic Teleoperation Controllers]
• Target Systems: Mostly limited to 2-DoF \cite{ref_tase_tele} or 7-DoF manipulators \cite{ref_tcst_delay}
• Delay compensation: sliding mode \cite{ref_iecon_sliding}, neural nets \cite{ref_tcyb_nn}, event-trigger \cite{ref_tcyb_event}
• Stability analysis: Lyapunov-Krasovskii Functional (LKF) \cite{ref_tcs_haptic} or passivity \cite{ref_tro_rcm}

• State-of-the-Art (SOTA) Matrix:
  + Li et al. \cite{ref_cybernetics_armhand}: 25-DoF, No delay, No stability proof, Limits only
  + AnyTeleop \cite{ref_anyteleop}: 26-DoF, No delay, Kinematic heuristic, Limit + Collision
  + Jing et al. \cite{ref_tase_tele}: 2-DoF, Time-varying delay, Lyapunov stability, None
  + Proposed CLIK: 25-DoF, Time-varying+Jitter, Rigorous LKF-LMI LISS, Limits+Collision+Synergy


=== SLIDE 4 ===
HEADER: Nonlinear Control | 4 | §3. Problem Formulation
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§3. Problem Formulation: Unified 25-DoF Kinematics

• Unified Joint Space Configuration $q(t) \in \mathbb{R}^{25}$:
  $$q(t) = [q_a(t)^T, q_h(t)^T]^T$$
  + $q_a(t) \in \mathbb{R}^6$: UR5 robotic arm joints
  + $q_h(t) \in \mathbb{R}^{19Active}$: Custom multi-fingered hand active joints
• Task Space Pose $x(t) \in \mathcal{X}$:
  + Relative Euclidean finger opposition distances ($n_f = 5$) used to prevent over-constraint
  + Task dimension $m = 3 + 3 + 5 = 11$, total joints $n = 25$ (highly redundant system)
• Differential Kinematics:
  $$v(t) = J(q(t)) \dot{q}(t)$$
  + $J(q(t)) \in \mathbb{R}^{11 \times 25}$: Analytical system Jacobian
  + $v(t) \in \mathbb{R}^{11}$: Spatial task-space velocity vector


=== SLIDE 5 ===
HEADER: Nonlinear Control | 5 | §3. Problem Formulation
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§3. Problem Formulation: Network Delay & Filtered Reference

• [COLOR: True-and-Noise Delay Model] (Assumption 3):
  $$\tau(t) = \tau_{true}(t) + n(t), \quad \|n(t)\| \le n_{max}$$
  + $\tau_{true}(t)$: Continuous, bounded true delay ($0 \le \tau_{true} \le \tau_m$, $|\dot{\tau}_{true}| \le d < 1$)
  + $n(t)$: Non-differentiable network packet jitter (noise)
• Filtered Reference Trajectory $x_{d,filt}(t)$ (LPF output):
  $$x_{d,filt}(t) = x_d(t - \tau_{filt}(t))$$
  + $\tau_{filt}(t)$ is continuous, differentiable, and $|\dot{\tau}_{filt}(t)| \le d < 1$
• Tracking Error Redefinition:
  + Redefined relative to $x_{d,filt}(t)$ to decouple non-differentiable jitter $n(t)$
  + Jitter acts as a bounded spatial tracking offset in the task space


=== SLIDE 6 ===
HEADER: Nonlinear Control | 6 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Singularity-Robust CLIK Law

• Control Objective:
  + Design joint velocity controller $\dot{q}(t)$ guaranteeing local ISS of tracking error
  + Systematically avoid singularities, joint limits, and collisions
• Singularity-Robust Damped Least-Squares (DLS) Law:
  $$\dot{q}(t) = J^{\dagger}_{DLS} [v_{d,filt}(t) + K_p e(t) + K_d e(t - \tau_{filt}(t))] + \left( I_{25} - J^{\dagger}_{DLS} J \right) \dot{q}_0(t)$$
  + $J^{\dagger}_{DLS} = J^T(JJ^T + \lambda^2 I)^{-1}$: DLS pseudo-inverse ($\lambda > 0$)
  + $K_p, K_d$: Positive-definite diagonal gain matrices
  + $K_d e(t - \tau_{filt}(t))$: Discrete delay-feedback compensation term
  + $\dot{q}_0(t)$: Secondary joint velocity vector for null-space projection

[BLOCK DIAGRAM: x_d(t) -> [Filter/Delay] -> x_d,filt -> [CLIK Controller] -> \dot{q}(t) -> [Robot System] -> x(t)]


========== PAGE 2 ==========

=== SLIDE 7 ===
HEADER: Nonlinear Control | 7 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: LKF Stability Analysis

• Augmented Tracking Error Dynamics:
  $$\dot{e}(t) = -K_p e(t) - K_d e(t - \tau_{filt}(t)) + d_t(t) + \Delta E(e_o)[K_p e(t) + K_d e(t - \tau_{filt}(t))]$$
  + $d_t(t)$: Total workspace perturbation (damping loss & null-space leakage)
  + $\Delta E(e_o) = I_m - E(e_o)$: Linearization residual of orientation error
• Candidate Lyapunov-Krasovskii Functional (LKF):
  $$V(t) = e^T(t) P e(t) + \int_{t - \tau_{filt}(t)}^{t} e^T(s) Q e(s) ds + \tau_m \int_{-\tau_m}^{0} \int_{t + \theta}^{t} \dot{e}^T(s) R \dot{e}^T(s) ds d\theta$$
  + $P, Q, R$: Positive-definite diagonal weight matrices
  + Integral terms accumulate energy stored in time-varying tracking errors


=== SLIDE 8 ===
HEADER: Nonlinear Control | 8 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Stability Proof via LMI

• Interval Partitioning & Bounding:
  + Integration intervals split into $[t - \tau_m, t - \tau_{filt}(t)]$ and $[t - \tau_{filt}(t), t]$
  + Jensen's inequality bounds the integration of acceleration terms
• Reciprocal Convexity Lemma (Park et al., 2011):
  + Combines partitioned segments using decision variable $S$
  + If LMI $\Omega_{ISS} = \Omega_1 + \Omega_2 - \Omega_3 > 0$ is satisfied, then:
    $$\dot{V}(t) \le -\lambda_{min}(\Omega_{ISS}) \|\eta(t)\|^2 + \gamma \|d_t(t)\|^2$$
  + $\eta(t) = [e^T(t), e^T(t-\tau_{filt}(t)), e^T(t-\tau_m)]^T \in \mathbb{R}^{3m}$
• Local Input-to-State Stability (LISS):
  + Tracking error converges to a bounded region proportional to perturbation $\|d_t(t)\|$


=== SLIDE 9 ===
HEADER: Nonlinear Control | 9 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Null-Space Multi-Objective Optimization

• Multi-Objective Potential Function $H(q) \in \mathbb{R}$ to minimize:
  $$H(q) = w_{lim} H_{lim}(q) + H_{coll}(q) + w_{post} H_{post}(q)$$
• 1. Joint Limit Avoidance $H_{lim}(q)$:
  $$H_{lim}(q) = \sum \left( \frac{q_i - \bar{q}_i}{q_{i,max} - q_{i,min}} \right)^2$$
• 2. Nominal Posture Synergy $H_{post}(q)$:
  $$H_{post}(q) = \sum \left( q_j - q_{j,nom} \right)^2$$
  + Constrains the 14-DoF hand redundancy to natural human poses
• 3. Real-Time Self-Collision Avoidance $H_{coll}(q)$:
  $$H_{coll}(q) = \sum h_{ab}(q)$$
  + Active only within capsule-pair influence distance $d_{inf}$:
    $$h_{ab}(q) = \frac{\sigma}{2} \left( \frac{1}{d_{ab}(q)} - \frac{1}{d_{inf}} \right)^2 \quad (\text{if } d_{ab} \le d_{inf})$$


=== SLIDE 10 ===
HEADER: Nonlinear Control | 10 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Gradient Saturation & Leakage Bounds

• Gradient Saturation (Choke Velocity):
  $$\dot{q}_0(t) = \text{sat}_{v_{max}} \left( -\alpha \nabla H(q) \right)$$
  + Element-wise saturation prevents command chattering and velocity explosion
  + Bounded potential vector norm: $\|\dot{q}_0(t)\| \le \sqrt{n} v_{max}$
  + Ensures DLS leakage perturbation $d_l(t)$ remains globally bounded
• [COLOR: Theorem: SVD-based Null-Space Leakage Bound] (Remark 2):
  + DLS pseudo-inverse creates task-space null-space projection leakage
  + Using Singular Value Decomposition $J = U \Sigma V^T$:
    $$J \left( I_{25} - J^{\dagger}_{DLS} J \right) = U \text{diag} \left( \frac{\sigma_i \lambda^2}{\sigma_i^2 + \lambda^2} \right) V^T$$
  + Diagonal elements are bounded by $\lambda/2$ for all $\sigma_i$ (by AM-GM inequality)
  + Leakage converges to $0$ as $\sigma_i \to 0$ (deep singularity)


=== SLIDE 11 ===
HEADER: Nonlinear Control | 11 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Practical Implementation

• Latency Low-Pass Filtering (LPF):
  + Raw packet latency $\tau(t) = t_{recv} - t_{send}$ processed in continuous-time:
    $$\tau_{filt}(s) = \frac{1}{T_f s + 1} \tau(s)$$
• [COLOR: LPF Constraint for LKF Stability]:
  + To strictly guarantee $|\dot{\tau}_{filt}(t)| \le d$, filter time constant $T_f$ must satisfy:
    $$T_f \ge \frac{\max |\tau(t) - \tau_{filt}(t)|}{d}$$
• Real-Time Collision Avoidance @ 100 Hz:
  + Computing collision gradients requires closest-point Jacobians:
    $$\frac{\partial d_{ab}(q)}{\partial q} = \hat{n}_{ab}^T J_{ab}(q)$$
  + Bounding-Volume Hierarchy (BVH) / Spatial Hashing culls inactive pairs ($d_{ab} > d_{inf}$)
  + Isolates Jacobian calculations only to active proximity sets


=== SLIDE 12 ===
HEADER: Nonlinear Control | 12 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: CLIK Control Algorithm

• CLIK Execution Loop (Algorithm 1):
  + 1. Measure delay: $\tau(t) = t_{recv} - t_{send}$
  + 2. LPF delay: $\dot{\tau}_{filt}(t) = \frac{1}{T_f} (\tau(t) - \tau_{filt}(t))$
  + 3. Compute reference trajectory $x_{d,filt}(t)$ and tracking error $e(t)$
  + 4. Solve DLS pseudo-inverse $J^{\dagger}_{DLS} = J^T\left(J J^T + \lambda^2 I_{11}\right)^{-1}$
  + 5. Cull capsule-pairs ($d_{ab} > d_{inf}$) using BVH spatial hashing
  + 6. Compute gradients $\nabla H$ and saturate null-space command $\dot{q}_0(t)$
  + 7. Calculate joint command velocity $\dot{q}(t)$ and apply to joints

• Experimental HIL Validation (Lab):
  + Laptop-PC real-time link achieved 30 FPS under 29-38 ms latency
  + 0% packet loss via flat binary 100-byte UDP stream (25 floats)
  + Result: Smooth, chatter-free teleoperation suitable for policy learning
