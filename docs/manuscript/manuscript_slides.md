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
  + Learning policies (3D Diffusion [1], 3D Flow [2]) require smooth human demonstration datasets
  + Markerless trackers (AnyTeleop [8], DexSim2Real [14]) use egocentric camera views [3, 12]
• Latency & Packet Jitter Challenges:
  + Over wireless networks, time-varying communication delays tau(t) are inevitable [15]
  + Jitter creates non-differentiable delay trajectories, breaking standard feedforward velocity terms
  + Kinematics of 25-DoF arm-hand are highly redundant, prone to self-collisions and singularities


=== SLIDE 3 ===
HEADER: Nonlinear Control | 3 | §2. Related Work
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§2. Related Work: State-of-the-Art (SOTA) Comparison

[COLOR: Table I: Comparison of Robotic Teleoperation Controllers]
• Target Systems: Mostly limited to 2-DoF [15] or 7-DoF manipulators [18]
• Delay compensation: sliding mode [29], neural nets [16], event-trigger [21]
• Stability analysis: Lyapunov-Krasovskii Functional (LKF) [30] or passivity [26]

• State-of-the-Art (SOTA) Matrix:
  + Li et al. [4]: 25-DoF, No delay, No stability proof, Limits only
  + AnyTeleop [8]: 26-DoF, No delay, Kinematic heuristic, Limit + Collision
  + Jing et al. [15]: 2-DoF, Time-varying delay, Lyapunov stability, None
  + Proposed CLIK: 25-DoF, Time-varying+Jitter, Rigorous LKF-LMI LISS, Limits+Collision+Synergy


=== SLIDE 4 ===
HEADER: Nonlinear Control | 4 | §3. Problem Formulation
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§3. Problem Formulation: Unified 25-DoF Kinematics

• Unified Joint Space Configuration q(t) in R^25:
    q(t) = [ qa(t)^T, qh(t)^T ]^T
  + qa(t) in R^6: UR5 robotic arm joints
  + qh(t) in R^19: Custom multi-fingered hand active joints
• Task Space Pose x(t) in X:
  + Relative Euclidean finger opposition distances (nf = 5) used to prevent over-constraint
  + Task dimension m = 3 + 3 + 5 = 11, total joints n = 25 (highly redundant system)
• Differential Kinematics:
    v(t) = J(q(t)) * q_dot(t)
  + J(q(t)) in R^(11x25): Analytical system Jacobian
  + v(t) in R^11: Spatial task-space velocity vector (wrist pose + finger distances)


=== SLIDE 5 ===
HEADER: Nonlinear Control | 5 | §3. Problem Formulation
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§3. Problem Formulation: Network Delay & Filtered Reference

• [COLOR: True-and-Noise Delay Model] (Assumption 3):
    tau(t) = tau_true(t) + n(t),   ||n(t)|| <= n_max
  + tau_true(t): Continuous, bounded true delay (0 <= tau_true <= tau_m, |tau_true_dot| <= d < 1)
  + n(t): Non-differentiable network packet jitter (noise)
• Filtered Reference Trajectory x_d,filt(t) (LPF output):
    x_d,filt(t) = x_d(t - tau_filt(t))
  + tau_filt(t) is continuous, differentiable, and |tau_filt_dot(t)| <= d < 1
• Tracking Error Redefinition:
  + Redefined relative to x_d,filt(t) to decouple non-differentiable jitter n(t)
  + Jitter acts as a bounded spatial tracking offset in the task space


=== SLIDE 6 ===
HEADER: Nonlinear Control | 6 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Singularity-Robust CLIK Law

• Control Objective:
  + Design joint velocity controller q_dot(t) guaranteeing local ISS of tracking error
  + Systematically avoid singularities, joint limits, and collisions
• Singularity-Robust Damped Least-Squares (DLS) Law:
    q_dot(t) = J_DLS_dagger * [v_d,filt(t) + K_p * e(t) + K_d * e(t - tau_filt(t))] + (I - J_DLS_dagger * J) * q0_dot(t)
  + J_DLS_dagger = J^T * (J * J^T + lambda^2 * I)^(-1): DLS pseudo-inverse (lambda > 0)
  + K_p, K_d: Positive-definite diagonal gain matrices
  + K_d * e(t - tau_filt(t)): Discrete delay-feedback compensation term
  + q0_dot(t): Secondary joint velocity vector for null-space projection

[BLOCK DIAGRAM: x_d(t) -> [Filter/Delay] -> x_d,filt -> [CLIK Controller] -> q_dot(t) -> [Robot System] -> x(t)]


========== PAGE 2 ==========

=== SLIDE 7 ===
HEADER: Nonlinear Control | 7 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: LKF Stability Analysis

• Augmented Tracking Error Dynamics:
    e_dot(t) = -K_p * e(t) - K_d * e(t - tau_filt(t)) + dt(t) + delta_E(eo) * [K_p * e(t) + K_d * e(t - tau_filt(t))]
  + dt(t): Total workspace perturbation (damping loss & null-space leakage)
  + delta_E(eo) = I_m - E(eo): Linearization residual of orientation error
• Candidate Lyapunov-Krasovskii Functional (LKF):
    V(t) = e(t)^T * P * e(t) + integral_{t-tau_filt}^{t} [ e(s)^T * Q * e(s) ] ds + tau_m * integral_{-tau_m}^{0} integral_{t+theta}^{t} [ e_dot(s)^T * R * e_dot(s) ] ds d_theta
  + P, Q, R: Positive-definite diagonal weight matrices
  + Integral terms accumulate energy stored in time-varying tracking errors


=== SLIDE 8 ===
HEADER: Nonlinear Control | 8 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Stability Proof via LMI

• Interval Partitioning & Bounding:
  + Integration intervals split into [t - tau_m, t - tau_filt(t)] and [t - tau_filt(t), t]
  + Jensen's inequality bounds the integration of acceleration terms
• Reciprocal Convexity Lemma (Park et al., 2011):
  + Combines partitioned segments using decision variable S
  + If LMI Omega_ISS = Omega_1 + Omega_2 - Omega_3 > 0 is satisfied, then:
    V_dot(t) <= -lambda_min(Omega_ISS) * ||eta(t)||^2 + gamma * ||dt(t)||^2
  + eta(t) = [ e(t)^T, e(t-tau_filt(t))^T, e(t-tau_m)^T ]^T in R^3m
• Local Input-to-State Stability (LISS):
  + Tracking error converges to a bounded region proportional to perturbation ||dt(t)||


=== SLIDE 9 ===
HEADER: Nonlinear Control | 9 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Null-Space Multi-Objective Optimization

• Multi-Objective Potential Function H(q) in R to minimize:
    H(q) = w_lim * H_lim(q) + H_coll(q) + w_post * H_post(q)
• 1. Joint Limit Avoidance H_lim(q):
    H_lim(q) = sum [ ((q_i - q_bar_i) / (q_i,max - q_i,min))^2 ]
• 2. Nominal Posture Synergy H_post(q):
    H_post(q) = sum [ (q_j - q_j,nom)^2 ]
  + Constrains the 14-DoF hand redundancy to natural human poses
• 3. Real-Time Self-Collision Avoidance H_coll(q):
    H_coll(q) = sum [ h_ab(q) ]
  + Active only within capsule-pair influence distance d_inf:
    h_ab(q) = (sigma / 2) * (1 / d_ab(q) - 1 / d_inf)^2   (if d_ab <= d_inf)


=== SLIDE 10 ===
HEADER: Nonlinear Control | 10 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Gradient Saturation & Leakage Bounds

• Gradient Saturation (Choke Velocity):
    q0_dot(t) = sat_v_max (-alpha * grad H(q))
  + Element-wise saturation prevents command chattering and velocity explosion
  + Bounded potential vector norm: ||q0_dot(t)|| <= sqrt(n) * v_max
  + Ensures DLS leakage perturbation dl(t) remains globally bounded
• [COLOR: Theorem: SVD-based Null-Space Leakage Bound] (Remark 2):
  + DLS pseudo-inverse creates task-space null-space projection leakage
  + Using Singular Value Decomposition J = U * S * V^T:
    J * (I - J_DLS_dagger * J) = U * diag( s_i * lambda^2 / (s_i^2 + lambda^2) ) * V^T
  + Diagonal elements are bounded by lambda / 2 for all s_i (by AM-GM inequality)
  + Leakage converges to 0 as s_i -> 0 (deep singularity)


=== SLIDE 11 ===
HEADER: Nonlinear Control | 11 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: Practical Implementation

• Latency Low-Pass Filtering (LPF):
  + Raw packet latency tau(t) = t_recv - t_send processed in continuous-time:
    tau_filt(s) = [ 1 / (T_f * s + 1) ] * tau(s)
• [COLOR: LPF Constraint for LKF Stability]:
  + To strictly guarantee |tau_filt_dot(t)| <= d, filter time constant T_f must satisfy:
    T_f >= ( max |tau(t) - tau_filt(t)| ) / d
• Real-Time Collision Avoidance @ 100 Hz:
  + Computing collision gradients requires closest-point Jacobians:
    partial d_ab(q) / partial q = n_bar_ab^T * J_ab(q)
  + Bounding-Volume Hierarchy (BVH) / Spatial Hashing culls inactive pairs (d_ab > d_inf)
  + Isolates Jacobian calculations only to active proximity sets


=== SLIDE 12 ===
HEADER: Nonlinear Control | 12 | §4. Proposed Scheme
FOOTER: HCM City Univ. of Technology, Faculty of Mechanical Engineering | Nguyen Tan Tien

§4. Proposed Scheme: CLIK Control Algorithm

• CLIK Execution Loop (Algorithm 1):
  + 1. Measure delay: tau(t) = t_recv - t_send
  + 2. LPF delay: tau_filt_dot(t) = (1 / T_f) * (tau(t) - tau_filt(t))
  + 3. Compute reference trajectory x_d,filt(t) and tracking error e(t)
  + 4. Solve DLS pseudo-inverse J_DLS_dagger = J^T * (J * J^T + lambda^2 * I)^(-1)
  + 5. Cull capsule-pairs (d_ab > d_inf) using BVH spatial hashing
  + 6. Compute gradients grad H and saturate null-space command q0_dot(t)
  + 7. Calculate joint command velocity q_dot(t) and apply to joints

• Experimental HIL Validation (Lab):
  + Laptop-PC real-time link achieved 30 FPS under 29-38 ms latency
  + 0% packet loss via flat binary 100-byte UDP stream (25 floats)
  + Result: Smooth, chatter-free teleoperation suitable for policy learning
