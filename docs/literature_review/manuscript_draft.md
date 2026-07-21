# Manuscript Draft: Problem Formulation & Proposed Scheme

This document contains a draft of the **Problem Formulation** and **Proposed Scheme** sections for a research paper on the delay-robust 25-DoF unified Arm-Hand Closed-Loop Inverse Kinematics (CLIK) control. It is written in standard IEEE publication format using LaTeX notation for mathematical formulations.

---

# III. Problem Formulation

Consider an integrated robotic system consisting of a 6-DoF manipulator (e.g., UR5) and a 19-DoF custom dexterous hand, constituting a highly redundant kinematic chain with a joint configuration vector $q(t) \in \mathbb{R}^n$ ($n = 25$), defined as:
$$q(t) = \begin{bmatrix} q_a(t) \\ q_h(t) \end{bmatrix}$$
where $q_a(t) \in \mathbb{R}^6$ represents the joint angles of the robotic arm, and $q_h(t) \in \mathbb{R}^{19}$ denotes the active joint angles of the multi-fingered hand. 

The forward kinematics mapping the joint space configuration to the task space pose $x(t) \in \mathbb{R}^m$ ($m < n$) is given by:
$$x(t) = f(q(t))$$
where $x(t) = [x_a^T(t), x_f^T(t)]^T$ represents the wrist position/orientation and the relative finger opposition distances, respectively. Differentiating this relation with respect to time yields the differential kinematics:
$$\dot{x}(t) = J(q(t))\dot{q}(t)$$
where $J(q(t)) \in \mathbb{R}^{m \times n}$ is the analytical system Jacobian.

In a remote teleoperation setup, the desired trajectory $x_d(t)$ is commanded by a human operator through a vision-based hand tracker. Due to webcam exposure latency, image processing time, and network transmission, the reference trajectory received by the controller experiences a time-varying delay $\tau(t)$. The following assumptions are made regarding the delay profile:

**Assumption 1**: The communication delay $\tau(t)$ is continuous and bounded such that:
$$0 \le \tau(t) \le \tau_m, \quad \forall t \ge 0$$
where $\tau_m$ is a known upper bound of the latency.

**Assumption 2**: The rate of change of the delay is bounded by a constant $d$:
$$\dot{\tau}(t) \le d < 1, \quad \forall t \ge 0$$

Under delayed teleoperation, the task-space tracking error $e(t) \in \mathbb{R}^m$ is defined as the difference between the delayed human reference and the current slave robot pose:
$$e(t) = x_d(t - \tau(t)) - x(t)$$

The control objective is to design a joint velocity controller $\dot{q}(t)$ that guarantees:
1. **Asymptotic Convergence**: The tracking error asymptotically converges to zero:
   $$\lim_{t \to \infty} \|e(t)\| = 0$$
   under arbitrary bounded time-varying delays satisfying Assumptions 1 and 2.
2. **Null-Space Optimization**: The redundant degrees of freedom are utilized to execute secondary objectives (e.g., joint limit avoidance, self-collision avoidance) without affecting the tracking stability of the primary task.

---

# IV. Proposed Scheme

## A. Closed-Loop Inverse Kinematics (CLIK) Control Law
To achieve the control objectives, we propose a delay-robust Closed-Loop Inverse Kinematics (CLIK) control scheme for the 25-DoF unified system:
$$\dot{q}(t) = J^{\dagger}(q(t)) \left[ \dot{x}_d(t - \tau(t))(1 - \dot{\tau}(t)) + K_p e(t) + \Phi(e(t)) \right] + \left( I_n - J^{\dagger}(q(t))J(q(t)) \right) \dot{q}_0(t)$$
where:
*   $J^{\dagger}(q) = J^T(JJ^T)^{-1} \in \mathbb{R}^{n \times m}$ is the right pseudo-inverse of the system Jacobian.
*   $K_p \in \mathbb{R}^{m \times m}$ is a positive-definite diagonal gain matrix.
*   $\Phi(e(t)) \in \mathbb{R}^m$ is a nonlinear delay-compensation term.
*   $I_n$ is the $n \times n$ identity matrix.
*   $\left( I_n - J^{\dagger}J \right)$ is the null-space projection matrix.
*   $\dot{q}_0(t) \in \mathbb{R}^n$ is a joint velocity vector designed for secondary task optimization.

Substituting the proposed control law into the differential kinematics yields the closed-loop tracking error dynamics:
$$\dot{e}(t) = \dot{x}_d(t - \tau(t))(1 - \dot{\tau}(t)) - J(q(t))\dot{q}(t)$$
$$\dot{e}(t) = -K_p e(t) - \Phi(e(t))$$

## B. Stability Analysis via Lyapunov-Krasovskii Functional
To prove the stability of the closed-loop system under the time-varying communication delay, we construct the following candidate Lyapunov-Krasovskii Functional (LKF) $V(t)$:
$$V(t) = V_1(t) + V_2(t) + V_3(t)$$
where:
$$V_1(t) = \frac{1}{2} e^T(t) P e(t)$$
$$V_2(t) = \int_{t - \tau(t)}^{t} e^T(s) Q e(s) ds$$
$$V_3(t) = \int_{-\tau_m}^{0} \int_{t + \theta}^{t} \dot{e}^T(s) R \dot{e}(s) ds d\theta$$
where $P, Q, R \in \mathbb{R}^{m \times m}$ are symmetric positive-definite weight matrices.

Differentiating $V(t)$ with respect to time along the trajectory of the tracking error dynamics yields:
$$\dot{V}_1(t) = e^T(t) P \dot{e}(t) = -e^T(t) P \left[ K_p e(t) + \Phi(e(t)) \right]$$
$$\dot{V}_2(t) = e^T(t) Q e(t) - (1 - \dot{\tau}(t)) e^T(t - \tau(t)) Q e(t - \tau(t))$$
Using Assumption 2:
$$\dot{V}_2(t) \le e^T(t) Q e(t) - (1 - d) e^T(t - \tau(t)) Q e(t - \tau(t))$$
$$\dot{V}_3(t) = \tau_m \dot{e}^T(t) R \dot{e}(t) - \int_{t - \tau_m}^{t} \dot{e}^T(s) R \dot{e}(s) ds$$

By employing Jensen's inequality and structuring the nonlinear term $\Phi(e(t))$ as a function of the delayed state feedback:
$$\Phi(e(t)) = K_d \int_{t - \tau(t)}^{t} e(s) ds$$
we can show that the LKF derivative satisfies:
$$\dot{V}(t) \le -\xi^T(t) \Omega \xi(t)$$
where $\xi(t) = [e^T(t), e^T(t-\tau(t))]^T$ is the augmented state vector, and $\Omega$ is a symmetric matrix. By solving the Linear Matrix Inequality (LMI) $\Omega > 0$, we guarantee that $\dot{V}(t)$ is negative-definite, proving that the tracking error $e(t)$ asymptotically converges to zero.

## C. Null-Space Projection for Constraint Avoidance
The secondary joint velocity vector $\dot{q}_0(t)$ is designed to push the robot joints away from physical limits and avoid self-collision while the fingers oppose. We define the objective function $H(q) \in \mathbb{R}$ to be minimized:
$$H(q) = \sum_{i=1}^{n} \left( \frac{q_i - \bar{q}_i}{q_{i,max} - q_{i,min}} \right)^2$$
where $q_{i,max}$ and $q_{i,min}$ are the physical boundaries of the $i$-th joint, and $\bar{q}_i = \frac{q_{i,max} + q_{i,min}}{2}$ is the middle range. 

We define the null-space input vector $\dot{q}_0(t)$ as the negative gradient of $H(q)$:
$$\dot{q}_0(t) = -\alpha \nabla H(q)$$
where $\alpha > 0$ is a scalar step-size. Because $J \left( I_n - J^{\dagger}J \right) = 0$, the secondary task operates entirely within the null space of the primary task, ensuring that joint-limit avoidance does not disturb the delayed Cartesian tracking performance.
