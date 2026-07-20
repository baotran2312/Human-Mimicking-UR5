# **Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation**

# **ByteDance Seed**

Full Author List in Contributions

# **Abstract**

Human-like dexterous hands with multiple fingers offer human-level manipulation capabilities, but training control policies that can directly deploy on real hardware remains difficult due to contactrich physics and imperfect actuation. We close this gap with a practical sim-to-real reinforcement learning (RL) framework that utilizes dense tactile feedback combined with joint torque sensing to explicitly regulate physical interactions. To enable effective sim-to-real transfer, we introduce (i) a computationally fast tactile simulation that computes distances between dense virtual tactile units and the object via parallel forward kinematics, providing high-rate, high-resolution touch signals needed by RL; (ii) a current-to-torque calibration that eliminates the need for torque sensors on dexterous hands by mapping motor current to joint torque; and (iii) actuator dynamics modeling to bridge the actuation gaps with randomization of non-ideal effects such as backlash, torque–speed saturation.

Using an asymmetric actor–critic PPO pipeline trained entirely in simulation, our policies deploy directly to a five-finger hand. The resulting policies demonstrated two essential skills: (1) command-based, controllable grasp force tracking, and (2) reorientation of objects in the hand, both of which were robustly executed without fine-tuning on the robot. By combining tactile and torque in the observation space with effective sensing/actuation modeling, our system provides a practical solution to achieve reliable dexterous manipulation. To our knowledge, this is the first demonstration of controllable grasping on a multi-finger dexterous hand trained entirely in simulation and transferred zero-shot on real hardware.

**Date:** 22 December, 2025

**Project Page:** <https://dexmanip-seed.github.io/dexmanip>

# **1 Introduction**

Reinforcement learning with sim-to-real has achieved remarkable success in locomotion community, leading to widespread industrial use cases of quadruped and humanoid robots. Pioneering works such as ANYmal's agile locomotion [\[22\]](#page-21-0) and soccer-playing humanoids [\[12,](#page-20-0) [39\]](#page-22-0) have demonstrated that a systematic design of domain randomization techniques, combined with RL techniques such as privileged learning of actuator dynamics and injection of noise in sensory observations, can bridge the reality gap for complex whole-body locomotion skills. Research in adaptive legged locomotion [\[43\]](#page-22-1) further shows that while standard Multilayer Perceptron (MLP) architectures sufficed for learning individual narrow control policies, achieving complex multi-skill behaviors often necessitates the orchestration of a mixture of specialized neural network experts.

Regarding sim-to-real techniques, through methods like domain randomization, system identification, and learning actuator compensation model [\[7,](#page-20-1) [16,](#page-20-2) [22,](#page-21-0) [25\]](#page-21-1), these technologies reduce the gap between the robot model itself and its interaction with the environment, enabling robots to move across complex terrains [\[5,](#page-20-3) [15,](#page-20-4) [18,](#page-21-2) [23,](#page-21-3) [49\]](#page-22-2), perform mobile manipulation [\[7,](#page-20-1) [11,](#page-20-5) [17,](#page-21-4) [29,](#page-21-5) [35\]](#page-22-3), and mimic human motion [\[16,](#page-20-2) [26\]](#page-21-6). Recent advancements like ASAP [\[16\]](#page-20-2) leverage residual action modeling to align simulated joint torques with real-world actuator responses, while UAN [\[7\]](#page-20-1) introduces unsupervised actuator networks to compensate for unmodeled nonlinear properties exhibiting in the real robots.

These approaches share a common paradigm: (1) physics-based simulation augmentation to cover hardware uncertainties (e.g., friction coefficients, motor saturation), (2) learning-based actuator modeling to replace traditional model-based analytical methods which are overly simplified and inaccurate, and (3) RL learning techniques (asymmetric actor-critic learning and/or curriculum learning) to inform critic with privileged information, or gradually train policies to increasingly realistic task conditions. Such techniques have enabled zero-shot deployment of dynamic skills like locomotion on uneven terrain and highly dynamic acrobatic flips, demonstrating the successful use of RL approach in deploying legged locomotion trained in simulation to reality in a zero-shot manner.

While sim-to-real RL has revolutionized legged locomotion, its application to dexterous in-hand manipulation remains challenging and not fully addressed. Due to limitations in tactile sensors and motor modeling, the sim-to-real technology has yet to be widely applied in the field of manipulation. This is rooted in the discrepancy arising from two unique differences and complexity in manipulation: (1) contact-rich physics involving multi-point, multi-contact interactions and the wide range of material variations in object (e.g., softness, deformations), which are difficult to model accurately in current physics simulations that are primarily catered for rigid body dynamics; and (2) sensorimotor coupling requiring seamless and effective fusion of tactile feedback, joint positions and torques, and high-dimensional visual perception. Nevertheless, prior works in Touch Dexterity [\[44\]](#page-22-4) and Robot Synesthesia [\[45\]](#page-22-5) have shown that tactile-aware policies can achieve robust in-hand reorientation using binary sensors and point-cloud representations.

These results suggest the huge potential of sim-to-real for dexterous manipulation, which is currently underrated. Hence, we are motivated to address the sim-to-real gaps in contact modeling and actuator dynamics in a systematically manner. Drawing inspiration from legged robotics, we developed a holistic sim-to-real recipe combining scalable tactile simulation, current-to-torque calibration, and actuator modelling with uncertainty in actuator dynamics can close the perception-action loop for dexterous manipulation. This work represents an endeavor towards realizing human-level manipulation capabilities through physics simulation driven reinforcement learning.

In the grand vision of building general-purpose robots [\[31\]](#page-21-7), multi-finger dexterous hands with anthropomorphic design have a unique advantage and great potential to offer human-like manipulation capabilities. However, their high degrees of freedom (DoFs) and complex, contact-rich dynamics present substantial control challenges. Consequently, most real-world robotic applications still rely on simple parallel-jaw grippers, and achieving robust in-hand manipulation—the ability to reposition and reorient objects within a fixed grasp—remains a seminal challenge in robotics. The core difficulty lies in the need to sense and control intricate multipoint contact interactions, a domain where humans excel through the seamless integration of touch and proprioception.

Deep reinforcement learning (DRL) trained in simulation has emerged as a promising pathway to overcome the inherent complexities of contact-rich manipulation. The sim-to-real paradigm, leveraging large-scale domain randomization, has produced landmark results, such as transferring in-hand rotation policies to a real five-finger hand [\[34\]](#page-22-6). This foundation was later extended to solve a Rubik's Cube [\[2\]](#page-20-6), establishing a robust methodology for bridging the reality gap through high-dimensional state spaces [\[40\]](#page-22-7). Recent efforts have further expanded the generalizability of this paradigm . Work such as [\[14\]](#page-20-7) demonstrated robust cube reorientation on an Allegro Hand through optimizing simulation fidelity and employing extensive randomization.

A pivotal trend in this progress is the transition from pure vision or proprioception toward multimodal feedback. The integration of tactile sensing has significantly enhanced manipulative precision and dexterity, enabling in-hand rotation using only dense binary touch signals [\[44\]](#page-22-4), fusing visuotactile and point-cloud inputs [\[45\]](#page-22-5), and

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1 A framework for learning a full-state policy integrating tactile sensing and joint torques for dexterous grasping and in-hand manipulation.

facilitating the manipulation of thin slender objects [19]. Complementing these advances, work on learning pre-grasping [38] shows that effective sim-to-real transfer is achievable for specialized single-purpose policies, demonstrating that functional dexterous behaviors can be acquired without reliance on manual heuristic programming.

Parallel progress in dexterous grasping [47] and contact-rich humanoid skills [27] highlights that with appropriate sensing and modeling, sim-to-real RL is a powerful tool for learning policies deployable on real hardware. Furthermore, while a standard MLP is often sufficient for narrow control tasks, complex behaviors can be better facilitated by orchestrating multiple specialized experts. Recent hierarchical approaches [41] have further demonstrated that long-horizon manipulation skills can be realized by orchestrating a repertoire of learned skills. This suggests a compositional pathway for integrating individual skills into complex, long-sequential manipulation tasks.

Despite these advances, a critical frontier remains largely unexplored: force-sensitive, closed-loop manipulation. While prior work has excelled at position-centric tasks (e.g., achieving a target orientation), the explicit regulation of interaction forces which is essential for handling fragile items or performing precision assembly, is often overlooked. Closing this gap necessitates policies that leverage full-state feedback, fusing high-resolution tactile sensing with joint-level torque control to perceive and command forces directly. As visualized in Fig. 1, this "touch & torque" observation space is the key to achieving reinforcement learning (RL) objectives for two fundamental skills: (i) force-controllable grasping for tracking commanded grip forces, and (ii) in-hand reorientation via controlled contact and slip.

However, the development of such full-state policies faces two profound practical hurdles that have prevented their widespread adoption:

- 1. **The Tactile Simulation Bottleneck:** Simulating high-resolution tactile contacts with high physical fidelity is computationally prohibitive at the scale required for deep RL. This forces a trade-off between simulation speed and accuracy, often resulting in brittle policies that have not sufficiently explored the contact dynamics encountered in reality.
- 2. **The Actuation Reality Gap:** Most commercially available dexterous hands use semi-direct-drive actuation and lack joint-level torque sensors, relying instead on motor current for torque estimation. The reality gap between idealized actuator models in simulation (e.g., perfect torque control) and the non-ideal dynamics of real motors (e.g., torque-speed curves, friction, backlash) presents a major barrier to successful sim-to-real transfer for force-based tasks.

In essence, discrepancies in simulating tactile contacts and modeling mechatronic actuator properties have prevented full-state policies from reaching their potential.

Motivated by the growing availability of high-DoF hands, this paper presents a reliable sim-to-real recipe for learning robust full-state policies. Our method, which trains entirely in simulation and deploys zero-shot to a real five-finger hand, is built on the following contributions designed to address the aforementioned gaps:

- **Full-State Policy Formulation:** We design policy observations that jointly incorporate dense tactile signals and estimated joint torques, providing the necessary feedback for explicitly inferring contact states and regulating interaction forces.
- **Computationally Efficient Tactile Simulation:** We introduce a fast, high-resolution tactile simulation method that approximates contacts by computing distances between a dense array of virtual tactile units and the object via parallel forward kinematics. This offers the high-rate signals needed for RL without sacrificing critical contact information.
- **Data-Driven Actuator Modeling and Randomization:** We bridge the actuation gap through a calibrated current-to-torque mapping that eliminates the need for physical torque sensors. We further model and randomize non-ideal motor dynamics (e.g., torque-velocity curves), dramatically reducing sim-to-real torque discrepancies.
- **Zero-Shot Sim-to-Real Deployment:** We demonstrate the efficacy of our integrated approach by successfully deploying policies for two contact-rich skills on real hardware: (1) **controllable grasping** with commanded force tracking and (2) **in-hand object reorientation**.

In summary, this work provides an easy-to-replicate handbook for training full-state RL policies on a representative 12-DoF dexterous hand. By integrating tactile and motor-current sensing with computationally efficient simulation and robust actuator modeling, we achieve zero-shot sim-to-real transfer of advanced, force-controllable manipulation skills. To the best of our knowledge, this represents the first demonstration of such controllable grasping with grip-force tracking and robust in-hand manipulation on a multi-finger hand, trained entirely in simulation and deployed without any fine-tuning.

# **2 Related Work**

We review research work closely related to our focused techniques in our sim2real pipeline for dextrous hands: (i) simulation of tactile sensors, (ii) modeling of motor dynamics, and (iii) current-to-torque mapping and estimation without direct joint torque sensors.

### **2.1 Tactile Simulation**

High-fidelity, fast tactile simulation is a long-standing obstacle for learning contact-rich skills. For visuotactile sensors, recent systems libraries have pushed both realism and throughput. Akinola et al. builds a library within Isaac Gym that synthesizes visuotactile images and contact-force distributions, and couples them with a policy-learning toolkit aimed at sim2real transfer [\[1\]](#page-20-8). In parallel, Nguyen et al. integrate a soft-body FEM simulator with an optical visuotactile rendering pipeline inside Isaac Sim to capture elastomer deformation for GelSight-like skins [\[33\]](#page-21-10). Zhang et al. focus on modeling multi-mode tactile imprints induced by different surface coatings/patterns and report high realism while remaining efficient enough for learning loops [\[46\]](#page-22-11). Beyond single-hand settings, works begin to exploit visuo–tactile simulation for complex, often bimanual, fine assembly: e.g., general sim2real protocols for marker-based visuotactile sensors [\[4\]](#page-20-9), bimanual visuotactile assembly via simulation fine-tuning [\[20\]](#page-21-11).

On the policy side, tactile-only or visuo–tactile in-hand reorientation has seen rapid progress. Yin et al.learns touch-only in-hand rotation on low-cost binary sensors, emphasizing robustness to sensing imperfections [\[44\]](#page-22-4). Yuan et al. propose a point-cloud tactile representation to fuse vision and touch for in-hand rotation [\[45\]](#page-22-5). Purely tactile in-hand manipulation with a torque-controlled hand was shown in [\[37\]](#page-22-12), while works also demonstrate DRL-based tactile control for slender cylindrical objects [\[19\]](#page-21-8). For broader dexterous/bimanual touch, Lin et al. studies bimanual tactile manipulation with sim-to-real deep RL [\[28\]](#page-21-12). Collectively, these efforts

motivate fast, high-fidelity tactile simulation and policy training; our distance-field–based tactile simulation targets this efficiency–fidelity trade-off specifically for RL.

### **2.2 Motor Modeling**

For dexterous hands and manipulation, recent systems explicitly incorporate motor/drive limits or taskinformed system identification. Huang et al. employ privileged learning, system identification, and reinforcement learning to transfer functional grasps across diverse hands [\[21\]](#page-21-13). Tactile in-hand works with torque-controlled hands to build policies on top of explicit joint torque models [\[37\]](#page-22-12). In non-prehensile manipulation, early sim-to-real studies showed the importance of accurate actuation/friction models plus ensemble dynamics to combat identification error [\[30\]](#page-21-14).

In our setting, we explicitly model the motor's torque–speed envelope [\[36\]](#page-22-13), then randomize these parameters (stall torque, speed constants, friction/ripple surrogates) to cover manufacturing tolerances and temperature/load variation. This follows the spirit of recent agile-mobility works where actuator models are aligned via residual learning or unsupervised actuator nets [\[8,](#page-20-10) [16\]](#page-20-2), but adapts them to the brushed-DC, gear-driven fingers typical of dexterous hands.

### **2.3 Current-to-Torque Alignment**

Most dexterous hands available nowadays typically have their SDKs to send over measurements of motor current but lack direct torque sensors at the joint level of the robot. When torque feedback (or torqueconditioned policy inputs) is desired, estimating a reliable current→torque mapping becomes critical. In industrial manipulator literature, motor-current–based wrench estimation has a long history: Kalman filtering and momentum-observer formulations are used to estimate Cartesian forces and torques from joint currents and states [\[13,](#page-20-11) [42\]](#page-22-14). Gold et al. consider torsional deflection plus motor current to estimate joint torques [\[10\]](#page-20-12). From the calibration and control perspective, current-based impedance control explicitly fits actuator current-torque ratios and friction for current-controlled robots–eschewing force-torque sensors, bypassing the need for forcetorque sensors [\[6\]](#page-20-13). System identification methods that identify the models and parameters of the drive gains and dynamics to ensure that the modeled dynamics is physically consistent with the real measurements [\[3\]](#page-20-14).

In this work, we adapt and apply these principles to the small electric motors that are commonly in dexterous hands for the design of direct-drive or semi-direct-drive of the joints. We fit a "current→torque map" under quasi-static conditions to capture effective torque constants and biases, accounting for the a minor loss of torque exertion due to the wear-and-tear of the gear train. Empirically, this mapping provides sufficiently accurate torque estimates for the RL policies and improves sim-to-real alignment,without physically building and embedding expensive miniaturized joint-torque sensors in the robot joints.

# **3 Methodology**

This section presents the technical details of our framework for learning two essential dexterous manipulation policies in simulation for zero-shot deployment on real hardware. We first formalize the problems in the RL framework for force-adaptive and controllable grasping, plus the in-hand object reorientation. Then, we describe our approach of designing the policy's full-state observation space, which includes both the tactile sensing and joint torque sensing via motor-current approximation. Finally, we delineate the integrated recipe for the sim-to-real transfer, and explain technical knowhows on how we bridge the critical gaps in tactile sensing, contact physics, and actuator dynamics.

### **3.1 Problem formulation**

This work addresses the **sim-to-real transfer** of dexterous manipulation policies for multi-fingered robotic hands, focusing on zero-shot deployment of simulation-trained policies onto real hardware. Despite progress in RL-based manipulation, transferring contact-rich, force-sensitive tasks, such as variable-force grasping and in-hand reorientation, remains challenging due to gaps in tactile simulation, actuator modeling, and contact physics.

Formally, given a dexterous hand with N fingers equipped with tactile sensors and current-controlled motors (no torque sensors), we learn a policy  $\pi: o_t \mapsto a_t$  that maps observations in simulation that perform successfully on the physical system without fine-tuning.

#### 3.2 MDP formulation

We formulate the control of a dexterous hand as a Markov Decision Process (MDP), defined by the tuple  $(S, A, T, R, \gamma, \rho_0)$ , where S is the state space, A is the action space,  $T(s_{t+1}|s_t, a_t)$  is the state transition probability,  $R(s_t, a_t)$  is the reward function,  $\gamma$  is the discount factor, and  $\rho_0$  is the initial state distribution. Our objective is to learn a policy  $\pi_{\theta}(a_t|o_t)$ , parameterized by  $\theta$ , that maximizes the expected cumulative reward  $\mathbb{E}\pi[\sum_{t=0}^T = \gamma^t R(s_t, a_t)]$ .

The core challenge we address is the **sim-to-real transfer** of this policy. While the policy is trained entirely within a simulated MDP  $M_{\rm sim}$ , where  $M_{\rm sim}$  represents the *simulated physics world* – the entire simulation environment including robot model, objects, and sensor / contact models. The training data is generated by the policy  $\pi$  interacting with  $M_{\rm sim}$ , and the challenge is that such a policy trained in  $M_{\rm sim}$  must execute successfully or at least perform reasonably well on the *real physical world*, governed by a real-world MDP  $M_{\rm real}$ , under a zero-shot deployment — meaning no fine-tuning on the real hardware.

The discrepancy between  $M_{\text{sim}}$  and  $M_{\text{real}}$ , known as the "reality gap", is particularly pronounced for dexterous manipulation due to:

- Tactile Simulation Gap: Inaccurate or computationally prohibitive modeling of high-resolution contact sensing;
- Actuator Dynamics Gap: Simplified models of motor dynamics that omit effects like torque-speed saturation, backlash, and current-to-torque nonlinearities;
- Contact Physics Gap: Differences in friction, material deformation, and multi-body contact dynamics between simulation and reality.

Formally, our agent is a dexterous hand with N fingers (N = 5 in this paper with 12-DoF active actuation). Its actions  $a_t \in A \subset \mathbb{R}^{12}$  are target joint positions for the 12-DoF hand, sent to a motor-level PD controller. The observations  $o_t \in O$  constitute a full-state representation, including proprioception (joint angles, velocities), tactile readings, and estimated joint torques (detailed in Section 3.5).

The policy  $\pi_{\theta}(a_t|o_t)$  is trained in simulation to solve two distinct, contact-rich tasks:

- Force-Adaptive Grasping: The policy needs to achieve a stable grasp and modulate the each finger's grip force to track a user-specified force command  $F_{\text{cmd}}$ .
- In-Hand Object Reorientation: The policy needs to rotate an in-hand object about a fixed axis through coordinated finger movements, keeping stable contact throughout the whole multi-contact interactions.

The success of our approach relies not only on the policy's architecture, but also on the careful design of  $M_{\text{sim}}$  to minimize the reality gap. The subsequent sections detail our methods for bridging the gaps in sensing (Section 3.6.2) and actuation (Section 3.6.4).

#### 3.2.1 Force-Adaptive Grasping

The objective of this task is to achieve and maintain stable grasps on a diverse set of objects with geometrically complex and physically varied properties including size, shape, mass distribution, and surface friction, while accurately exerting user-specified force levels. The policy must modulate grip forces in real time according to object characteristics and external disturbances to prevent slip or excessive deformation, thereby demonstrating robust and adaptive grasping under dynamic real-world conditions.

#### 3.2.2 In-Hand Object Rotation

This task requires the precise and controlled rotation of a grasped object about a predefined spatial axis using continuous and compliant finger gaits. Successful rotation entails rich, multi-point frictional contacts and

requires coordinated finger motions that induce controlled slip and rolling interactions between fingertips and the object surface. The policy must maintain stability during manipulation by adjusting contact forces and reposing fingers as needed to avoid dropping or losing control of the object, thereby achieving smooth and continuous rotational reorientation.

Experiments use a 12-DoF direct-drive dexterous hand (xHand) [\[9\]](#page-20-15) and are conducted in the IsaacLab simulation environment [\[32\]](#page-21-15).

# **3.3 Identification of Sim-to-Real Gaps**

Our study identified 4 primary sim-to-real gaps:

- (1) **Perceptual Gap**: Simulations often provide perfect ground-truth state information, such as object pose and joint angles. However, the real systems rely on these states from noisy sources, including errors from camera calibration, varying lighting conditions, occlusions, and the limitations of vision-based pose estimation, resulting in an unavoidable mismatch between the policy's ideal expected observations and the actual ones.
- (2) **Discrepancies in Actuator Dynamics**: Real actuators, such as the commonly used electric motors, have complex and non-ideal behaviors that are often simplified or omitted in the physics simulation. It ranges from significant mechanical discrepancies in the physical gear chain system, such as mechanical backlash in gear transmissions, to nonlinearities such as static (stiction) and dynamic friction, torque-speed curves (e.g., motor current saturation at higher velocities), motor response delays, and imprecise current-to-torque mapping. These unmodeled acutator dynamics cause the commanded actions in reality exert different forces and thus exhibits different motion patterns on the real robot.
- (3) **Discrepancies in Contact Physics**: The contact dynamics in the simulation is primarily based on rigid bodies, which often have limited accuracy in surface geometry due to the use of simplified meshes, and also miss out the material properties (e.g., stiffness, friction, restitution) which are more complex to numerically compute. Therefore, for ensuring reasonable computational cost, to some extent, most available simulation engines nowadays trade off the complexity and fidelity of real physical interactions, including multi-contact modeling, deformation of soft and compliant materials, varying friction coefficients, rolling resistance, and the stiffness-damping properties of the building materials of the robot. All these result in policies that fail to effectively explore and exploit rich tactile feedback in simulation, which is essential for dexterous in-hand manipulation.
- (4) **Unseen State Distribution Shift**: Given the above main sources of discrepancies, even with robust training with randomization of key physical properties, a policy may still encounter novel object properties such as materials, shapes, masses, frictions–scenarios and configurations that are not exactly the same or not seen during training. This Out-of-Distribution (OOD) problem leads to compounding errors as the policy operates in those states where its learned value functions and policies are no longer accurate, causing performance deterioration or even failures while deployed in the real system.

# **3.4 Training setup for Reinforcement Learning**

#### **3.4.1 Force-adaptive grasping**

A critical factor in training effective RL policies is the quality and efficiency of data collection within the simulation environment. For dexterous manipulation, the chosen training scenario profoundly impacts both the speed of policy convergence and the ultimate quality and robustness of the grasp.

Several noticeable limitations of traditional tabletop grasping are summerized as follows:

**Conventional RL training for dexterous hands often mimics human tabletop grasping**: an object is placed on a surface, and the hand, initialized from a random pose above it, must learn to approach, orient, and grasp the object. While intuitive, this paradigm introduces significant inefficiencies and challenges:

**High Sample Inefficiency and Complex Reward Engineering**: A randomly initialized policy must discover this complex sequence of actions (approaching, palm reorientation, and finger closure) through extensive trial-and-error. This exploration process is exceptionally time and computationally expensive. It often

necessitates intricate curriculum learning and meticulously engineered reward functions to incrementally guide the policy toward successful behavior, adding considerable complexity to the training pipeline.

**Reward Hacking and Non-Robust Emergent Behaviors**: To maximize the reward, policies frequently learn to exploit the simulation-specific physics that are inconsistent with the real-world physics, resulting in policies that fail to translate to reality with reasonable success rate – a phenomenon known as reward hacking (typically, the unrealistic frictions). This can result in unnatural, unstable, and aesthetically non-human-like grasping strategies (e.g., precarious two-finger pinches or exploiting specific contact points). These strategies are often brittle and fail catastrophically upon real-world deployment due to the inevitable sim-to-real gaps in dynamics and contact physics, rendering the trained policy ineffective.

To overcome these limitations, we introduce a novel training paradigm: an inverted "catch-the-object" setup. As illustrated in Figure [\[6\]](#page-16-0), the dexterous hand is fixed in a palm-up orientation. Objects are procedurally dropped from above into its workspace, and the policy's objective is to robustly catch and stabilize them.

This method is implemented with several key randomization strategies to enhance robustness and generalization.

**Object Properties**: Mass, size, shape, and friction coefficients are randomized.

**Initial Conditions**: Introduced randomization to create variations of the initial orientation of different objects, before releasing them into the palm of the dexterous hand.

**Inverted Catching Setup-Learning Grasping via Catching**: The Inverted Catching setup turns the hand upside-down and drops different objects into the hand with an open-palm configuration, using gravity to naturally pull the object towards the hand with varied velocities and poses, making the RL policy easier to learn how to grab and hold the targeted object firmly. In such a setting, the robot avoids the pre-grasp interactions which can lead to hacking the physics simulation.

The proposed approach offers several distinct advantages that directly address the shortcomings of traditional methods:

**Improved Sample Efficiency**: By leveraging gravity to naturally bring the object into the hand's workspace, the exploration problem is vastly simplified. The policy can focus its learning effort on the core challenge of finger coordination and force modulation upon contact, rather than on the initial approach phase. This leads to a significant reduction in training time and computational resources.

**Mitigation of Reward Hacking and Emergence of Human-Like Grasps**: The catching dynamic encourages the formation of enveloping, multi-point contact grasps that naturally center the object in the palm. This inherently suppresses the emergence of unnatural, reward-hacking strategies and promotes stable, humanpreferred grasping configurations. The resulting policies are not only more robust but also more aesthetically plausible.

**Enhanced Sim-to-Real Transferability**: The observations required for this task are proprioception (joint angles, velocities), inferred torque, and tactile contact signals, which can be primarily measured on real hardware. By avoiding reliance on hard-to-simulate or hard-to-measure visual features (like precise object pose from a specific camera angle) and fostering robust contact-rich interaction, this paradigm minimizes the perceptual and dynamics sim-to-real gaps. Consequently, simulation-trained policies can be deployed very successfully on physical robots in a zero-shot manner.

In summary, this inverted catching setup serves as a highly efficient and effective "gymnasium" for training policies for grasping various objects, which accelerates learning and encourages robust grasping strategies, and fundamentally reduces the reality gap, serving as a practical handbook for achieving reliable sim-to-real dexterous manipulation.

#### **3.4.2 In-Hand Object Rotation**

The in-hand object rotation task is designed to test the dexterous hand's ability to perform fine, contactrich manipulation. A standardized cube is chosen as the manipulation object to provide a structured and reproducible benchmark. This allows us to focus on the core challenge of learning coordinated finger gaits rather than adapting to infinite object geometries.

<span id="page-8-1"></span>

| Input                  | Dim | Actor        | Critic       |
|------------------------|-----|--------------|--------------|
| Hand joints angles     | 12D | ✓            | <b>√</b>     |
| Hand joints torque     | 12D | $\checkmark$ | $\checkmark$ |
| Object position        | 3D  | $\checkmark$ | $\checkmark$ |
| Object linear velocity | 3D  | $\checkmark$ | $\checkmark$ |
| Contact force          | 5D  | $\checkmark$ | $\checkmark$ |
| Contact center         | 15D | $\checkmark$ | $\checkmark$ |
| Fingertip positions    | 15D | $\checkmark$ | $\checkmark$ |
| Force command          | 1D  | $\checkmark$ | $\checkmark$ |

|    | \  | $\Omega$ 1  | C   | C 1 1.         |          |
|----|----|-------------|-----|----------------|----------|
| 10 | 3) | Observation | tor | force-adaptive | grasping |
|    |    |             |     |                |          |

| Input                       | Dim | Actor        | Critic       |
|-----------------------------|-----|--------------|--------------|
| Hand joints angles          | 12D | ✓            | ✓            |
| Relative target orientation | 6D  | $\checkmark$ | $\checkmark$ |
| Last actions                | 12D | $\checkmark$ | $\checkmark$ |
| Contact center              | 15D | $\checkmark$ | $\checkmark$ |
| Contact force               | 5D  | $\checkmark$ | $\checkmark$ |
| Fingertip positions         | 15D | $\checkmark$ | $\checkmark$ |
| Object orientation          | 6D  | ×            | $\checkmark$ |
| Fingertip velocities        | 30D | ×            | $\checkmark$ |
| Fingertip rotations         | 30D | ×            | $\checkmark$ |
| Hand joints velocities      | 12D | ×            | $\checkmark$ |
| Object linear velocity      | 3D  | ×            | $\checkmark$ |
| Object angular velocity     | 3D  | ×            | $\checkmark$ |
|                             |     |              |              |

**(b)** Observation for in-hand object rotation

**Table 1** Observation for our tasks

Constrained Rotation for Focused Learning: Given the limited degrees of freedom (DoF) of the adopted dexterous hand compared to the human hand, we constrain the agent to rotate the cube around a single, predefined axis. This simplification reduces the problem's dimensionality, allowing the policy to master the fundamental dynamics of controlled slipping and rolling contacts without being overwhelmed by the complexity of full 3D rotation. Mastery of single-axis rotation is a critical prerequisite for more general manipulation.

To encourage the policy to learn a continuous, sustainable manipulation skill rather than just a single, static reorientation, we implement a dynamic training curriculum. The task is structured as a series of sequential goals: The target pose for the policy is not a fixed absolute orientation. Instead, it is defined as a 90-degree rotation from the object's current orientation around the designated axis.

Each time the policy successfully achieves a 90° rotation (within a specified tolerance), this event is recorded as a success. The environment then automatically updates the target to a new one, another 90° further along the same axis.

This "shifting goal" mechanism forces the policy to learn a continuous regrasping behavior. It must not only achieve a specific orientation but also recover stability and prepare for the next manipulation step, mimicking the continuous nature of in-hand manipulation performed by humans. This is far more challenging and informative than learning a one-shot rotation from a fixed start to a fixed end pose.

#### <span id="page-8-0"></span>3.5 Policy learning in simulation

#### 3.5.1 Force-adaptive grasping

The complete set of observations used during policy training is detailed in Table 1a. In the real-world deployment setup, the object's three-dimensional position is estimated through a vision-based pipeline: the Z-coordinate is computed from the distance between the palm center and the centroid of the segmented object mask, where the mask is inferred using the Segment Anything Model (SAM) [24]. The X and Y coordinates are held fixed relative to the hand's initial pose, simplifying the state estimation under the assumption of limited lateral motion during grasping. The object's linear velocity is then numerically derived from the sequence of estimated Z-coordinates using a finite difference method. Additionally, a force command scalar in the range [0, 1] is provided as part of the observation, indicating the desired grasping intensity, enabling the policy to modulate grip force according to task requirements. During the simulation training process, we designed a reward mechanism to guide the policy toward achieving stable and reliable grasping during inverted catching tasks.

Once the fingers are in contact with the object, we employ torque command rewards  $R_{\text{torque}}$  to encourage the

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 2 Calibration and alignment of current-force (real robot) versus torque-force (simulation) properties.

agent to apply specific joint torques. The target torque  $\tau_{\text{target}=\tau_{\text{max}}\cdot F_{\text{cmd}}}$  is a continuous value provided by the instruction of the environment, and the reward function has a special handling for the thumb compared to the other fingers.

The total reward for this component is the sum of the individual finger torque rewards, which is applicable only when the finger is in contact with the object.

$$R_{\text{torque}} = w_{\text{torque}} \cdot \sum_{i \in \text{fingers}} R_{\text{torque},i} \cdot I_{\text{contact},i},$$
 (1)

where  $w_{\text{torque}}$  is the weight of this reward,  $I_{\text{contact},i}$  is the indicator function, which is 1 if the finger i is in contact with the object and 0 otherwise.

The reward for the thumb is binary, contingent on its torque being within a valid operational range defined by the hardware limits  $[\tau_{min}, \tau_{max}]$ :

$$R_{\text{torque, thumb}} = I(\tau_{min} \le \tau_{thumb} \le \tau_{max}),$$
 (2)

Other four fingers' reward is a Gaussian function centered around the instructed target, multiplied by a validity mask.

$$R_{\text{torque},i} = \exp\left(-\frac{(\tau_i - \tau_{\text{target}})^2}{2\sigma^2}\right) \cdot I(\tau_{min} \le \tau_{thumb} \le \tau_{max}),\tag{3}$$

where  $\sigma$  is a tunable parameter determined by the desired torque tracking precision.

For the contact force, similar to the joint torque reward, an appropriate contact force  $F_{\text{target}} = F_{\text{cmd}} \cdot F_{\text{max}}$  is encouraged.

$$R_{\text{force}} = w_{\text{force}} \cdot \sum_{i \in \text{fingers}} R_{\text{force},i} \cdot I_{\text{contact},i}, \tag{4}$$

$$R_{\text{force, thumb}} = I(F_{min} \le F_{\text{thumb}} \le F_{max}),$$
 (5)

$$R_{\text{force},i} = \exp\left(-\frac{(F_i - F_{\text{target}})^2}{2\sigma^2}\right) \cdot I(F_{min} \le F_i \le F_{max}),\tag{6}$$

where  $F_i$  is the fingertip contact force of the root joint (finger i),  $\sigma$  is the standard deviation of the Gaussian function, and  $I(\cdot)$  is the indicator function.

In addition to standard grasping rewards, we introduce a novel four-finger consistency reward to promote coordinated motion among the four homologous and structurally similar fingers. This term regulates the

<span id="page-10-0"></span>

| Reward           | Formula                                                                                                                                       | Weight              |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| Close to goal    | $\frac{\frac{1.0}{ d_{\text{rot}}  + \epsilon} \times \frac{1}{1 + e^{\alpha(d_{\text{goal}} - \delta_{\text{dist}})}}}{  a_t - a_{t-1}  ^2}$ | $w_{\mathrm{goal}}$ |
| Action penalty   | $  a_t - a_{t-1}  ^2$                                                                                                                         | $w_{\rm action}$    |
| Reset reward     | Condition                                                                                                                                     | Value               |
| Reach goal bonus | $d_{\rm rot} < \delta_{\rm rot}^{\rm threshold}$ and $d_{\rm pos} < \delta_{\rm pos}^{\rm threshold}$                                         | $C_{\rm Bonus}$     |

**Table 2** Reward for in-hand object rotation

uniformity of their flexion and extension angles. In the human hand, biomechanical constraints limit the range of independent motion of the metacarpophalangeal joints, causing adjacent fingers to move correlatively. In contrast, robotic fingers are driven by independent motors. Without explicit coordination, reinforcement learning policies often produce unnatural and inefficient postures. The consistency reward encourages more human-like motion patterns, leading to more stable and physically plausible grasps with improved force distribution.

The penalty  $R_{\text{diff}}$  to quantify the difference in-between four fingers is defined as:

$$R_{\text{diff}} = w_{\text{diff}} \cdot \text{Var}(\{q_j \mid j \in \text{inner fingers}\}),$$
 (7)

where  $w_{\text{diff}}$  is the penalty weight,  $q_j$  is position of the first joint of finger j,  $\text{Var}(\cdot)$  is variance of the set of joint positions.

The outer finger movement penalty  $R_{\text{outter}}$  is defined as:

$$R_{\text{outter}} = w_{\text{outter}} \cdot ||q_{\text{outter}} - c_{\text{outter}}||_2,$$
 (8)

where  $w_{\text{outter}}$  is the weight of this penalty,  $q_{\text{outter}}$  is current joint position vector for the outer joints (index, middle, ring, and little fingers),  $c_{\text{outter}}$  is center position vector of the outer joints.

In addition, a set of standard penalty terms is integrated to further shape the agent's behavior. To discourage early termination of episodes, a terminal state penalty is introduced. Moreover, an action rate penalty is applied to limit large fluctuations in action commands between consecutive time steps, thereby promoting smoother and more stable control policies. This penalty is formally defined as:

$$R_{\text{action}} = w_{\text{action}} \cdot ||a_t - a_{t-1}||_2^2, \tag{9}$$

where  $R_{\text{action}}$  is the action rate penalty,  $w_{\text{action}}$  is the weight of this penalty,  $a_t$  is the action vector at the current time step t, and  $a_{t-1}$  is the action vector at the previous time step t-1. And a joint velocity L2 penalty to discourage high joint velocities. This encourages the agent to generate smoother and more stable movements, avoiding abrupt motions. The penalty is defined as:

$$R_{\text{vel}} = w_{\text{vel}} \cdot ||\dot{q}||_2^2,\tag{10}$$

where  $\dot{q}$  represents the joint velocity vector, and  $w_{\rm vel}$  is the corresponding penalty weight.

### 3.5.2 In-Hand Object Rotation

Effective sim-to-real transfer requires policy observations that are not only informative but also practically obtainable on real hardware. We therefore design our policies around two feasible perception paradigms, as detailed in Tab. 1b:

**Proprioception-Only Policy:** This policy relies solely on the robot's internal state, joint angles, velocities, tactile signals  $(F_i, \mu_i)$ , and actions, without any explicit object pose information. It must infer the object's state and rotation progress implicitly from the history of contact interactions, making it robust to the absence of external sensors but potentially more challenging to train.

<span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

![](_page_11_Figure_1.jpeg)

(a) Modeling of contact points to approximate tactile sensors that mitigate the sim-to-real gap.

**(b)** Stress-Strain curve of the rubber materials of the tactile sensor on the dexterous hands.

Figure 3 Contact point modeling and material properties.

**IMU-Augmented Policy:** This policy incorporates direct orientation feedback from a low-cost inertial measurement unit (IMU) embedded within the object. This provides a clear, real-world measurable signal of the object's state, simplifying the policy's task. The target orientation is specified relative to the robot's palm frame.

A critical implementation detail for the IMU-augmented policy is the choice of orientation representation. We avoid quaternions due to their double-cover property (where a single rotation is represented by both  $\mathbf{q}$  and  $-\mathbf{q}$ ), which introduces a discontinuity that can severely disrupt gradient-based learning. Instead, we adopt the continuous 6D rotation representation [48], which provides a smoother learning landscape and leads to more stable and efficient policy convergence.

Reward Engineering for Continuous Manipulation: Simply rewarding the agent for minimizing the rotation error to a target can lead to a suboptimal local minimum: the policy learns to hold the object perfectly still to avoid any position penalty, thus never attempting the rotation. To overcome this, we designed a composite reward function (Tab. 2) that jointly minimizes both rotation error and position drift. Furthermore, a large bonus is provided upon successfully reaching a target orientation. This combination incentivizes the policy to perform complete rotations and then re-stabilize the object, emulating the continuous finger-gaiting motions seen in human dexterous manipulation. This reward structure was essential for promoting dynamic and continuous in-hand rotation rather than static grasping.

#### 3.6 Sim-to-Real Transfer

We identified three primary contributors to the sim-to-real gap in our task: tactile sensing, complex in-hand contact simulation, and actuator dynamics. Our approach closes this gap through precise problem formulation and tailored simulation techniques.

#### 3.6.1 Tactile Sensor Simulation

We implement a rapid, massively tactile sensor simulation using parallel forward kinematics, which calculates the distance from each sensor unit to the N-nearest surface points on the object.

We consider a dexterous hand with N fingers. Typically, N = 5. The sensor set of the entire dexterous hand can be denoted as S. On the fingertip of each finger i (where  $i \in \{1, 2, ..., N\}$ ), there is a set of sensors, which we represent as  $S_i$ . Therefore, the total sensor set is the union of all fingertip sensor sets:

$$S = \bigcup_{i=1}^{N} S_i.$$

For each individual sensor j on each fingertip i (where  $j \in \{1, 2, ..., M_i\}$ , and  $M_i$  is the total number of sensors on the i-th fingertip), we define its key attributes:

**Position**: Through forward kinematics, we can calculate the spatial coordinates of each sensor from the hand degrees-of-freedom position, denoted as  $p_{ij} \in \mathbb{R}^3$ .

**Activation**: On the real robot hand, the sensor is activated while reading is greater than a threshold. In the simulation, as shown in Fig. 3a, the nearest point  $p_o$  on the surface of the object is found for each sensor  $p_{ij}$ . A vector from the object pointing to each tacxiel,  $v_{oij}$ , is defined. The contact is detected if the sensor has deformed, which is determined by evaluating the dot product of  $v_{oij}$  and the normal vector of the object at  $p_o$ . The vector from  $p_o$  to  $p_{ij}$  is given by: $v_{oij} = p_{ij} - p_o$ . Let  $n_o$  be the outward-pointing normal vector at the object point  $p_o$ . If the dot product of  $v_{oij}$  and  $n_o$  is negative, it implies that the sensor has penetrated the object's surface, indicating a deformation. All points meeting such a condition are considered as equivalent contact points.

Contact 
$$\iff v_{oij} \cdot n_o < 0.$$
 (11)

**Force**: On the real robot, normal forces are measured by the sensors. In the simulation, it is approximated using Mooney-Rivlin stress: strain relationship model to quantify the contact force of each sensor based on the distance  $d_{ij} = |v_o ij|$  between the sensor and the object surface and the total contact force per finger, denoted as  $f_{ij} \in \mathbb{R}^+$ .

In our model, we use the fully actuated robotic hand xHand with 12 degrees of freedom. Each fingertip has  $M_i = 120$  sensors, so the entire hand has  $\sum_{i=1}^{N} M_i = 600$  sensors. These sensors are concentrated in five independent "dense groups". This high local resolution on each fingertip can capture fine pressure distributions.

#### <span id="page-12-0"></span>3.6.2 Complex Contact Simulation

Simulating complex multi-point contacts in dexterous manipulation is a primary challenge. To bridge the sim-to-real gap for contact physics, we employ domain randomization on key object properties during policy training, including surface friction, damping, and restitution coefficients. This forces the policy to learn robust strategies that are invariant to these physical uncertainties, rather than overfitting to precise but inaccurate simulated dynamics.

To efficiently process the high-dimensional output of our simulated tactile sensors, we abstract the raw data into a compact, semantically meaningful representation  $T_i$  for each fingertip i. This representation is designed to mirror the information a policy could feasibly extract from real tactile sensors, balancing detail with computational tractability. We define  $T_i$  as a tuple capturing the two most critical aspects of contact: its magnitude and its location.

$$T_i = (F_i, \mu_i) \tag{12}$$

**Total Contact Force** ( $F_i$ ): This scalar value represents the magnitude of the interaction. On the real robot, this is the sum of normal forces from all individual taxels on the fingertip. In simulation, it is computed equivalently as the sum of the forces  $f_{ij}$  from all active virtual sensors j on the fingertip:

$$F_i = \sum_{j \in S_i^*} f_{ij}. \tag{13}$$

We model the contact force  $f_{ij}$  at each tactile unit using a general hyper-elastic material model to capture the non-linear relationship between the penetration distance  $d_{ij}$  and reaction force. Here, the Mooney-Rivlin model is employed as the standard and generic formulation for such soft contacts. Specifically, because the rubber material used in the xHand fingertip is relatively rigid, under the common contact forces, the force-penetration curve for this material operates primarily in a low-nonlinearity and near-linear range relevant to typical manipulation forces, as shown in Fig. 3b. Therefore, for computational efficiency during the training of reinforcement learning, we use a linear spring approximation of the contact force in the simulator:  $f_{ij} \approx k \cdot d_{ij}$ .

Force-Weared Mean Contact Position ( $\mu_i$ ): This 3D vector represents the center of pressure, a vital cue for estimating contact geometry and object pose. It is calculated as the force-weighted average of the positions

<span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

Figure 4 Visualization of real-world and simulated contact data during an in-hand rotation task. The close alignment between the contact points (Top) and contact forces (Bottom) shows the high fidelity of our contact simulation.

pij of all active sensors:

$$\mu_i = \frac{\sum j \in S_i^* fij \cdot pij}{F_i}.$$
 (14)

Substituting  $f_{ij} \approx k \cdot d_{ij}$  simplifies this calculation in simulation, making it highly efficient without loss of generality:

$$\mu_{i,sim} \approx \frac{\sum_{j \in S_i^*} d_{ij} \cdot k \cdot p_{ij}}{D_i \cdot k} = \frac{\sum_{j \in S_i^*} d_{ij} \cdot p_{ij}}{D_i}.$$

This abstraction serves a dual purpose: it drastically reduces the observation space dimensionality for the RL policy, and it provides a transferable feature representation. As validated in Fig. 4, the distributions of  $F_i$  and  $\mu_i$  are highly consistent between simulation and the real world. Crucially, the simulated contact region envelops the real-world data, confirming that the policy learns from a sufficient and covering distribution of contact scenarios, enabling effective sim-to-real transfer of tactile-based manipulation skills.

#### 3.6.3 Current-Torque Calibration

A critical sim-to-real gap arises from the fact that our real-world dexterous hand provides motor current measurements but lacks direct joint torque sensors, whereas our simulation has direct access to idealized joint torques. To enable our torque-conditioned policy to function on real hardware, we must establish a reliable mapping from measured motor current to the simulated joint torque that the policy expects.

We bridge this perception gap through a one-time, per-joint calibration procedure. For each finger joint, we applied a spectrum of external forces to the fingertip in both the real and simulated environments. In the real world, we recorded the motor current  $I_{\rm real}$  and the resultant contact force  $F_{\rm real}$  measured from the high-resolution tactile sensors on the fingertip. In parallel, within simulation, we recorded the idealized joint torque  $\tau_{\rm sim}$  and the simulated contact force  $F_{\rm sim}$  for the same applied force conditions.

Our analysis, summarized in Fig. 2, confirmed a strong linear correlation between contact force and both motor current (in the real world) and joint torque (in simulation). This linearity allows us to model the relationships as:

$$F_{\rm real} \approx \alpha \cdot I_{\rm real} \quad \text{and} \quad F_{\rm sim} \approx \beta \cdot \tau_{\rm sim}$$
 (15)

where  $\alpha$  and  $\beta$  are the estimated proportionality constants for the real and simulated systems, respectively.

We then identified the maximum observed values: the maximum real-world current  $I_{\text{max}}$ , the maximum simulated torque  $\tau_{\text{max}}$ , and their corresponding maximum contact forces  $F_{\text{max}, \text{ real}}$  and  $F_{\text{max}, \text{ sim}}$ . The final calibration step involves normalizing both the real-world current and simulated torque signals into a unified, dimensionless force proxy scale of [0, 1]:

$$I_{\text{norm}} = \frac{I_{\text{real}}}{I_{\text{max}}} \approx \frac{F_{\text{real}}}{F_{\text{max, real}}}$$
 (16)

$$\tau_{\text{norm}} = \frac{\tau_{\text{sim}}}{\tau_{\text{max}}} \approx \frac{F_{\text{sim}}}{F_{\text{max, sim}}}$$
(17)

This normalization effectively aligns the real-world current reading with the simulated joint torque value through their shared relationship to contact force. During real-world deployment, the policy now receives  $I_{\text{norm}}$  in the "joint torque" field of its observation space. This signal is semantically consistent with the  $\tau_{\text{norm}}$  it was trained on in simulation, enabling zero-shot transfer without requiring physical joint torque sensors. This process ensures that a policy command for a specific "simulated torque" results in a functionally equivalent "real-world force" exertion.

#### <span id="page-14-0"></span>3.6.4 Actuator Model Randomization

To bridge the sim-to-real gap in dexterous manipulation, we use a randomized actuator model that captures real-world motor nonlinearities, such as torque-velocity saturation and backlash hysteresis, improving policy robustness against hardware variations. The torque is computed via a PD controller with randomized gains:

$$\tau_c = k_p \cdot (q_{\text{ref}} - q_m) + k_d \cdot (\dot{q}_{\text{ref}} - \dot{q}_m), \qquad (18)$$

where  $k_p$  and  $k_d$  are the proportional and derivative gains, respectively, and  $q_m$  denotes the measured joint position.

A key feature of the model is the incorporation of backlash, simulating mechanical dead zones due to gear play. The effective torque is modulated by a deadband function:

$$\tau_b = \begin{cases} 0 & \text{if } |q_{\text{ref}} - q_m| < \epsilon \\ \tau_c & \text{otherwise} \end{cases}, \tag{19}$$

where  $\epsilon$  is the backlash threshold, randomized during training to mimic physical wear and tolerance variations.

Further, the torque is constrained by a velocity-dependent saturation function due to the DC motor's characteristics:

$$\tau_{\text{sat}}^{+}(\dot{q}) = \tau_0 \left( 1 - \frac{|\dot{q}|}{\dot{q}_{\text{max}}} \right), \quad \tau_{\text{sat}}^{-}(\dot{q}) = -\tau_0 \left( 1 - \frac{|\dot{q}|}{\dot{q}_{\text{max}}} \right),$$
(20)

where  $\tau_0$  is the stall torque and  $\dot{q}_{\text{max}}$  is the maximum no-load velocity. The final applied torque is given by:

$$\tau_{\text{applied}} = \eta \cdot \text{clip}\left(\tau_b, \ \tau_{\text{sat}}^-(\dot{q}), \ \tau_{\text{sat}}^+(\dot{q})\right),$$
(21)

where  $\eta$  is a randomized factor that accounts for variations in motor torque constant and drive efficiency.

All model parameters are resampled for each actuator at the beginning of every training episode. This comprehensive randomization strategy forces the control policy to adapt to a wide spectrum of actuator imperfections, significantly improving sim-to-real transfer performance.

#### 4 Experimental Results

#### 4.1 Force-adaptive grasping

To determine the most effective reward structure for this purpose, we conducted an ablation study on the two primary reward terms responsible for regulating interaction forces:  $R_{\text{Force}}$  (penalizing inaccurate fingertip contact forces) and  $R_{\text{torque}}$  (penalizing inaccurate joint torques).

Our analysis, summarized in Tab. 3, reveals that each reward term shapes the policy's behavior distinctly, and their combination is strictly necessary for achieving robust and adaptive force-modulated grasping:

| $\overline{R_{\mathrm{Force}}}$ | $R_{\rm torque}$ | Contact force range | Joint torque percent |
|---------------------------------|------------------|---------------------|----------------------|
| <b>√</b>                        | ✓                | $44.50 \sim 93.92$  | $1.09 \sim 1.52$     |
| $\checkmark$                    | ×                | $37.29 \sim 70.02$  | $0.69 \sim 0.75$     |
| ×                               | $\checkmark$     | $38.45 \sim 47.12$  | $0.85 \sim 1.02$     |

**Table 3** Ablation on the rewards  $R_{\text{Force}}$  and  $R_{\text{torque}}$ .

<span id="page-15-1"></span><span id="page-15-0"></span>![](_page_15_Figure_2.jpeg)

Figure 5 Joint torque and contact forces under controllable force commands with different reward settings.

 $R_{\text{Force}}$  Only: Policies produce moderate contact forces and low joint torques. The policy learns to achieve contact but avoids applying high torques, potentially leading to weak or slip-prone grasps.

 $R_{\text{torque}}$  Only: Policies achieve moderate joint torques but low contact forces. This suggests the policy learns to "tense" the joints without effectively transmitting force through the kinematic chain to the fingertips, resulting in inefficient grasping.

 $R_{\text{Force}} + R_{\text{torque}}$ : The combination results in significantly stronger and wider ranges for both contact forces and joint torques. This synergistic effect indicates the policy learns to coordinate joint actuation and fingertip contact simultaneously, leading to more forceful, stable, and dynamic grasping.

Furthermore, we evaluated the policy's ability to track a variable force command. As shown in Fig. 5, both the exerted contact force and joint torque exhibit an approximately linear relationship with the commanded input, confirming the policy's capacity for fine-grained force control. The strength of this linear correlation directly reflects the efficacy of the reward structure:

Correlation Strength: 
$$(R_{\text{Force}} + R_{\text{torque}}) > R_{\text{Force}} > R_{\text{torque}}$$
 (22)

This graded response demonstrates that  $R_{\rm Force}$  is the primary driver for precise force tracking at the end-\neffector, while  $R_{\rm torque}$  acts as a crucial regularizer that ensures the forces are generated through mechanically\nefficient and plausible joint-level actuation. Together, they enable the precise, whole-hand force control that\nis a hallmark of human dexterity.

# 4.2 In-Hand Object Rotation

To quantitatively evaluate the contribution of each sensory component, we conducted a series of real-world ablation studies. Each policy was evaluated over ten trials, with performance measured using three metrics: the number of consecutive successful rotations, average duration per success, and time until failure (see Tab. 4).

The complete observation configuration, comprising force-weighted contact center, contact force, and 6D orientation, has achieved the best performance with an average of 25.1 consecutive successes and a mean

| Table 4 | Ablation | analysis | of multiple | observation | combinations | in the i | eal world. |
|---------|----------|----------|-------------|-------------|--------------|----------|------------|
|---------|----------|----------|-------------|-------------|--------------|----------|------------|

<span id="page-16-1"></span>

| Contact  | Contact | Force    | Orientation    | Cons. success                | Average | Median | Succ.      | Time to     |
|----------|---------|----------|----------------|------------------------------|---------|--------|------------|-------------|
| center   | force   | weighted | representation | trials(sorted)               | Average | Median | time(Ave.) | fall (Ave.) |
| ✓        | ✓       | ✓        | 6D             | 8,10,12,15,19,25,26,35,46,55 | 25.1    | 22.0   | 3.36       | 84.3        |
| ✓        | ✓       | ✓        | Quaternion     | 1,1,1,1,2,3,4,5,5,6          | 2.9     | 2.5    | 5.24       | 15.2        |
| <b>√</b> | ✓       | ×        | 6D             | 3,6,8,10,11,13,13,15,21,22   | 12.2    | 12.0   | 3.75       | 45.7        |
| ✓        | ×       | ✓        | 6D             | 2,6,10,11,13,14,16,18,21,21  | 13.2    | 13.5   | 3.19       | 42.1        |
| ✓        | ×       | ×        | 6D             | 1,3,6,6,7,8,9,11,11,13       | 7.5     | 7.5    | 4.03       | 30.2        |
| ×        | ×       | ×        | 6D             | 1,1,1,1,1,1,1,1,2            | 1.1     | 1.0    | 2.82       | 3.10        |
| ✓        | ✓       | ✓        | ×              | 2,4,5,8,11,12,13,16,17,24    | 11.2    | 11.5   | 2.59       | 29.0        |

<span id="page-16-0"></span>![](_page_16_Picture_2.jpeg)

Figure 6 Visualization results of force-adaptive grasping tasks in Real-world and simulation environments

duration of 3.36 seconds per success. Replacing the 6D rotation representation with quaternions resulted in a significant performance degradation, reducing the average consecutive successes to 2.9. This sharp decline underscores the importance of a continuous and singularity-free rotation representation for policy learning.

Removing the force-weighted contact center reduced the success count to 12.2, while omitting contact force feedback resulted in 13.2 successes. When both tactile-based components were ablated, performance further decreased to 7.5 consecutive successes. The baseline configuration without any contact sensing performed poorest, averaging only 1.1 successes, which highlights the indispensable role of tactile feedback for in-hand manipulation. Additionally, a policy without explicit orientation feedback achieved 11.2 successes, still underperforming compared to the full observation setup.

These results unequivocally demonstrate that tactile information (i.e., particularly force, weighted contact features and measured contact forces), as well as the 6D object orientation representation are critical for robust and continuous in-hand rotation.

### 4.3 Simulation and Experimental Results

As illustrated in Fig. 6 and Fig. 7, the results from both simulation and real-world experiments are presented for the tasks of force-adaptive grasping and in-hand rotation. A high degree of behavioral consistency is observed between the simulated policy and its real-world execution, demonstrating effective sim-to-real transfer. The policy exhibits remarkably similar motion trajectories and contact patterns across both domains.

Fig. 8 further demonstrates the response to different force commands, showing that the policy modulates grasp intensity as intended. Moreover, as shown in Fig. 9, the policy successfully grasps novel irregular objects not encountered during training, maintaining stable and conforming contact, indicating strong generalization capability to unseen object geometries. This consistency in motion and force response underscores the efficacy of our simulation framework and policy learning approach in bridging the reality gap.

<span id="page-17-0"></span>![](_page_17_Figure_0.jpeg)

Figure 7 Visualization results of in-hand manipulation tasks in real-world and simulation environments

<span id="page-17-1"></span>![](_page_17_Figure_2.jpeg)

<span id="page-17-2"></span>Figure 8 Grasping objects with controllable magnitudes of grasping forces – from low to high strength.

![](_page_17_Figure_4.jpeg)

Figure 9 Force-adaptive grasping of irregularly shaped objects that are unseen during training.

# 5 Discussion and Concluding Remarks

This work successfully demonstrated that robust, force-sensitive dexterous manipulation can be achieved through simulation-trained policies deployed zero-shot on real hardware. Our integrated approach, which combines full-state tactile-torque observations, computationally efficient tactile simulation, and targeted actuator modeling, effectively bridged the primary sim-to-real gaps that have historically plagued dexterous manipulation. The results show that our policies not only perform the tasks of force-adaptive grasping and in-hand rotation but do so with a level of robustness and generalization that is critical for real-world applications.

**Generalization and Robustness.** A key strength of our method is its ability to generalize. The policy's successful manipulation of unseen, irregularly shaped objects (Fig. 9) indicates that it learned fundamental physical principles of grasping and manipulation, rather than simply memorizing specific object geometries.

This emergent robustness is a direct consequence of our extensive domain randomization across contact physics, object properties, and, most importantly, actuator dynamics. By forcing the policy to adapt to a wide spectrum of non-ideal motor behaviors (e.g., randomized torque-speed saturation, backlash), we ensured it would not overfit to a perfect simulated actuator model, thereby achieving remarkable stability on the real hardware.

**Limitations and Future Work.** Despite its success, our approach has limitations that point toward fruitful future research directions. First, our current tactile simulation, while fast and effective, is a geometric approximation. Integrating a real-time, differentiable soft-body simulator could capture more nuanced contact phenomena like shear forces and multi-directional object slip, potentially unlocking even more delicate manipulation skills. Second, our tasks, though complex, were executed in a structured setting with a fixed hand. The logical next step is to integrate these dexterous hand policies with a mobile manipulator for tasks requiring whole-body coordination and mobility. Finally, our state estimation relied on external vision and onboard IMUs. Learning to perform these tasks directly from raw visual and tactile sensory inputs, without privileged state information, remains a significant but necessary challenge for achieving full autonomy.

**Broader Impact.** This work serves as a comprehensive handbook for sim-to-real dexterous manipulation. The techniques for current-to-torque calibration and actuator randomization are not limited to the xHand but are applicable to the growing ecosystem of affordable, current-controlled robotic hands. By providing a clear path to bypass the need for expensive joint-level torque sensors and computationally prohibitive tactile simulators, we lower the barrier to entry for research on force-based manipulation.

**Conclusion.** In conclusion, we have presented a holistic solution to the sim-to-real transfer problem for dexterous, force-controlled manipulation. We showed that by explicitly addressing the gaps in tactile sensing, contact physics, and, most critically, actuator dynamics, it is possible to train simulation policies that execute robustly and effectively on real hardware in a zero-shot manner.

This study confirms that policies trained with these carefully designed components can execute complex force-adaptive tasks on physical hardware with zero-shot deployment. This framework offers a practical and reproducible recipe for robust sim-to-real dexterous manipulation that underpins the wider deployment of dexterous robotic hands to operate autonomously and safely in human-centric environments where fine force adaptation and active regulation of interaction forces are indispensable.

# **6 Contributions and Acknowledgements**

- Research Ideas and Conceptualization: Haoyu Dong, Zhibin Li, Zhe Zhao.
- Training: Haoyu Dong, Zhengmao He, Yang Li, Zhe Zhao.
- Evaluation: Haoyu Dong, Zhengmao He, Zhe Zhao.
- Hardware Infrastructure Development: Haoyu Dong, Zhe Zhao.
- Software Infrastructure Development: Haoyu Dong, Zhengmao He, Yang Li, Xinyu Yi, Zhe Zhao.
- Writing: All authors.
- Research Direction and Team Lead: Zhibin Li.

We express our sincere appreciation to Wenjia Zhu for fostering an emphasis on high-impact research. The authors thank Yonghui Wu for his strategic insight in the direction of research priorities. The authors also acknowledge the Department Head, Hang Li, for his dedicated support throughout this project.

# **References**

- <span id="page-20-8"></span>[1] Iretiayo Akinola, Jie Xu, Jan Carius, Dieter Fox, and Yashraj Narang. Tacsl: A library for visuotactile sensor simulation and learning. IEEE Transactions on Robotics, 41:2645–2661, 2025. ISSN 1941-0468. doi: 10.1109/tro.2025.3547267.
- <span id="page-20-6"></span>[2] Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, Jonas Schneider, Nikolas Tezak, Jerry Tworek, Peter Welinder, Lilian Weng, Qiming Yuan, Wojciech Zaremba, Lei Zhang, and OpenAI. Solving rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.
- <span id="page-20-14"></span>[3] Sébastien Briot and Maxime Gautier. Global identification of joint drive gains and dynamic parameters of parallel robots. Multibody System Dynamics, 136, 01 2013. doi: 10.1007/s11044-013-9403-6.
- <span id="page-20-9"></span>[4] Weihang Chen, Jing Xu, Fanbo Xiang, Xiaodi Yuan, Hao Su, and Rui Chen. General-purpose sim2real protocol for learning contact-rich manipulation with marker-based visuotactile sensors. IEEE Transactions on Robotics, 40:1509–1526, 2024. doi: 10.1109/TRO.2024.3352969.
- <span id="page-20-3"></span>[5] Xuxin Cheng, Kexin Shi, Ananye Agarwal, and Deepak Pathak. Extreme parkour with legged robots. In Towards Generalist Robots: Learning Paradigms for Scalable Skill Acquisition @ CoRL2023, 2023.
- <span id="page-20-13"></span>[6] Jelmer de Wolde, Luzia Knoedler, Gianluca Garofalo, and Javier Alonso-Mora. Current-based impedance control for interacting with mobile manipulators. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 753–760, 2024. doi: 10.1109/IROS58592.2024.10802856.
- <span id="page-20-1"></span>[7] Nicholas Fey et al. Bridging the sim-to-real gap for athletic loco-manipulation. ICLR 2025 Robot Learning Workshop (extended abstract), 2025.
- <span id="page-20-10"></span>[8] Nolan Fey, Gabriel B. Margolis, Martin Peticco, and Pulkit Agrawal. Bridging the sim-to-real gap for athletic loco-manipulation, 2025.
- <span id="page-20-15"></span>[9] Generic. Xhand product page. <https://www.robotera.com/en/goods1/4.html>.
- <span id="page-20-12"></span>[10] Tobias Gold, Andreas Völz, and Knut Graichen. External torque estimation for an industrial robot arm using joint torsion and motor current measurements. IFAC-PapersOnLine, 52:352–357, 01 2019. doi: 10.1016/j.ifacol. 2019.11.700.
- <span id="page-20-5"></span>[11] Huy Ha, Yihuai Gao, Zipeng Fu, Jie Tan, and Shuran Song. UMI-on-legs: Making manipulation policies mobile with manipulation-centric whole-body controllers. In 8th Annual Conference on Robot Learning, 2024. URL <https://openreview.net/forum?id=3i7j8ZPnbm>.
- <span id="page-20-0"></span>[12] Tuomas Haarnoja, Ben Moran, Guy Lever, Sandy H. Huang, Dhruva Tirumala, Jan Humplik, Markus Wulfmeier, Saran Tunyasuvunakool, Noah Y. Siegel, Roland Hafner, Michael Bloesch, Kristian Hartikainen, Arunkumar Byravan, Leonard Hasenclever, Yuval Tassa, Fereshteh Sadeghi, Nathan Batchelor, Federico Casarini, Stefano Saliceti, Charles Game, Neil Sreendra, Kushal Patel, Marlon Gwira, Andrea Huber, Nicole Hurley, Francesco Nori, Raia Hadsell, and Nicolas Heess. Learning agile soccer skills for a bipedal robot with deep reinforcement learning. Science Robotics, 9(89):eadi8022, 2024. doi: 10.1126/scirobotics.adi8022. URL [https://www.science.](https://www.science.org/doi/abs/10.1126/scirobotics.adi8022) [org/doi/abs/10.1126/scirobotics.adi8022](https://www.science.org/doi/abs/10.1126/scirobotics.adi8022).
- <span id="page-20-11"></span>[13] Linyan Han, Jianliang Mao, Pengfei Cao, Yahui Gan, and Shihua Li. Toward sensorless interaction force estimation for industrial robots using high-order finite-time observers. IEEE Transactions on Industrial Electronics, 69(7): 7275–7284, 2022. doi: 10.1109/TIE.2021.3095820.
- <span id="page-20-7"></span>[14] Ankur Handa, Arthur Allshire, Viktor Makoviychuk, Aleksei Petrenko, Ritvik Singh, Jingzhou Liu, Denys Makoviichuk, Karl Van Wyk, Alexander Zhurkevich, Balakumar Sundaralingam, Yashraj Narang, Jean-Francois Lafleche, Dieter Fox, and Gavriel State. Dextreme: Transfer of agile in-hand manipulation from simulation to reality. In Proc. IEEE Int. Conf. on Robotics and Automation (ICRA), 2023. doi: 10.1109/ICRA48891.2023.10160216. See also arXiv:2210.13702.
- <span id="page-20-4"></span>[15] Junzhe He, Chong Zhang, Fabian Jenelten, Ruben Grandia, Moritz Bächer, and Marco Hutter. Attentionbased map encoding for learning generalized legged locomotion. Science Robotics, 10(105):eadv3604, 2025. doi: 10.1126/scirobotics.adv3604. URL <https://www.science.org/doi/abs/10.1126/scirobotics.adv3604>.
- <span id="page-20-2"></span>[16] Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbabu, Chaoyi Pan, Zeji Yi, Guannan Qu, Kris Kitani, Jessica Hodgins, Linxi "Jim" Fan, Yuke Zhu, Changliu

- Liu, and Guanya Shi. Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills. arXiv preprint arXiv:2502.01143, 2025.
- <span id="page-21-4"></span>[17] Zhengmao He, Kun Lei, Yanjie Ze, Koushil Sreenath, Zhongyu Li, and Huazhe Xu. Learning visual quadrupedal loco-manipulation from demonstrations. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 9102–9109, 2024. doi: 10.1109/IROS58592.2024.10802742.
- <span id="page-21-2"></span>[18] David Hoeller, Nikita Rudin, Dhionis Sako, and Marco Hutter. Anymal parkour: Learning agile navigation for quadrupedal robots. Science Robotics, 9(88):eadi7566, 2024. doi: 10.1126/scirobotics.adi7566. URL [https:](https://www.science.org/doi/abs/10.1126/scirobotics.adi7566) [//www.science.org/doi/abs/10.1126/scirobotics.adi7566](https://www.science.org/doi/abs/10.1126/scirobotics.adi7566).
- <span id="page-21-8"></span>[19] Wenbin Hu, Bidan Huang, Wang Wei Lee, Sicheng Yang, Yu Zheng, and Zhibin Li. Dexterous in-hand manipulation of slender cylindrical objects through deep reinforcement learning with tactile sensing. Robotics and Autonomous Systems, 186:104904, 2025.
- <span id="page-21-11"></span>[20] Binghao Huang, Yixuan Wang, Xinyi Yang, Yiyue Luo, and Yunzhu Li. 3D ViTac:Learning Fine-Grained Manipulation with Visuo-Tactile Sensing. In Proceedings of Robotics: Conference on Robot Learning(CoRL), 2024.
- <span id="page-21-13"></span>[21] Linyi Huang, Hui Zhang, Zijian Wu, Sammy Christen, and Jie Song. Fungrasp: Functional grasping for diverse dexterous hands. IEEE Robotics and Automation Letters, 10(6):6175–6182, 2025. doi: 10.1109/LRA.2025.3561573.
- <span id="page-21-0"></span>[22] Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, and Marco Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics, 4(26):eaau5872, 2019.
- <span id="page-21-3"></span>[23] Hyeongjun Kim, Hyunsik Oh, Jeongsoo Park, Yunho Kim, Donghoon Youm, Moonkyu Jung, Minho Lee, and Jemin Hwangbo. High-speed control and navigation for quadrupedal robots on complex and discrete terrain. Science Robotics, 10(102):eads6192, 2025. doi: 10.1126/scirobotics.ads6192. URL [https://www.science.org/](https://www.science.org/doi/abs/10.1126/scirobotics.ads6192) [doi/abs/10.1126/scirobotics.ads6192](https://www.science.org/doi/abs/10.1126/scirobotics.ads6192).
- <span id="page-21-16"></span>[24] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023.
- <span id="page-21-1"></span>[25] Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. Science robotics, 5(47):eabc5986, 2020.
- <span id="page-21-6"></span>[26] Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Guy Tevet, Koushil Sreenath, and C. Karen Liu. Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2508.08241) [2508.08241](https://arxiv.org/abs/2508.08241).
- <span id="page-21-9"></span>[27] Toru Lin, Kartik Sachdev, Linxi Fan, Jitendra Malik, and Yuke Zhu. Sim-to-real reinforcement learning for vision-based dexterous manipulation on humanoids. arXiv preprint arXiv:2502.20396, 2025.
- <span id="page-21-12"></span>[28] Yijiong Lin, Alex Church, Max Yang, Haoran Li, John Lloyd, Dandan Zhang, and Nathan F. Lepora. Bi-touch: Bimanual tactile manipulation with sim-to-real deep reinforcement learning. IEEE Robotics and Automation Letters, 8(9):5472–5479, 2023. doi: 10.1109/LRA.2023.3295991.
- <span id="page-21-5"></span>[29] Minghuan Liu, Zixuan Chen, Xuxin Cheng, Yandong Ji, Rizhao Qiu, Ruihan Yang, and Xiaolong Wang. Visual whole-body control for legged loco-manipulation. The 8th Conference on Robot Learning, 2024.
- <span id="page-21-14"></span>[30] Kendall Lowrey, Svetoslav Kolev, Jeremy Dao, Aravind Rajeswaran, and Emanuel Todorov. Reinforcement learning for non-prehensile manipulation: Transfer from simulation to physical system. 2018 IEEE International Conference on Simulation, Modeling, and Programming for Autonomous Robots (SIMPAR), pages 35–42, 2018.
- <span id="page-21-7"></span>[31] Robert McCarthy, Daniel CH Tan, Dominik Schmidt, Fernando Acero, Nathan Herr, Yilun Du, Thomas G Thuruthel, and Zhibin Li. Towards generalist robot learning from internet video: A survey. Journal of Artificial Intelligence Research, 83, 2025.
- <span id="page-21-15"></span>[32] Mayank Mittal, Calvin Yu, Qinxi Yu, Jingzhou Liu, Nikita Rudin, David Hoeller, Jia Lin Yuan, Ritvik Singh, Yunrong Guo, Hammad Mazhar, Ajay Mandlekar, Buck Babich, Gavriel State, Marco Hutter, and Animesh Garg. Orbit: A unified simulation framework for interactive robot learning environments. IEEE Robotics and Automation Letters, 8(6):3740–3747, 2023. doi: 10.1109/LRA.2023.3270034.
- <span id="page-21-10"></span>[33] Duc Huy Nguyen, Tim Schneider, Guillaume Duret, Alap Kshirsagar, Boris Belousov, and Jan Peters. Tacex: Gelsight tactile simulation in isaac sim – combining soft-body and visuotactile simulators, 2024.

- <span id="page-22-6"></span>[34] OpenAI, Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafał Józefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, Jonas Schneider, Szymon Sidor, Josh Tobin, Peter Welinder, Lilian Weng, and Wojciech Zaremba. Learning dexterous in-hand manipulation. arXiv preprint arXiv:1808.00177, 2018. doi: 10.48550/arXiv.1808.00177.
- <span id="page-22-3"></span>[35] Tifanny Portela, Gabriel B. Margolis, Yandong Ji, and Pulkit Agrawal. Learning force control for legged manipulation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 15366–15372, 2024. doi: 10.1109/ICRA57147.2024.10611066.
- <span id="page-22-13"></span>[36] Young-Ha Shin, Tae-Gyu Song, Gwanghyeon Ji, and Hae-Won Park. Actuator-constrained reinforcement learning for high-speed quadrupedal locomotion, 2023.
- <span id="page-22-12"></span>[37] Leon Sievers, Johannes Pitz, and Berthold Bäuml. Learning purely tactile in-hand manipulation with a torquecontrolled hand. In International Conference on Robotics and Automation, pages 2745–2751, 2022. doi: 10.1109/ ICRA46639.2022.9812093.
- <span id="page-22-8"></span>[38] Zhaole Sun, Kai Yuan, Wenbin Hu, Chuanyu Yang, and Zhibin Li. Learning pregrasp manipulation of objects from ungraspable poses. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 9917–9923. IEEE, 2020.
- <span id="page-22-0"></span>[39] Dhruva Tirumala, Markus Wulfmeier, Ben Moran, Sandy Huang, Jan Humplik, Guy Lever, Tuomas Haarnoja, Leonard Hasenclever, Arunkumar Byravan, Nathan Batchelor, Neil Sreendra, Kushal Patel, Marlon Gwira, Francesco Nori, Martin Riedmiller, and Nicolas Heess. Learning robot soccer from egocentric vision with deep reinforcement learning, 2024. URL <https://arxiv.org/abs/2405.02425>.
- <span id="page-22-7"></span>[40] Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomization for transferring deep neural networks from simulation to the real world. In Proc. IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS), Vancouver, Canada, 2017. doi: 10.1109/IROS.2017.8202133.
- <span id="page-22-10"></span>[41] Eleftherios Triantafyllidis, Fernando Acero, Zhaocheng Liu, and Zhibin Li. Hybrid hierarchical learning for solving complex sequential tasks using the robotic manipulation network roman. Nature Machine Intelligence, 5(9): 991–1005, 2023.
- <span id="page-22-14"></span>[42] Arne Wahrburg, Johannes Bös, Kim D. Listmann, Fan Dai, Björn Matthias, and Hao Ding. Motor-current-based estimation of cartesian contact forces and torques for robotic manipulators and its application to force control. IEEE Transactions on Automation Science and Engineering, 15(2):879–886, 2018. doi: 10.1109/TASE.2017.2691136.
- <span id="page-22-1"></span>[43] Chuanyu Yang, Kai Yuan, Qiuguo Zhu, Wanming Yu, and Zhibin Li. Multi-expert learning of adaptive legged locomotion. Science Robotics, 5(49):eabb2174, 2020.
- <span id="page-22-4"></span>[44] Zhao-Heng Yin, Binghao Huang, Yuzhe Qin, Qifeng Chen, and Xiaolong Wang. Rotating without seeing: Towards in-hand dexterity through touch. In Robotics: Science and Systems (RSS), Daegu, Republic of Korea, 2023. doi: 10.15607/RSS.2023.XIX.036.
- <span id="page-22-5"></span>[45] Ying Yuan, Haichuan Che, Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Kang-Won Lee, Yi Wu, Soo-Chul Lim, and Xiaolong Wang. Robot synesthesia: In-hand manipulation with visuotactile sensing. arXiv preprint arXiv:2312.01853, 2023.
- <span id="page-22-11"></span>[46] Chaofan Zhang, Shaowei Cui, Jingyi Hu, Tianyu Jiang, Tiandong Zhang, Rui Wang, and Shuo Wang. Tacflex: Multimode tactile imprints simulation for visuotactile sensors with coating patterns. IEEE Transactions on Robotics, 41:3965–3985, 2025. doi: 10.1109/TRO.2025.3576970.
- <span id="page-22-9"></span>[47] Hui Zhang, Zijian Wu, Linyi Huang, Sammy Christen, and Jie Song. Robustdexgrasp: Robust dexterous grasping of general objects from single-view perception. arXiv preprint arXiv:2504.05287, 2025.
- <span id="page-22-15"></span>[48] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5745–5753, 2019.
- <span id="page-22-2"></span>[49] Ziwen Zhuang, Zipeng Fu, Jianren Wang, Christopher Atkeson, Sören Schwertfeger, Chelsea Finn, and Hang Zhao. Robot parkour learning. In Conference on Robot Learning (CoRL), 2023.