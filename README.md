# inv_pend_robust
Classic project about stabilizing inverted pendulum with robust sliding mode control. 

11/03 Update (Sebastian)
Made script for plots, something is wrong with dt either the one in article is different or the equations are wrong.

To do:
- more plots, more situations
- animation in pygame (?)

3/03 Update (Sebastian)
Added v3 script
Simulation of an inverted pendulum on a cart using a cascaded control strategy: a PD controller for cart positioning and Sliding Mode Control (SMC) for pendulum stabilization.

### Overview

The system stabilizes an unstable equilibrium point (upright position) while simultaneously translating the cart to a target coordinate ($x = 0$).

### Key Features
Cascaded Control: Outer PD loop generates target tilt angles; inner SMC loop executes torque/force requirements.
Sliding Mode Control: High robustness against model uncertainties and external disturbances.
Chattering Reduction: Uses a saturation-based smoothing function instead of a raw signum function to prevent high-frequency oscillations.
Real-time Physics: Implements semi-implicit Euler integration for stability at $dt = 0.01s$.