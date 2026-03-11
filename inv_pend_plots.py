import numpy as np
import matplotlib.pyplot as plt

# Params
m_cart = 1.0
m_pend = 0.1
l_pend = 0.5
g = 9.8
dt = 0.001
t_end = 10.0

def sim_pend(k1, c1, with_disturbance=False):
    t = np.arange(0, t_end + dt, dt)
    n = len(t)

    x1 = np.zeros(n)  # theta
    x2 = np.zeros(n)  # theta_dot
    ref_signal_hist = np.zeros(n)

    x1[0] = 0.0
    x2[0] = 0.0

    for i in range(n-1):
        ti = t[i]

        # signal and its derivatives
        ref_signal = np.sin(2 * np.pi * ti)
        ref_dot = 2 * np.pi * np.cos(2 * np.pi * ti)
        ref_ddot = -4 * np.pi**2 * np.sin(2 * np.pi * ti)

        ref_signal_hist[i] = ref_signal

        # error and sliding variable
        e1 = ref_signal - x1[i]
        e1_dot = ref_dot - x2[i]
        s1 = e1_dot + c1 * e1

        # control u 
        u = (m_cart + m_pend) * g * np.tan(x1[i]) - m_pend * l_pend * x2[i] * np.sin(x1[i]) + (k1 * np.sign(s1) + ref_ddot - c1 * x2[i] + c1 * ref_dot) * (m_pend * l_pend * np.cos(x1[i])**2 - (m_cart + m_pend)) / np.cos(x1[i])
        
        disturbance = np.random.rand(0, 1) * np.sin(2 * np.pi * ti) if with_disturbance else 0.0
        # system dynamics
        x1_dot = x2[i]
        x2_dot = (-(m_cart + m_pend) * g * np.sin(x1[i]) + m_pend * l_pend * x2[i] * np.cos(x1[i]) * np.sin(x1[i]) + u * np.cos(x1[i]))/(m_pend * l_pend * np.cos(x1[i])**2 - (m_cart + m_pend)) + disturbance

        # Euler 
        x1[i+1] = x1[i] + x1_dot * dt
        x2[i+1] = x2[i] + x2_dot * dt

    return t, x1, x2, ref_signal_hist


# plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# test 1 - fig 2 / fig 3
axs[0].set_title("Test 1: k1=500, c1=100")
t, x1, x2, ref_signal = sim_pend(k1=500, c1=100)
axs[0].plot(t, ref_signal, label='theta (rad)', color='blue')
axs[0].plot(t, x2, label='theta_dot (rad/s)', color='orange')
axs[0].legend()

plt.tight_layout()
plt.show()