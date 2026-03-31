import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

M = 1.0
m = 0.1
l = 0.5
g = 9.8

dt_plot = 0.01
dt_int = 1e-4


def reference_signal(t):
    r = np.sin(2 * np.pi * t)
    r_dot = 2 * np.pi * np.cos(2 * np.pi * t)
    r_ddot = -4 * np.pi**2 * np.sin(2 * np.pi * t)
    return r, r_dot, r_ddot


def disturbance_signal(t, enabled=False):
    if enabled:
        return 0.5 * np.sin(2 * np.pi * t)
    return 0.0


def dynamics_article_literal(t, state, k1, c1, with_disturbance=False):
    """
    Poprawiona implementacja:
    - zachowany regulator SMC z sign(s1)
    - poprawione człony na x2**2
    """
    x1, x2 = state

    r, r_dot, r_ddot = reference_signal(t)
    d = disturbance_signal(t, enabled=with_disturbance)

    e1 = r - x1
    e1_dot = r_dot - x2
    s1 = e1_dot + c1 * e1

    c = np.cos(x1)
    s = np.sin(x1)

    if abs(c) < 1e-9:
        c = 1e-9 if c >= 0 else -1e-9

    denom = m * l * c**2 - (M + m)

    u = (
        (M + m) * g * np.tan(x1)
        - m * l * (x2**2) * s
        + (k1 * np.sign(s1) + r_ddot - c1 * x2 + c1 * r_dot) * denom / c
    )

    x1_dot = x2
    x2_dot = (
        (-(m + M) * g * s + m * l * (x2**2) * c * s + u * c) / denom
        + d
    )

    return np.array([x1_dot, x2_dot]), u, r, d


def rk4_step(t, state, h, k1, c1, with_disturbance=False):
    k1_vec, _, _, _ = dynamics_article_literal(t, state, k1, c1, with_disturbance)
    k2_vec, _, _, _ = dynamics_article_literal(
        t + 0.5 * h,
        state + 0.5 * h * k1_vec,
        k1,
        c1,
        with_disturbance
    )
    k3_vec, _, _, _ = dynamics_article_literal(
        t + 0.5 * h,
        state + 0.5 * h * k2_vec,
        k1,
        c1,
        with_disturbance
    )
    k4_vec, _, _, _ = dynamics_article_literal(
        t + h,
        state + h * k3_vec,
        k1,
        c1,
        with_disturbance
    )

    new_state = state + (h / 6.0) * (k1_vec + 2 * k2_vec + 2 * k3_vec + k4_vec)
    return new_state


def simulate_article_exact(k1, c1, t_end, with_disturbance=False):
    t_out = np.arange(0.0, t_end + dt_plot, dt_plot)
    n_out = len(t_out)

    x1_out = np.zeros(n_out)
    x2_out = np.zeros(n_out)
    r_out = np.zeros(n_out)
    u_out = np.zeros(n_out)
    d_out = np.zeros(n_out)

    state = np.array([0.0, 0.0], dtype=float)
    t_curr = 0.0

    x1_out[0] = state[0]
    x2_out[0] = state[1]
    r_out[0] = reference_signal(0.0)[0]
    d_out[0] = disturbance_signal(0.0, enabled=with_disturbance)

    for i in range(1, n_out):
        t_target = t_out[i]

        while t_curr < t_target - 1e-15:
            h = min(dt_int, t_target - t_curr)
            state = rk4_step(t_curr, state, h, k1, c1, with_disturbance)
            t_curr += h

        x1_out[i] = state[0]
        x2_out[i] = state[1]

        _, u_i, r_i, d_i = dynamics_article_literal(
            t_curr, state, k1, c1, with_disturbance
        )
        r_out[i] = r_i
        u_out[i] = u_i
        d_out[i] = d_i

    return t_out, x1_out, x2_out, r_out, u_out, d_out


def expanded_limits(y_main, y_ref=None, margin_ratio=0.15, min_half_range=1.0):
    y_all = np.array(y_main, dtype=float)
    if y_ref is not None:
        y_all = np.concatenate([y_all, np.array(y_ref, dtype=float)])

    y_min = np.min(y_all)
    y_max = np.max(y_all)

    center = 0.5 * (y_min + y_max)
    half_range = 0.5 * (y_max - y_min)
    half_range = max(half_range * (1.0 + margin_ratio), min_half_range)

    return center - half_range, center + half_range


def plot_response(t, x1, r, title, ylim=None):
    plt.figure(figsize=(7.0, 4.5))

    plt.plot(t, x1, linewidth=1.6, label="Odpowiedź układu")
    plt.plot(t, r, linestyle="--", linewidth=1.8, label="Sygnał zadany")

    plt.xlim(t[0], t[-1])

    if ylim is None:
        plt.ylim(expanded_limits(x1, r, margin_ratio=0.20, min_half_range=1.5))
    else:
        plt.ylim(ylim)

    plt.xlabel("Czas [s]")
    plt.ylabel("Kąt / pozycja")
    plt.title(title)
    plt.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_states(t, x1, x2, r, title, ylim=None):
    plt.figure(figsize=(7.0, 4.5))

    plt.plot(t, x1, linewidth=1.4, label="x1 (kąt)")
    plt.plot(t, x2, linewidth=1.2, label="x2 (prędkość kątowa)")
    plt.plot(t, r, linestyle="--", linewidth=1.8, label="Sygnał zadany")

    plt.xlim(t[0], t[-1])

    if ylim is None:
        plt.ylim(expanded_limits(np.concatenate([x1, x2]), r, margin_ratio=0.20, min_half_range=8.0))
    else:
        plt.ylim(ylim)

    plt.xlabel("Czas [s]")
    plt.ylabel("Wartości stanów")
    plt.title(title)
    plt.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def animate_pendulum(t, theta_real, theta_ref, length=1.0, interval=80):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-1.2 * length, 1.2 * length)
    ax.set_ylim(-1.2 * length, 1.2 * length)
    ax.grid(True, alpha=0.3)
    ax.set_title("Animacja odwróconego wahadła")

    pivot = np.array([0.0, 0.0])

    line_real, = ax.plot([], [], 'o-', lw=3, label="Wahadło rzeczywiste")
    line_ref, = ax.plot([], [], 'o--', lw=2, label="Wahadło zadane")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")

    ax.plot([0, 0], [0, length], ':', linewidth=1)

    def init():
        line_real.set_data([], [])
        line_ref.set_data([], [])
        time_text.set_text("")
        return line_real, line_ref, time_text

    def update(frame):
        th_r = theta_real[frame]
        th_ref = theta_ref[frame]

        x_real = length * np.sin(th_r)
        y_real = length * np.cos(th_r)

        x_ref = length * np.sin(th_ref)
        y_ref = length * np.cos(th_ref)

        line_real.set_data([pivot[0], x_real], [pivot[1], y_real])
        line_ref.set_data([pivot[0], x_ref], [pivot[1], y_ref])

        time_text.set_text(f"Czas = {t[frame]:.2f} s")
        return line_real, line_ref, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(t), init_func=init,
        interval=interval, blit=True, repeat=True
    )

    plt.show()
    return ani


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans"
    })

    t1, x1_1, x2_1, r1, u1, d1 = simulate_article_exact(
        k1=500,
        c1=100,
        t_end=10.0,
        with_disturbance=False
    )

    plot_response(
        t1, x1_1, r1,
        "Odpowiedź układu (SMC), sygnał zadany sin(2πt), k1 = 500, c1 = 100",
        ylim=(-1.2, 1.2)
    )

    plot_states(
        t1, x1_1, x2_1, r1,
        "Zmienne stanu układu (SMC), k1 = 500, c1 = 100",
        ylim=(-8.0, 8.0)
    )

    t2, x1_2, x2_2, r2, u2, d2 = simulate_article_exact(
        k1=700,
        c1=102,
        t_end=5.0,
        with_disturbance=False
    )

    plot_response(
        t2, x1_2, r2,
        "Odpowiedź układu (SMC), sygnał zadany sin(2πt), k1 = 700, c1 = 102",
        ylim=(-1.5, 1.5)
    )

    plot_states(
        t2, x1_2, x2_2, r2,
        "Zmienne stanu układu (SMC), k1 = 700, c1 = 102",
        ylim=(-10.0, 10.0)
    )

    t3, x1_3, x2_3, r3, u3, d3 = simulate_article_exact(
        k1=700,
        c1=102,
        t_end=5.0,
        with_disturbance=True
    )

    plot_response(
        t3, x1_3, r3,
        "Odpowiedź układu (SMC) z zakłóceniem 0.5·sin(2πt), k1 = 700, c1 = 102",
        ylim=(-1.5, 1.5)
    )

    plot_states(
        t3, x1_3, x2_3, r3,
        "Zmienne stanu układu (SMC) z zakłóceniem, k1 = 700, c1 = 102",
        ylim=(-10.0, 10.0)
    )

    animate_pendulum(t2, x1_2, r2, length=1.0, interval=20)