import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

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
    state = [x1, x2]
    x1 = pendulum angle
    x2 = angular velocity
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
    k1_vec, u1, _, _ = dynamics_article_literal(t, state, k1, c1, with_disturbance)

    k2_state = state + 0.5 * h * k1_vec
    k2_vec, u2, _, _ = dynamics_article_literal(
        t + 0.5 * h,
        k2_state,
        k1,
        c1,
        with_disturbance
    )

    k3_state = state + 0.5 * h * k2_vec
    k3_vec, u3, _, _ = dynamics_article_literal(
        t + 0.5 * h,
        k3_state,
        k1,
        c1,
        with_disturbance
    )

    k4_state = state + h * k3_vec
    k4_vec, u4, _, _ = dynamics_article_literal(
        t + h,
        k4_state,
        k1,
        c1,
        with_disturbance
    )

    new_state = state + (h / 6.0) * (k1_vec + 2 * k2_vec + 2 * k3_vec + k4_vec)

    # uśrednione sterowanie reprezentujące krok RK4
    u_rk4 = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0

    return new_state, u_rk4


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

    _, u0, _, _ = dynamics_article_literal(0.0, state, k1, c1, with_disturbance)
    u_out[0] = u0

    for i in range(1, n_out):
        t_target = t_out[i]

        u_accum = 0.0
        u_count = 0

        while t_curr < t_target - 1e-15:
            h = min(dt_int, t_target - t_curr)

            state, u_step = rk4_step(
                t_curr, state, h, k1, c1, with_disturbance
            )

            u_accum += u_step
            u_count += 1
            t_curr += h

        x1_out[i] = state[0]
        x2_out[i] = state[1]

        r_out[i] = reference_signal(t_curr)[0]
        d_out[i] = disturbance_signal(t_curr, enabled=with_disturbance)

        u_out[i] = u_accum / u_count if u_count > 0 else u_out[i - 1]

    return t_out, x1_out, x2_out, r_out, u_out, d_out


def simulate_article_exact_with_raw_u(k1, c1, t_end, with_disturbance=False):
    """
    Dodatkowa symulacja do logowania 'surowego' u z poziomu integracji.
    Używamy jej tylko do wykresu porównawczego raw vs avg.
    """
    t_out = np.arange(0.0, t_end + dt_plot, dt_plot)
    n_out = len(t_out)

    x1_out = np.zeros(n_out)
    x2_out = np.zeros(n_out)
    r_out = np.zeros(n_out)
    u_out = np.zeros(n_out)
    d_out = np.zeros(n_out)

    t_u = []
    u_raw = []

    state = np.array([0.0, 0.0], dtype=float)
    t_curr = 0.0

    x1_out[0] = state[0]
    x2_out[0] = state[1]
    r_out[0] = reference_signal(0.0)[0]
    d_out[0] = disturbance_signal(0.0, enabled=with_disturbance)

    _, u0, _, _ = dynamics_article_literal(0.0, state, k1, c1, with_disturbance)
    u_out[0] = u0
    t_u.append(0.0)
    u_raw.append(u0)

    for i in range(1, n_out):
        t_target = t_out[i]
        u_accum = 0.0
        u_count = 0

        while t_curr < t_target - 1e-15:
            h = min(dt_int, t_target - t_curr)

            _, u_start, _, _ = dynamics_article_literal(
                t_curr, state, k1, c1, with_disturbance
            )
            t_u.append(t_curr)
            u_raw.append(u_start)

            state, u_step = rk4_step(
                t_curr, state, h, k1, c1, with_disturbance
            )

            u_accum += u_step
            u_count += 1
            t_curr += h

        x1_out[i] = state[0]
        x2_out[i] = state[1]
        r_out[i] = reference_signal(t_curr)[0]
        d_out[i] = disturbance_signal(t_curr, enabled=with_disturbance)
        u_out[i] = u_accum / u_count if u_count > 0 else u_out[i - 1]

    return t_out, x1_out, x2_out, r_out, u_out, d_out, np.array(t_u), np.array(u_raw)


def reconstruct_cart_position_from_force(t, theta, theta_dot, u, M, m, l):
    dt = t[1] - t[0]
    theta_ddot = np.gradient(theta_dot, dt)

    x_ddot = (
        u
        - m * l * np.cos(theta) * theta_ddot
        + m * l * np.sin(theta) * (theta_dot ** 2)
    ) / (M + m)

    x_dot = np.zeros_like(t)
    x = np.zeros_like(t)

    for i in range(1, len(t)):
        x_dot[i] = x_dot[i - 1] + 0.5 * (x_ddot[i - 1] + x_ddot[i]) * dt
        x[i] = x[i - 1] + 0.5 * (x_dot[i - 1] + x_dot[i]) * dt

    return x, x_dot, x_ddot


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


def plot_u_comparison(t_avg, u_avg, t_raw, u_raw, title):
    plt.figure(figsize=(9.0, 4.5))
    plt.plot(t_raw, u_raw, linewidth=0.6, label="u_raw (integration-level)")
    plt.plot(t_avg, u_avg, linewidth=1.8, label="u_avg (averaged per output sample)")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xlabel("Czas [s]")
    plt.ylabel("Sterowanie u")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cart_states(t, x, x_dot, x_ddot, u, title):
    fig, ax1 = plt.subplots(figsize=(8.5, 5.0))

    ax1.plot(t, x, linewidth=1.5, label="x (pozycja wózka)")
    ax1.plot(t, x_dot, linewidth=1.3, label="x_dot (prędkość wózka)")
    ax1.plot(t, x_ddot, linewidth=1.2, label="x_ddot (przyspieszenie wózka)")
    ax1.set_xlabel("Czas [s]")
    ax1.set_ylabel("x, x_dot, x_ddot")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t, u, linewidth=1.1, linestyle="--", label="u (siła sterująca)")
    ax2.set_ylabel("u")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def animate_cart_pendulum(
    t,
    x_cart,
    theta_real,
    theta_ref,
    u,
    pendulum_length=1.0,
    interval=40,
    cart_width=0.45,
    cart_height=0.22,
    wheel_radius=0.05,
    camera_half_width=2.5,
):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_aspect("equal")
    ax.set_ylim(-0.35, pendulum_length + 0.8)
    ax.grid(True, alpha=0.25)
    ax.set_title("Animacja wózka z odwróconym wahadłem")
    ax.set_xlabel("Pozycja pozioma")
    ax.set_ylabel("Pozycja pionowa")

    track_y = 0.0
    cart_y = 0.10
    pivot_y = cart_y + 0.5 * cart_height

    track_line, = ax.plot([], [], 'k-', lw=1.2, label="Tor wózka")

    cart_body = patches.Rectangle(
        (-cart_width / 2, cart_y - cart_height / 2),
        cart_width,
        cart_height,
        fill=False,
        linewidth=2,
        edgecolor='black'
    )
    wheel_left = patches.Circle(
        (0.0, cart_y - cart_height / 2 - wheel_radius),
        wheel_radius,
        fill=False,
        linewidth=1.6
    )
    wheel_right = patches.Circle(
        (0.0, cart_y - cart_height / 2 - wheel_radius),
        wheel_radius,
        fill=False,
        linewidth=1.6
    )

    ax.add_patch(cart_body)
    ax.add_patch(wheel_left)
    ax.add_patch(wheel_right)

    line_real, = ax.plot([], [], 'o-', lw=3, label="Wahadło rzeczywiste")
    line_ref, = ax.plot([], [], 'o--', lw=2, label="Wahadło zadane")
    force_arrow, = ax.plot([], [], lw=2)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    force_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    pos_text = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    ax.legend(loc="upper right")

    def init():
        line_real.set_data([], [])
        line_ref.set_data([], [])
        force_arrow.set_data([], [])
        track_line.set_data([], [])
        time_text.set_text("")
        force_text.set_text("")
        pos_text.set_text("")
        return (
            track_line,
            cart_body, wheel_left, wheel_right,
            line_real, line_ref, force_arrow,
            time_text, force_text, pos_text
        )

    def update(frame):
        xc = x_cart[frame]
        th_r = theta_real[frame]
        th_ref = theta_ref[frame]
        u_curr = u[frame]

        ax.set_xlim(xc - camera_half_width, xc + camera_half_width)

        track_line.set_data(
            [xc - 1.2 * camera_half_width, xc + 1.2 * camera_half_width],
            [track_y, track_y]
        )

        cart_body.set_xy((xc - cart_width / 2, cart_y - cart_height / 2))
        wheel_left.center = (xc - 0.13, cart_y - cart_height / 2 - wheel_radius)
        wheel_right.center = (xc + 0.13, cart_y - cart_height / 2 - wheel_radius)

        x_tip_real = xc + pendulum_length * np.sin(th_r)
        y_tip_real = pivot_y + pendulum_length * np.cos(th_r)

        x_tip_ref = xc + pendulum_length * np.sin(th_ref)
        y_tip_ref = pivot_y + pendulum_length * np.cos(th_ref)

        line_real.set_data([xc, x_tip_real], [pivot_y, y_tip_real])
        line_ref.set_data([xc, x_tip_ref], [pivot_y, y_tip_ref])

        arrow_scale = 0.001
        arrow_len = np.clip(u_curr * arrow_scale, -0.4, 0.4)
        force_arrow.set_data(
            [xc, xc + arrow_len],
            [cart_y + 0.22, cart_y + 0.22]
        )

        time_text.set_text(f"t = {t[frame]:.2f} s")
        force_text.set_text(f"u = {u_curr:.2f}")
        pos_text.set_text(f"x = {xc:.3f}")

        return (
            track_line,
            cart_body, wheel_left, wheel_right,
            line_real, line_ref, force_arrow,
            time_text, force_text, pos_text
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=interval,
        blit=False,
        repeat=True
    )

    plt.tight_layout()
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

    # ===== CASE 1 =====
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

    # ===== CASE 2 =====
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

    # Dodatkowo: pokaż raw vs avg dla sterowania
    _, _, _, _, _, _, t_u2, u_raw2 = simulate_article_exact_with_raw_u(
        k1=700,
        c1=102,
        t_end=5.0,
        with_disturbance=False
    )

    plot_u_comparison(
        t2, u2, t_u2, u_raw2,
        "Rzeczywisty sygnał sterujący vs uśredniony sygnał zapisywany"
    )

    # Rekonstrukcja ruchu wózka z u_avg
    x_cart_2, x_cart_dot_2, x_cart_ddot_2 = reconstruct_cart_position_from_force(
        t2, x1_2, x2_2, u2, M, m, l
    )

    plot_cart_states(
        t2, x_cart_2, x_cart_dot_2, x_cart_ddot_2, u2,
        "Stany wózka zrekonstruowane z uśrednionej siły sterującej"
    )

    animate_cart_pendulum(
        t2,
        x_cart_2,
        x1_2,
        r2,
        u2,
        pendulum_length=1.0,
        interval=20
    )

    # ===== CASE 3 =====
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