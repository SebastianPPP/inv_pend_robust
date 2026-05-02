import pygame
import numpy as np
import collections

M, m, l, g = 1.0, 0.1, 0.5, 9.8
k1, c1 = 700, 102
dt = 0.01

WIDTH, HEIGHT = 1200, 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (0, 200, 0)
GRAY = (200, 200, 200)

def dynamics(t, state, with_disturbance=False):
    x1, x2, x_cart, x_cart_dot = state 
    
    x_target = 0.0
    Kp_pos = 0.03
    Kd_pos = 0.12
    
    r = Kp_pos * (x_target - x_cart) - Kd_pos * x_cart_dot
    r = np.clip(r, -0.15, 0.15)
    r_dot = 0.0 
    r_ddot = 0.0
    
    e1 = r - x1
    c1_safe = 15 
    s1 = (r_dot - x2) + c1_safe * e1
    
    c, s = np.cos(x1), np.sin(x1)
    if abs(c) < 1e-9: c = 1e-9 * np.sign(c)

    denom = m * l * c**2 - (M + m)
    
    k1_safe = 50
    smooth_sign = s1 / (abs(s1) + 0.05) 
    
    u = ((M + m) * g * np.tan(x1) 
         - m * l * (x2**2) * s 
         + (k1_safe * smooth_sign + r_ddot - c1_safe * x2 + c1_safe * r_dot) * denom / c)
    
    u = np.clip(u, -150, 150)
    
    d = 5.0 * np.sin(2 * np.pi * t) if with_disturbance else 0.0
    x2_dot = (-(m + M) * g * s + m * l * (x2**2) * c * s + u * c) / denom + d
    
    x_cart_ddot = (u - m * l * c * x2_dot + m * l * s * (x2**2)) / (M + m)
    
    return np.array([x2, x2_dot]), u, x_cart_ddot, r

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Inverted Pendulum - SMC")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
        self.state = np.array([0.0, 0.0, 3.0, 0.0])
        self.t = 0.0
        self.dist_enabled = False
        
        self.history_r = collections.deque(maxlen=400)
        self.history_theta = collections.deque(maxlen=400)
        self.history_u = collections.deque(maxlen=400)
        self.history_x = collections.deque(maxlen=400)

    def draw_plot(self, data_list, x_offset, y_offset, width, height, color, label, ref_list=None):
        pygame.draw.rect(self.screen, BLACK, (x_offset, y_offset, width, height), 1)
        if len(data_list) < 2: return
        
        all_vals = data_list + (ref_list if ref_list else [])
        max_val = max(all_vals) if all_vals else 1.0
        min_val = min(all_vals) if all_vals else -1.0
        if max_val == min_val:
            max_val += 0.1
            min_val -= 0.1
        span = max(max_val - min_val, 0.1)

        def get_y(val):
            return y_offset + height - int((val - min_val) / span * height)

        if ref_list:
            points_ref = [(x_offset + i * (width/400), get_y(v)) for i, v in enumerate(ref_list)]
            if len(points_ref) > 1: pygame.draw.lines(self.screen, GRAY, False, points_ref, 1)

        points = [(x_offset + i * (width/400), get_y(v)) for i, v in enumerate(data_list)]
        pygame.draw.lines(self.screen, color, False, points, 2)
        
        txt = self.font.render(label, True, BLACK)
        self.screen.blit(txt, (x_offset, y_offset - 25))

    def run(self):
        running = True
        while running:
            self.screen.fill(WHITE)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d: self.dist_enabled = not self.dist_enabled

            k_v1, u, a_cart, r = dynamics(self.t, self.state, self.dist_enabled)
            
            self.state[1] += k_v1[1] * dt
            self.state[3] += a_cart * dt
            self.state[0] += self.state[1] * dt
            self.state[2] += self.state[3] * dt
            
            self.t += dt
            
            self.history_theta.append(self.state[0])
            self.history_r.append(r)
            self.history_u.append(u)
            self.history_x.append(self.state[2])

            scale_len = 200
            scale_x = 100
            
            center_x = WIDTH // 2
            cx = int(center_x + self.state[2] * scale_x)
            cy = 300 
            
            pygame.draw.line(self.screen, BLACK, (0, cy + 25), (WIDTH, cy + 25), 2)
            pygame.draw.circle(self.screen, GREEN, (center_x, cy + 25), 7) 
            
            cart_w, cart_h = 80, 40
            pygame.draw.rect(self.screen, BLUE, (cx - cart_w//2, cy - cart_h//2, cart_w, cart_h))
            
            pend_x = cx + np.sin(self.state[0]) * l * scale_len
            pend_y = cy - np.cos(self.state[0]) * l * scale_len
            pygame.draw.line(self.screen, RED, (cx, cy), (pend_x, pend_y), 5)
            pygame.draw.circle(self.screen, RED, (int(pend_x), int(pend_y)), 10)

            self.draw_plot(list(self.history_theta), 30, 550, 350, 200, RED, "Kąt wahadła vs Cel", list(self.history_r))
            self.draw_plot(list(self.history_x), 420, 550, 350, 200, GREEN, "Pozycja wózka (x)")
            self.draw_plot(list(self.history_u), 810, 550, 350, 200, BLUE, "Siła sterująca (u)")

            status = "WŁĄCZONE" if self.dist_enabled else "WYŁĄCZONE"
            self.screen.blit(self.font.render(f"Czas: {self.t:.2f}s", True, BLACK), (20, 20))
            self.screen.blit(self.font.render(f"Zakłócenia (D): {status}", True, BLACK), (20, 45))
            self.screen.blit(self.font.render(f"Pozycja wózka: {self.state[2]:.2f}m", True, BLACK), (20, 70))
            self.screen.blit(self.font.render(f"Kąt wychylenia: {np.degrees(self.state[0]):.1f} st", True, BLACK), (20, 95))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()