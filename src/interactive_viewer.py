"""
Interactive Model Viewer for Highway-Env
==========================================
Real-time visualization with adjustable environment parameters.

Usage:
    python src/interactive_viewer.py

Controls:
    SPACE  - Pause / Resume
    ESC/Q  - Quit
"""

import gymnasium
import highway_env
import pygame
import numpy as np
import torch
import sys
import os
from pathlib import Path
from stable_baselines3 import DQN, PPO

# =============================================================================
#  MODEL REGISTRY  -  display name -> (path, loader_class)
# =============================================================================

MODEL_REGISTRY = {
    "dqn_baseline": ("highway_dqn/dqn_baseline_1M.zip", DQN),
    "dqn_accel":    ("highway_dqn_accel/dqn_accel_final_1M.zip", DQN),
    "ppo_baseline": ("models/ppo_baseline.zip", PPO),
    "dqn_plr":      ("models/dqn_plr.zip", DQN),
}

# Observation config matching the trained models (shape 7x5)
OBS_CFG = {
    "type": "Kinematics",
    "vehicles_count": 7,
    "features": ["presence", "x", "y", "vx", "vy"],
    "features_range": {"x": [-100, 100], "y": [-100, 100],
                       "vx": [-20, 20], "vy": [-20, 20]},
    "absolute": False, "normalize": True,
    "see_behind": True, "order": "sorted",
}

# =============================================================================
#  COLORS
# =============================================================================

C_BG    = (15, 17, 23)
C_CARD  = (27, 32, 40)
C_BORD  = (48, 54, 61)
C_TXT   = (240, 246, 252)
C_TXT2  = (139, 148, 158)
C_MUT   = (89, 98, 108)
C_CYAN  = (63, 185, 207)
C_GREEN = (63, 185, 80)
C_RED   = (248, 81, 73)
C_YEL   = (210, 153, 34)

# =============================================================================
#  UI HELPERS
# =============================================================================

def txt(font, s, color):
    return font.render(str(s) if s else " ", True, color)

def rrect(surf, color, rect, r=6):
    pygame.draw.rect(surf, color, rect, border_radius=r)


class Slider:
    """Horizontal slider."""
    def __init__(self, x, y, w, lo, hi, val, label, step=1, fmt="{:.0f}"):
        self.x, self.y, self.w = x, y, w
        self.lo, self.hi, self.val = lo, hi, val
        self.label, self.step, self.fmt = label, step, fmt
        self.dragging = False

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self._hit(ev.pos):
            self.dragging = True; self._set(ev.pos[0]); return True
        if ev.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if ev.type == pygame.MOUSEMOTION and self.dragging:
            self._set(ev.pos[0]); return True
        return False

    def _hit(self, pos):
        return self.x-10 <= pos[0] <= self.x+self.w+10 and self.y-10 <= pos[1] <= self.y+30

    def _set(self, mx):
        r = max(0.0, min(1.0, (mx - self.x) / self.w))
        self.val = max(self.lo, min(self.hi, round((self.lo + r*(self.hi-self.lo)) / self.step) * self.step))

    def draw(self, surf, sfont):
        hx = self.x + int((self.val-self.lo)/(self.hi-self.lo)*self.w)
        ty = self.y + 6
        pygame.draw.rect(surf, C_BORD, (self.x, ty, self.w, 6), border_radius=3)
        if hx > self.x:
            pygame.draw.rect(surf, C_CYAN, (self.x, ty, hx-self.x, 6), border_radius=3)
        pygame.draw.circle(surf, (255,255,255), (hx, ty+3), 8)
        pygame.draw.circle(surf, C_CYAN, (hx, ty+3), 8, 2)
        surf.blit(txt(sfont, self.label, C_TXT2), (self.x, self.y-16))
        vs = txt(sfont, self.fmt.format(self.val), C_TXT)
        pygame.draw.rect(surf, C_BORD, (self.x+self.w+8, self.y-1, vs.get_width()+10, 20), border_radius=4)
        surf.blit(vs, (self.x+self.w+13, self.y+1))


class RadioGroup:
    """Radio-button selector."""
    def __init__(self, x, y, w, options, selected, label):
        self.x, self.y, self.w = x, y, w
        self.options, self.selected, self.label = options, selected, label
        self.bh = 24

    def handle(self, ev):
        if ev.type != pygame.MOUSEBUTTONDOWN: return False
        mx, my = ev.pos
        for i, opt in enumerate(self.options):
            by = self.y + 18 + i * (self.bh + 3)
            if self.x <= mx <= self.x+self.w and by <= my <= by+self.bh:
                if self.selected != opt:
                    self.selected = opt; return True
        return False

    def draw(self, surf, sfont):
        surf.blit(txt(sfont, self.label, C_TXT2), (self.x, self.y))
        for i, opt in enumerate(self.options):
            by = self.y + 18 + i * (self.bh + 3)
            active = opt == self.selected
            rrect(surf, C_CYAN if active else C_BORD, pygame.Rect(self.x, by, self.w, self.bh))
            surf.blit(txt(sfont, opt, C_BG if active else C_TXT2),
                      (self.x+8, by + (self.bh - sfont.get_height())//2))


class Toggle:
    """Two-state toggle button."""
    def __init__(self, x, y, w, states, idx, label):
        self.x, self.y, self.w, self.h = x, y, w, 24
        self.states, self.idx, self.label = states, idx, label

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            if self.x <= mx <= self.x+self.w and self.y+16 <= my <= self.y+16+self.h:
                self.idx = (self.idx+1) % len(self.states); return True
        return False

    @property
    def value(self): return self.states[self.idx]

    def draw(self, surf, sfont):
        surf.blit(txt(sfont, self.label, C_TXT2), (self.x, self.y))
        by = self.y + 16
        rrect(surf, C_CYAN, pygame.Rect(self.x, by, self.w, self.h))
        v = txt(sfont, self.value, C_BG)
        surf.blit(v, (self.x + (self.w-v.get_width())//2, by + (self.h-v.get_height())//2))

# =============================================================================
#  MAIN
# =============================================================================

def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    available = [k for k, (p, _) in MODEL_REGISTRY.items() if Path(p).exists()]
    if not available:
        print("No model files found. Check MODEL_REGISTRY paths."); sys.exit(1)
    print(f"Available models: {available}")

    cur_model_name = available[0]
    model = DQN.load(MODEL_REGISTRY[cur_model_name][0], device=device)

    # ---- Pygame ----
    pygame.init()
    EW, EH = 600, 150          # env render area
    PW = 260                    # side panel width
    WW, WH = EW + PW, 480      # total window

    screen = pygame.display.set_mode((WW, WH))
    pygame.display.set_caption("Highway Interactive Viewer")
    clock = pygame.time.Clock()
    sfont = pygame.font.SysFont("Segoe UI", 12)
    bfont = pygame.font.SysFont("Segoe UI", 14, bold=True)

    # ---- Widgets ----
    px, pw = EW + 12, PW - 24
    sl_lanes = Slider(px, 50,  pw, 2, 6, 3,       "Lanes",    step=1)
    sl_vehs  = Slider(px, 95,  pw, 5, 50, 25,     "Vehicles", step=5)
    sl_dens  = Slider(px, 140, pw, 0.5, 3.0, 1.0, "Density",  step=0.1, fmt="{:.1f}")
    sl_fps   = Slider(px, 185, pw, 5, 120, 30,    "FPS",      step=5)
    tg_aggr  = Toggle(px, 220, pw, ["Normal", "Aggressive"], 0, "Vehicles Behavior")
    rg_model = RadioGroup(px, 270, pw, available, available[0], "Model")

    widgets = [sl_lanes, sl_vehs, sl_dens, sl_fps, tg_aggr, rg_model]

    # ---- Create env once ----
    base_cfg = {
        "lanes_count": 3, "vehicles_count": 25, "vehicles_density": 1.0,
        "duration": 60, "simulation_frequency": 30, "policy_frequency": 2,
        "collision_reward": -10.0, "high_speed_reward": 0.3,
        "right_lane_reward": 0.0, "lane_change_reward": 0.0,
        "reward_speed_range": [20, 30], "normalize_reward": False,
        "observation": OBS_CFG,
        "render_agent": True, "offscreen_rendering": True,
    }
    env = gymnasium.make("highway-v0", config=base_cfg, render_mode="rgb_array")

    def read_cfg():
        """Read current widget values as a dict."""
        return {"lanes": int(sl_lanes.val), "vehicles": int(sl_vehs.val),
                "density": round(sl_dens.val, 1), "aggressive": tg_aggr.idx == 1}

    def apply_cfg(cfg):
        """Push widget config into the env (applied on next reset)."""
        vtype = "highway_env.vehicle.behavior.AggressiveVehicle" if cfg["aggressive"] \
                else "highway_env.vehicle.behavior.IDMVehicle"
        env.unwrapped.config.update({
            "lanes_count": cfg["lanes"],
            "vehicles_count": cfg["vehicles"],
            "vehicles_density": cfg["density"],
            "other_vehicles_type": vtype,
        })

    prev_cfg = read_cfg()
    apply_cfg(prev_cfg)
    obs, _ = env.reset()

    # ---- State ----
    paused = False
    episode = 0
    ep_reward = 0.0
    ep_steps = 0
    total_ep = 0
    total_crashes = 0
    total_reward = 0.0

    running = True
    while running:
        # ---- Events ----
        cfg_changed = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
            for w in widgets:
                if w.handle(ev):
                    cfg_changed = True

        if not running:
            break

        # ---- Config change -> update env in-place and reset ----
        cfg = read_cfg()
        if cfg != prev_cfg:
            cfg_changed = True

        if rg_model.selected != cur_model_name:
            cur_model_name = rg_model.selected
            path, cls = MODEL_REGISTRY[cur_model_name]
            model = cls.load(path, device=device)
            print(f"Switched to: {cur_model_name}")
            cfg_changed = True

        if cfg_changed:
            apply_cfg(cfg)
            obs, _ = env.reset()
            prev_cfg = cfg.copy()
            ep_reward = 0.0
            ep_steps = 0
            episode += 1

        # ---- Step ----
        if not paused:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            if done or truncated:
                crashed = done and not truncated
                total_ep += 1
                total_reward += ep_reward
                if crashed:
                    total_crashes += 1
                tag = "CRASH" if crashed else "OK"
                print(f"Ep {episode+1}: {tag}  reward={ep_reward:.1f}  steps={ep_steps}")
                ep_reward = 0.0
                ep_steps = 0
                episode += 1
                obs, _ = env.reset()

        # ---- Draw ----
        screen.fill(C_BG)

        # Env frame (scale to fit EW x EH)
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surf = pygame.transform.scale(surf, (EW, EH))
            screen.blit(surf, (0, 0))
        pygame.draw.rect(screen, C_BORD, (0, 0, EW, EH), 2)

        # Info bar
        iy = EH + 8
        rrect(screen, C_CARD, pygame.Rect(8, iy, EW - 16, 42), 6)
        screen.blit(txt(bfont, f"Ep {episode+1}", C_TXT), (16, iy+4))
        rc = C_GREEN if ep_reward >= 0 else C_RED
        screen.blit(txt(sfont, f"Reward: {ep_reward:.1f}", rc), (110, iy+6))
        screen.blit(txt(sfont, f"Steps: {ep_steps}", C_TXT2), (230, iy+6))
        if paused:
            screen.blit(txt(bfont, "PAUSED", C_YEL), (340, iy+4))
        if total_ep > 0:
            avg_r = total_reward / total_ep
            cr = 100 * total_crashes / total_ep
            screen.blit(txt(sfont,
                f"Avg: {avg_r:.2f} | Crash: {cr:.0f}% | {cur_model_name}",
                C_MUT), (16, iy+24))

        # Side panel
        rrect(screen, C_CARD, pygame.Rect(EW+4, 4, PW-8, WH-8), 8)
        pygame.draw.rect(screen, C_BORD, pygame.Rect(EW+4, 4, PW-8, WH-8), 1, border_radius=8)
        screen.blit(txt(bfont, "Settings", C_TXT), (px, 12))

        for w in widgets:
            w.draw(screen, sfont)

        screen.blit(txt(sfont, "SPACE: pause  ESC: quit", C_MUT), (px, WH-24))

        pygame.display.flip()
        clock.tick(int(sl_fps.val))

    # ---- Cleanup ----
    try:
        env.close()
    except Exception:
        pass
    pygame.quit()

    if total_ep > 0:
        print(f"\nDone. {total_ep} episodes, avg reward {total_reward/total_ep:.2f}, "
              f"crash rate {100*total_crashes/total_ep:.0f}%")


if __name__ == "__main__":
    main()
