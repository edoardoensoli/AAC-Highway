"""
Interactive Model Tester with Real-Time Parameter Adjustment
============================================================

A testing tool with interactive sliders in the pygame window.
Parameters take effect after each episode (when the car crashes or ends).

Usage:
    python interactive_tester.py

Controls:
    - Drag sliders to adjust parameters
    - Parameters update at the start of each new episode
    - Close the window or press ESC to quit
"""

import gymnasium
import highway_env
import torch
import pygame
from stable_baselines3 import DQN, PPO
import os

# ============================================================================
# CONFIGURATION - Modify these paths to match your trained models
# ============================================================================

# Model configuration - set the algorithm and path for your trained model
MODEL_TYPE = "DQN"  # "DQN" or "PPO"
MODEL_PATH = "highway_dqn/model"  # Path to your saved model

# ============================================================================

# Modern color palette
class Colors:
    # Base colors
    BG_DARK = (15, 17, 23)
    BG_MEDIUM = (22, 27, 34)
    BG_LIGHT = (33, 38, 45)
    BG_CARD = (27, 32, 40)
    
    # Text colors
    TEXT_PRIMARY = (240, 246, 252)
    TEXT_SECONDARY = (139, 148, 158)
    TEXT_MUTED = (89, 98, 108)
    
    # Accent colors
    ACCENT_BLUE = (56, 139, 253)
    ACCENT_CYAN = (63, 185, 207)
    ACCENT_GREEN = (63, 185, 80)
    ACCENT_YELLOW = (210, 153, 34)
    ACCENT_ORANGE = (219, 109, 40)
    ACCENT_RED = (248, 81, 73)
    ACCENT_PURPLE = (163, 113, 247)
    
    # UI elements
    SLIDER_TRACK = (48, 54, 61)
    SLIDER_FILL = (56, 139, 253)
    SLIDER_HANDLE = (255, 255, 255)
    BORDER = (48, 54, 61)
    BORDER_HOVER = (88, 94, 101)
    
    # Status colors
    SUCCESS = (63, 185, 80)
    WARNING = (210, 153, 34)
    DANGER = (248, 81, 73)


def draw_rounded_rect(surface, color, rect, radius, alpha=255):
    """Draw a rounded rectangle with optional transparency"""
    if alpha < 255:
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect.width, rect.height), border_radius=radius)
        surface.blit(s, rect.topleft)
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_gradient_rect(surface, rect, color_top, color_bottom, radius=0):
    """Draw a rectangle with vertical gradient"""
    for i in range(rect.height):
        ratio = i / rect.height
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
        pygame.draw.line(surface, (r, g, b), 
                        (rect.x, rect.y + i), 
                        (rect.x + rect.width - 1, rect.y + i))


def draw_glow(surface, pos, radius, color, intensity=0.3):
    """Draw a soft glow effect"""
    for i in range(radius, 0, -2):
        alpha = int(255 * intensity * (1 - i / radius))
        s = pygame.Surface((i * 2, i * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (i, i), i)
        surface.blit(s, (pos[0] - i, pos[1] - i))


class Slider:
    """A modern draggable slider widget for pygame"""
    
    def __init__(self, x, y, width, min_val, max_val, initial, label, step=1, format_str="{:.0f}", icon=""):
        self.x = x
        self.y = y
        self.width = width
        self.height = 24
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.step = step
        self.format_str = format_str
        self.icon = icon
        self.dragging = False
        self.hovering = False
        self.handle_radius = 9
        
    def get_handle_x(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.x + int(ratio * self.width)
    
    def set_value_from_x(self, mouse_x):
        ratio = (mouse_x - self.x) / self.width
        ratio = max(0, min(1, ratio))
        raw_value = self.min_val + ratio * (self.max_val - self.min_val)
        # Snap to step
        self.value = round(raw_value / self.step) * self.step
        self.value = max(self.min_val, min(self.max_val, self.value))
    
    def check_hover(self, pos):
        mx, my = pos
        self.hovering = (self.x - 10 <= mx <= self.x + self.width + 10 and 
                        self.y - 5 <= my <= self.y + self.height + 5)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            hx = self.get_handle_x()
            # Check if clicking on handle or track
            if (self.x <= mx <= self.x + self.width and 
                self.y - 5 <= my <= self.y + self.height + 5):
                self.dragging = True
                self.set_value_from_x(mx)
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            self.check_hover(event.pos)
            if self.dragging:
                self.set_value_from_x(event.pos[0])
                return True
        return False
    
    def draw(self, surface, font, small_font):
        handle_x = self.get_handle_x()
        
        # Track background with rounded ends
        track_height = 6
        track_y = self.y + (self.height - track_height) // 2
        track_rect = pygame.Rect(self.x, track_y, self.width, track_height)
        pygame.draw.rect(surface, Colors.SLIDER_TRACK, track_rect, border_radius=3)
        
        # Filled portion with gradient effect
        fill_width = handle_x - self.x
        if fill_width > 0:
            fill_rect = pygame.Rect(self.x, track_y, fill_width, track_height)
            pygame.draw.rect(surface, Colors.ACCENT_CYAN, fill_rect, border_radius=3)
        
        # Handle glow when active
        if self.dragging or self.hovering:
            draw_glow(surface, (handle_x, self.y + self.height // 2), 20, Colors.ACCENT_CYAN, 0.4)
        
        # Handle with border
        handle_color = Colors.SLIDER_HANDLE if not self.dragging else Colors.ACCENT_CYAN
        pygame.draw.circle(surface, handle_color, (handle_x, self.y + self.height // 2), self.handle_radius)
        pygame.draw.circle(surface, Colors.ACCENT_CYAN if self.dragging else Colors.BORDER, 
                          (handle_x, self.y + self.height // 2), self.handle_radius, 2)
        
        # Label with icon
        label_text = f"{self.icon} {self.label}" if self.icon else self.label
        label_surf = small_font.render(label_text, True, Colors.TEXT_SECONDARY)
        surface.blit(label_surf, (self.x, self.y - 20))
        
        # Value badge
        value_str = self.format_str.format(self.value)
        value_surf = small_font.render(value_str, True, Colors.TEXT_PRIMARY)
        value_rect = value_surf.get_rect()
        badge_rect = pygame.Rect(self.x + self.width + 8, self.y + 2, value_rect.width + 12, 20)
        pygame.draw.rect(surface, Colors.BG_LIGHT, badge_rect, border_radius=4)
        surface.blit(value_surf, (badge_rect.x + 6, badge_rect.y + 3))


class DropdownButton:
    """A modern button that cycles through options"""
    
    def __init__(self, x, y, width, options, initial_index, label, colors=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = 36
        self.options = options
        self.index = initial_index
        self.label = label
        self.hovering = False
        # Colors for each option
        self.colors = colors or [Colors.ACCENT_GREEN, Colors.ACCENT_BLUE, Colors.ACCENT_RED]
        
    @property
    def value(self):
        return self.options[self.index]
    
    def check_hover(self, pos):
        mx, my = pos
        self.hovering = (self.x <= mx <= self.x + self.width and 
                        self.y <= my <= self.y + self.height)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.check_hover(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if (self.x <= mx <= self.x + self.width and 
                self.y <= my <= self.y + self.height):
                self.index = (self.index + 1) % len(self.options)
                return True
        return False
    
    def draw(self, surface, font, small_font):
        color = self.colors[self.index]
        
        # Button background with subtle gradient
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        
        # Hover glow
        if self.hovering:
            glow_rect = pygame.Rect(self.x - 3, self.y - 3, self.width + 6, self.height + 6)
            draw_rounded_rect(surface, color, glow_rect, 10, 60)
        
        # Button fill
        draw_rounded_rect(surface, Colors.BG_LIGHT, rect, 8)
        
        # Colored left accent bar
        accent_rect = pygame.Rect(self.x, self.y, 4, self.height)
        pygame.draw.rect(surface, color, accent_rect, border_radius=2)
        
        # Border
        border_color = color if self.hovering else Colors.BORDER
        pygame.draw.rect(surface, border_color, rect, 2, border_radius=8)
        
        # Option text with icon
        text = font.render(self.value, True, Colors.TEXT_PRIMARY)
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        surface.blit(text, text_rect)
        
        # Click hint
        hint = small_font.render("click to change", True, Colors.TEXT_MUTED)
        hint_rect = hint.get_rect(right=self.x + self.width - 5, centery=self.y + self.height // 2)
        
        # Label
        label_text = small_font.render(self.label, True, Colors.TEXT_SECONDARY)
        surface.blit(label_text, (self.x, self.y - 20))


class ControlPanel:
    """Modern card-style panel containing all sliders and controls"""
    
    def __init__(self, x, y, width):
        self.x = x
        self.y = y
        self.width = width
        self.height = 260
        
        # Create widgets with icons
        slider_x = x + 20
        slider_width = width - 100
        
        self.lanes_slider = Slider(slider_x, y + 55, slider_width, 2, 6, 4, "Lanes", step=1, icon="🛣️")
        self.vehicles_slider = Slider(slider_x, y + 110, slider_width, 5, 100, 25, "Vehicles", step=5, icon="🚗")
        self.density_slider = Slider(slider_x, y + 165, slider_width, 0.1, 3.0, 1.0, "Density", step=0.1, format_str="{:.1f}", icon="📊")
        self.traffic_button = DropdownButton(slider_x, y + 210, slider_width + 55, 
                                             ["🐢 Defensive", "🚗 Normal", "🏎️ Aggressive"], 1, "Traffic Behavior")
        
        self.widgets = [self.lanes_slider, self.vehicles_slider, self.density_slider, self.traffic_button]
        
    def get_params(self):
        traffic_types = ["Defensive", "IDM", "Aggressive"]
        return {
            'lanes_count': int(self.lanes_slider.value),
            'vehicles_count': int(self.vehicles_slider.value),
            'vehicles_density': self.density_slider.value,
            'vehicle_type': traffic_types[self.traffic_button.index],
        }
    
    def get_vehicle_type_string(self):
        vtype = self.get_params()['vehicle_type']
        if vtype == 'Defensive':
            return "highway_env.vehicle.behavior.DefensiveVehicle"
        elif vtype == 'Aggressive':
            return "highway_env.vehicle.behavior.AggressiveVehicle"
        else:
            return "highway_env.vehicle.behavior.IDMVehicle"
    
    def handle_event(self, event):
        for widget in self.widgets:
            if widget.handle_event(event):
                return True
        return False
    
    def draw(self, surface, font, small_font, title_font):
        # Card shadow
        shadow_rect = pygame.Rect(self.x + 3, self.y + 3, self.width, self.height)
        draw_rounded_rect(surface, (0, 0, 0), shadow_rect, 12, 80)
        
        # Card background
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        draw_rounded_rect(surface, Colors.BG_CARD, panel_rect, 12)
        
        # Top accent line
        accent_rect = pygame.Rect(self.x, self.y, self.width, 4)
        pygame.draw.rect(surface, Colors.ACCENT_CYAN, accent_rect, 
                        border_top_left_radius=12, border_top_right_radius=12)
        
        # Border
        pygame.draw.rect(surface, Colors.BORDER, panel_rect, 1, border_radius=12)
        
        # Title with icon
        title = title_font.render("⚙️  Parameters", True, Colors.TEXT_PRIMARY)
        surface.blit(title, (self.x + 15, self.y + 15))
        
        # Subtitle
        subtitle = small_font.render("Changes apply on next episode", True, Colors.TEXT_MUTED)
        surface.blit(subtitle, (self.x + 15, self.y + 38))
        
        # Draw widgets
        for widget in self.widgets:
            widget.draw(surface, font, small_font)


class StatsPanel:
    """Modern card-style panel showing episode statistics"""
    
    def __init__(self, x, y, width):
        self.x = x
        self.y = y
        self.width = width
        self.height = 130
        self.episode = 0
        self.avg_reward = 0
        self.crash_rate = 0
        self.last_outcome = ""
        
    def update(self, episode, avg_reward, crash_rate, last_outcome):
        self.episode = episode
        self.avg_reward = avg_reward
        self.crash_rate = crash_rate
        self.last_outcome = last_outcome
        
    def draw(self, surface, font, small_font, title_font):
        # Card shadow
        shadow_rect = pygame.Rect(self.x + 3, self.y + 3, self.width, self.height)
        draw_rounded_rect(surface, (0, 0, 0), shadow_rect, 12, 80)
        
        # Card background
        panel_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        draw_rounded_rect(surface, Colors.BG_CARD, panel_rect, 12)
        
        # Top accent line
        accent_rect = pygame.Rect(self.x, self.y, self.width, 4)
        pygame.draw.rect(surface, Colors.ACCENT_PURPLE, accent_rect, 
                        border_top_left_radius=12, border_top_right_radius=12)
        
        # Border
        pygame.draw.rect(surface, Colors.BORDER, panel_rect, 1, border_radius=12)
        
        # Title
        title = title_font.render("📊  Statistics", True, Colors.TEXT_PRIMARY)
        surface.blit(title, (self.x + 15, self.y + 15))
        
        # Stats grid
        y_offset = self.y + 50
        col_width = self.width // 2
        
        stats = [
            ("Episode", str(self.episode), Colors.ACCENT_BLUE),
            ("Avg Reward", f"{self.avg_reward:.1f}", Colors.ACCENT_GREEN),
            ("Crash Rate", f"{self.crash_rate:.1f}%", Colors.ACCENT_ORANGE),
            ("Status", self.last_outcome, Colors.ACCENT_CYAN),
        ]
        
        for i, (label, value, color) in enumerate(stats):
            col = i % 2
            row = i // 2
            x = self.x + 15 + col * col_width
            y = y_offset + row * 38
            
            # Label
            label_surf = small_font.render(label, True, Colors.TEXT_MUTED)
            surface.blit(label_surf, (x, y))
            
            # Value
            value_surf = font.render(value, True, color)
            surface.blit(value_surf, (x, y + 16))


def create_env(control_panel):
    """Create environment with current parameters"""
    params = control_panel.get_params()
    vehicle_type = control_panel.get_vehicle_type_string()
    
    config = {
        "lanes_count": params['lanes_count'],
        "vehicles_count": params['vehicles_count'],
        "vehicles_density": params['vehicles_density'],
        "duration": 60,
        "simulation_frequency": 30,
        "other_vehicles_type": vehicle_type,
        "screen_width": 600,
        "screen_height": 200,
        "offscreen_rendering": True,  # Don't let env manage pygame
    }
    
    env = gymnasium.make(
        "highway-v0",
        config=config,
        render_mode='rgb_array'  # We'll blit this to our custom window
    )
    return env


def main():
    print("="*60)
    print("  Highway-Env Interactive Model Tester")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH + ".zip"):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("\nPlease update MODEL_PATH at the top of this file")
        print("to point to your trained model.\n")
        return
    
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
        print("\nUsing CUDA GPU")
    else:
        device = "cpu"
        print("\nUsing CPU")
    
    # Load model
    print(f"Loading {MODEL_TYPE} model from {MODEL_PATH}...")
    
    if MODEL_TYPE.upper() == "DQN":
        model = DQN.load(MODEL_PATH, device=device)
    elif MODEL_TYPE.upper() == "PPO":
        model = PPO.load(MODEL_PATH, device=device)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    print("Model loaded successfully!")
    print("\nUse the sliders to adjust parameters.")
    print("Changes take effect on the next episode.\n")
    
    # Initialize pygame FIRST, before any environment
    pygame.init()
    pygame.display.init()
    
    # Window dimensions
    env_width, env_height = 600, 200
    panel_width = 300
    window_width = env_width + panel_width
    window_height = max(env_height + 180, 420)
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("🚗 Highway-Env Interactive Tester")
    
    # Fonts
    try:
        font = pygame.font.SysFont('Segoe UI', 15)
        small_font = pygame.font.SysFont('Segoe UI', 12)
        title_font = pygame.font.SysFont('Segoe UI', 16, bold=True)
        big_font = pygame.font.SysFont('Segoe UI', 18, bold=True)
    except:
        font = pygame.font.SysFont('Arial', 15)
        small_font = pygame.font.SysFont('Arial', 12)
        title_font = pygame.font.SysFont('Arial', 16, bold=True)
        big_font = pygame.font.SysFont('Arial', 18, bold=True)
    
    clock = pygame.time.Clock()
    
    # Create UI panels
    control_panel = ControlPanel(env_width + 10, 10, panel_width - 20)
    stats_panel = StatsPanel(env_width + 10, 280, panel_width - 20)
    
    # Statistics
    episode_count = 0
    total_reward = 0
    crashes = 0
    running = True
    env = None
    
    # Main loop
    while running:
        # Create environment with current parameters
        env = create_env(control_panel)
        
        episode_count += 1
        episode_reward = 0
        episode_steps = 0
        
        done = truncated = False
        obs, info = env.reset()
        
        while not (done or truncated) and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                else:
                    control_panel.handle_event(event)
            
            if not running:
                break
            
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Render environment to array
            frame = env.render()
            
            # Clear screen with dark background
            screen.fill(Colors.BG_DARK)
            
            # Draw environment frame with border
            if frame is not None:
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                # Environment container with rounded border effect
                env_rect = pygame.Rect(5, 5, env_width - 10, env_height - 10)
                screen.blit(frame_surface, (0, 0))
            
            # Border around env with glow
            border_rect = pygame.Rect(0, 0, env_width, env_height)
            pygame.draw.rect(screen, Colors.BORDER, border_rect, 2, border_radius=4)
            
            # Info bar below environment with card style
            info_y = env_height + 15
            info_rect = pygame.Rect(10, info_y - 5, env_width - 20, 55)
            draw_rounded_rect(screen, Colors.BG_CARD, info_rect, 8)
            pygame.draw.rect(screen, Colors.BORDER, info_rect, 1, border_radius=8)
            
            # Episode info
            episode_text = big_font.render(f"Episode {episode_count}", True, Colors.TEXT_PRIMARY)
            screen.blit(episode_text, (20, info_y + 2))
            
            # Reward with color based on value
            reward_color = Colors.ACCENT_GREEN if episode_reward > 0 else Colors.ACCENT_RED
            reward_text = font.render(f"Reward: {episode_reward:.1f}", True, reward_color)
            screen.blit(reward_text, (150, info_y + 5))
            
            # Steps
            steps_text = font.render(f"Steps: {episode_steps}", True, Colors.TEXT_SECONDARY)
            screen.blit(steps_text, (280, info_y + 5))
            
            # Current params row
            params = control_panel.get_params()
            param_items = [
                (f"🛣️ {params['lanes_count']}", Colors.ACCENT_CYAN),
                (f"🚗 {params['vehicles_count']}", Colors.ACCENT_BLUE),
                (f"📊 {params['vehicles_density']:.1f}", Colors.ACCENT_PURPLE),
                (f"🎯 {params['vehicle_type']}", Colors.ACCENT_ORANGE),
            ]
            x_offset = 20
            for text, color in param_items:
                param_surf = small_font.render(text, True, color)
                screen.blit(param_surf, (x_offset, info_y + 30))
                x_offset += param_surf.get_width() + 25
            
            # Draw UI panels
            control_panel.draw(screen, font, small_font, title_font)
            stats_panel.draw(screen, font, small_font, title_font)
            
            # Update display
            pygame.display.flip()
            clock.tick(60)  # Target 60 FPS for smooth rendering
        
        # Episode ended - close env but NOT pygame
        if env is not None:
            # Just delete the env, don't call close() which may affect pygame
            del env
            env = None
        
        if not running:
            break
            
        # Update stats
        crashed = done and not truncated
        if crashed:
            crashes += 1
        
        total_reward += episode_reward
        avg_reward = total_reward / episode_count
        crash_rate = 100 * crashes / episode_count
        outcome = "💥 CRASH" if crashed else "✅ SURVIVED"
        
        stats_panel.update(episode_count, avg_reward, crash_rate, outcome)
        
        # Print to console too
        print(f"Episode {episode_count}: {outcome} | Reward: {episode_reward:.1f} | Crash Rate: {crash_rate:.1f}%")
        
        # Brief pause before next episode - keep processing events
        pause_start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - pause_start < 500 and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    control_panel.handle_event(event)
            
            # Redraw during pause with nice transition screen
            screen.fill(Colors.BG_DARK)
            
            # Centered message card
            msg_width, msg_height = 300, 80
            msg_x = (env_width - msg_width) // 2
            msg_y = (env_height - msg_height) // 2
            msg_rect = pygame.Rect(msg_x, msg_y, msg_width, msg_height)
            
            draw_rounded_rect(screen, Colors.BG_CARD, msg_rect, 12)
            pygame.draw.rect(screen, Colors.ACCENT_CYAN, msg_rect, 2, border_radius=12)
            
            # Loading text
            msg = title_font.render("🔄 Starting Next Episode...", True, Colors.TEXT_PRIMARY)
            msg_rect_text = msg.get_rect(center=(env_width // 2, env_height // 2 - 10))
            screen.blit(msg, msg_rect_text)
            
            # Subtitle
            sub = small_font.render("Adjusting parameters", True, Colors.TEXT_MUTED)
            sub_rect = sub.get_rect(center=(env_width // 2, env_height // 2 + 15))
            screen.blit(sub, sub_rect)
            
            # Border around env area
            pygame.draw.rect(screen, Colors.BORDER, (0, 0, env_width, env_height), 2, border_radius=4)
            
            control_panel.draw(screen, font, small_font, title_font)
            stats_panel.draw(screen, font, small_font, title_font)
            pygame.display.flip()
            clock.tick(30)
    
    pygame.quit()
    
    # Final stats
    print("\n" + "="*60)
    print("  TESTING COMPLETE")
    print("="*60)
    print(f"  Total Episodes: {episode_count}")
    print(f"  Average Reward: {total_reward/max(1,episode_count):.2f}")
    print(f"  Crash Rate: {100*crashes/max(1,episode_count):.1f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
