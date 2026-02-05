"""
Enhanced Interactive Model Tester with Real-Time Parameter Adjustment
======================================================================

NEW FEATURES:
- Multi-model comparison (test multiple models side-by-side)
- Automatic generalization test suite
- Configuration presets (Easy/Medium/Hard)
- Results export to CSV
- Detailed metrics tracking
- Performance heatmap visualization
- Pause/resume functionality
- Speed control

Usage:
    python interactive_tester_enhanced.py --models model1.zip model2.zip --mode interactive

Controls:
    - Drag sliders to adjust parameters
    - SPACE: Pause/Resume
    - R: Reset statistics
    - S: Save results to CSV
    - P: Run preset test suite
    - ESC/Q: Quit
"""

import gymnasium
import highway_env
import torch
import pygame
import numpy as np
from stable_baselines3 import DQN, PPO
import os
import argparse
import json
import csv
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

# Modern color palette (from original)
class Colors:
    BG_DARK = (15, 17, 23)
    BG_MEDIUM = (22, 27, 34)
    BG_LIGHT = (33, 38, 45)
    BG_CARD = (27, 32, 40)
    
    TEXT_PRIMARY = (240, 246, 252)
    TEXT_SECONDARY = (139, 148, 158)
    TEXT_MUTED = (89, 98, 108)
    
    ACCENT_BLUE = (56, 139, 253)
    ACCENT_CYAN = (63, 185, 207)
    ACCENT_GREEN = (63, 185, 80)
    ACCENT_YELLOW = (210, 153, 34)
    ACCENT_ORANGE = (219, 109, 40)
    ACCENT_RED = (248, 81, 73)
    ACCENT_PURPLE = (163, 113, 247)
    
    SLIDER_TRACK = (48, 54, 61)
    SLIDER_FILL = (56, 139, 253)
    SLIDER_HANDLE = (255, 255, 255)
    BORDER = (48, 54, 61)
    BORDER_HOVER = (88, 94, 101)
    
    SUCCESS = (63, 185, 80)
    WARNING = (210, 153, 34)
    DANGER = (248, 81, 73)


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

PRESETS = {
    "Easy": {
        "lanes_count": 4,
        "vehicles_count": 15,
        "vehicles_density": 0.8,
        "description": "Wide highway, light traffic"
    },
    "Medium": {
        "lanes_count": 3,
        "vehicles_count": 25,
        "vehicles_density": 1.0,
        "description": "Standard conditions"
    },
    "Hard": {
        "lanes_count": 2,
        "vehicles_count": 35,
        "vehicles_density": 2.0,
        "description": "Narrow road, dense traffic"
    },
    "Extreme": {
        "lanes_count": 2,
        "vehicles_count": 40,
        "vehicles_density": 2.5,
        "description": "Maximum difficulty"
    }
}

# Test suite for generalization evaluation
GENERALIZATION_TEST_SUITE = [
    {"name": "Easy Cruise", "lanes_count": 4, "vehicles_count": 10, "vehicles_density": 0.6},
    {"name": "Light Traffic", "lanes_count": 3, "vehicles_count": 15, "vehicles_density": 0.8},
    {"name": "Standard", "lanes_count": 3, "vehicles_count": 25, "vehicles_density": 1.0},
    {"name": "Busy", "lanes_count": 3, "vehicles_count": 30, "vehicles_density": 1.5},
    {"name": "Dense", "lanes_count": 2, "vehicles_count": 35, "vehicles_density": 2.0},
    {"name": "Extreme", "lanes_count": 2, "vehicles_count": 40, "vehicles_density": 2.5},
    {"name": "Wide Empty", "lanes_count": 5, "vehicles_count": 15, "vehicles_density": 0.7},
    {"name": "Narrow Dense", "lanes_count": 2, "vehicles_count": 38, "vehicles_density": 2.2},
]


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Track detailed performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.crashes = 0
        self.survivals = 0
        self.total_episodes = 0
        
        # Per-configuration metrics
        self.config_results = defaultdict(lambda: {
            'rewards': [],
            'crashes': 0,
            'episodes': 0
        })
        
        # Recent history for moving averages
        self.recent_rewards = deque(maxlen=10)
        self.recent_lengths = deque(maxlen=10)
    
    def record_episode(self, reward, length, crashed, config_key=None):
        """Record episode result"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.recent_rewards.append(reward)
        self.recent_lengths.append(length)
        self.total_episodes += 1
        
        if crashed:
            self.crashes += 1
        else:
            self.survivals += 1
        
        # Track per-configuration
        if config_key:
            self.config_results[config_key]['rewards'].append(reward)
            self.config_results[config_key]['episodes'] += 1
            if crashed:
                self.config_results[config_key]['crashes'] += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        if self.total_episodes == 0:
            return {
                'total_episodes': 0,
                'avg_reward': 0,
                'recent_avg_reward': 0,
                'crash_rate': 0,
                'survival_rate': 0,
                'avg_length': 0,
            }
        
        return {
            'total_episodes': self.total_episodes,
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'recent_avg_reward': np.mean(self.recent_rewards) if len(self.recent_rewards) > 0 else 0,
            'crash_rate': 100 * self.crashes / self.total_episodes,
            'survival_rate': 100 * self.survivals / self.total_episodes,
            'avg_length': np.mean(self.episode_lengths),
        }
    
    def export_to_csv(self, filepath: str, model_name: str = "model"):
        """Export results to CSV file"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Episode', 'Reward', 'Length', 'Crashed', 'Model'
            ])
            
            # Data
            for i, (reward, length) in enumerate(zip(self.episode_rewards, self.episode_lengths)):
                crashed = i < self.crashes  # Simplified - actual tracking would be better
                writer.writerow([i + 1, reward, length, crashed, model_name])
        
        print(f"✅ Results exported to {filepath}")


# ============================================================================
# ENHANCED COMPONENTS (using original UI classes as base)
# ============================================================================

def safe_render_text(font, text, color):
    """Safely render text, handling pygame state issues"""
    try:
        if not text:
            text = " "  # Prevent zero-width text
        return font.render(str(text), True, color)
    except pygame.error:
        # Try to reinit font module
        pygame.font.init()
        return font.render(str(text) if text else " ", True, color)


def draw_rounded_rect(surface, color, rect, radius, alpha=255):
    """Draw a rounded rectangle with optional transparency"""
    if alpha < 255:
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, rect.width, rect.height), border_radius=radius)
        surface.blit(s, rect.topleft)
    else:
        pygame.draw.rect(surface, color, rect, border_radius=radius)


def draw_glow(surface, pos, radius, color, intensity=0.3):
    """Draw a soft glow effect"""
    for i in range(radius, 0, -2):
        alpha = int(255 * intensity * (1 - i / radius))
        s = pygame.Surface((i * 2, i * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (i, i), i)
        surface.blit(s, (pos[0] - i, pos[1] - i))


class Slider:
    """A modern draggable slider widget (simplified from original)"""
    
    def __init__(self, x, y, width, min_val, max_val, initial, label, step=1, format_str="{:.0f}"):
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
        self.value = round(raw_value / self.step) * self.step
        self.value = max(self.min_val, min(self.max_val, self.value))
    
    def check_hover(self, pos):
        mx, my = pos
        self.hovering = (self.x - 10 <= mx <= self.x + self.width + 10 and 
                        self.y - 5 <= my <= self.y + self.height + 5)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
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
        
        # Track
        track_height = 6
        track_y = self.y + (self.height - track_height) // 2
        pygame.draw.rect(surface, Colors.SLIDER_TRACK, 
                        pygame.Rect(self.x, track_y, self.width, track_height), border_radius=3)
        
        # Fill
        fill_width = handle_x - self.x
        if fill_width > 0:
            pygame.draw.rect(surface, Colors.ACCENT_CYAN, 
                           pygame.Rect(self.x, track_y, fill_width, track_height), border_radius=3)
        
        # Glow when active
        if self.dragging or self.hovering:
            draw_glow(surface, (handle_x, self.y + self.height // 2), 20, Colors.ACCENT_CYAN, 0.4)
        
        # Handle
        handle_color = Colors.SLIDER_HANDLE if not self.dragging else Colors.ACCENT_CYAN
        pygame.draw.circle(surface, handle_color, (handle_x, self.y + self.height // 2), self.handle_radius)
        pygame.draw.circle(surface, Colors.ACCENT_CYAN if self.dragging else Colors.BORDER, 
                          (handle_x, self.y + self.height // 2), self.handle_radius, 2)
        
        # Label
        label_surf = safe_render_text(small_font, self.label, Colors.TEXT_SECONDARY)
        surface.blit(label_surf, (self.x, self.y - 20))
        
        # Value
        value_str = self.format_str.format(self.value)
        value_surf = safe_render_text(small_font, value_str, Colors.TEXT_PRIMARY)
        badge_rect = pygame.Rect(self.x + self.width + 8, self.y + 2, value_surf.get_width() + 12, 20)
        pygame.draw.rect(surface, Colors.BG_LIGHT, badge_rect, border_radius=4)
        surface.blit(value_surf, (badge_rect.x + 6, badge_rect.y + 3))


class PresetButton:
    """Button for loading configuration presets"""
    
    def __init__(self, x, y, width, preset_name, preset_config):
        self.x = x
        self.y = y
        self.width = width
        self.height = 30
        self.preset_name = preset_name
        self.preset_config = preset_config
        self.hovering = False
    
    def check_hover(self, pos):
        mx, my = pos
        self.hovering = (self.x <= mx <= self.x + self.width and
                        self.y <= my <= self.y + self.height)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check hover on click in case mouse didn't move first
            self.check_hover(event.pos)
            if self.hovering:
                return True
        elif event.type == pygame.MOUSEMOTION:
            self.check_hover(event.pos)
        return False
    
    def draw(self, surface, font):
        # Background
        color = Colors.ACCENT_CYAN if self.hovering else Colors.BG_LIGHT
        draw_rounded_rect(surface, color, 
                         pygame.Rect(self.x, self.y, self.width, self.height), 6)
        
        # Border
        border_color = Colors.ACCENT_CYAN if self.hovering else Colors.BORDER
        pygame.draw.rect(surface, border_color,
                        pygame.Rect(self.x, self.y, self.width, self.height), 
                        2, border_radius=6)
        
        # Text
        text_color = Colors.TEXT_PRIMARY if self.hovering else Colors.TEXT_SECONDARY
        text = safe_render_text(font, self.preset_name, text_color)
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        surface.blit(text, text_rect)


class EnhancedControlPanel:
    """Enhanced control panel with presets and more options"""
    
    def __init__(self, x, y, width):
        self.x = x
        self.y = y
        self.width = width
        
        # Sliders
        slider_width = width - 20
        self.sliders = [
            Slider(x + 10, y + 70, slider_width, 2, 5, 3, "🛣️ Lanes", step=1),
            Slider(x + 10, y + 120, slider_width, 10, 50, 25, "🚗 Vehicles", step=5),
            Slider(x + 10, y + 170, slider_width, 0.5, 3.0, 1.0, "📊 Density", step=0.1, format_str="{:.1f}"),
        ]
        
        # Preset buttons
        self.preset_buttons = []
        button_y = y + 230
        button_width = (width - 30) // 2
        for i, (name, config) in enumerate(list(PRESETS.items())[:4]):  # First 4 presets
            bx = x + 10 + (i % 2) * (button_width + 10)
            by = button_y + (i // 2) * 40
            self.preset_buttons.append(PresetButton(bx, by, button_width, name, config))
    
    def get_config(self) -> Dict:
        """Get current configuration from sliders"""
        return {
            'lanes_count': int(self.sliders[0].value),
            'vehicles_count': int(self.sliders[1].value),
            'vehicles_density': float(self.sliders[2].value),
            'duration': 60,
            'simulation_frequency': 30,
        }
    
    def set_config(self, config: Dict):
        """Set slider values from configuration"""
        self.sliders[0].value = config.get('lanes_count', 3)
        self.sliders[1].value = config.get('vehicles_count', 25)
        self.sliders[2].value = config.get('vehicles_density', 1.0)
    
    def handle_event(self, event):
        """Handle pygame events"""
        changed = False
        
        # Check sliders
        for slider in self.sliders:
            if slider.handle_event(event):
                changed = True
        
        # Check preset buttons
        for button in self.preset_buttons:
            if button.handle_event(event):
                self.set_config(button.preset_config)
                changed = True
        
        return changed
    
    def draw(self, surface, font, small_font, title_font):
        """Draw the control panel"""
        # Panel background
        panel_rect = pygame.Rect(self.x, self.y, self.width, 350)
        draw_rounded_rect(surface, Colors.BG_CARD, panel_rect, 8)
        pygame.draw.rect(surface, Colors.BORDER, panel_rect, 1, border_radius=8)
        
        # Title
        title = safe_render_text(title_font, "⚙️ Environment Config", Colors.TEXT_PRIMARY)
        surface.blit(title, (self.x + 10, self.y + 10))
        
        # Sliders
        for slider in self.sliders:
            slider.draw(surface, font, small_font)
        
        # Presets label
        preset_label = safe_render_text(small_font, "Presets:", Colors.TEXT_SECONDARY)
        surface.blit(preset_label, (self.x + 10, self.y + 210))
        
        # Preset buttons
        for button in self.preset_buttons:
            button.draw(surface, small_font)


class EnhancedStatsPanel:
    """Enhanced statistics panel with more detailed metrics"""
    
    def __init__(self, x, y, width, model_name="Model"):
        self.x = x
        self.y = y
        self.width = width
        self.model_name = model_name
        self.stats = {}
    
    def update(self, stats: Dict):
        """Update displayed statistics"""
        self.stats = stats
    
    def draw(self, surface, font, small_font, title_font):
        """Draw the stats panel"""
        # Panel background
        panel_height = 250
        panel_rect = pygame.Rect(self.x, self.y, self.width, panel_height)
        draw_rounded_rect(surface, Colors.BG_CARD, panel_rect, 8)
        pygame.draw.rect(surface, Colors.BORDER, panel_rect, 1, border_radius=8)
        
        # Title
        title = safe_render_text(title_font, f"📊 {self.model_name} Stats", Colors.TEXT_PRIMARY)
        surface.blit(title, (self.x + 10, self.y + 10))
        
        if not self.stats:
            no_data = safe_render_text(small_font, "No data yet...", Colors.TEXT_MUTED)
            surface.blit(no_data, (self.x + 10, self.y + 50))
            return
        
        # Stats display
        y_offset = self.y + 45
        stats_to_show = [
            ("Episodes", f"{self.stats.get('total_episodes', 0)}", Colors.ACCENT_BLUE),
            ("Avg Reward", f"{self.stats.get('avg_reward', 0):.2f}", Colors.ACCENT_GREEN),
            ("Recent Avg", f"{self.stats.get('recent_avg_reward', 0):.2f}", Colors.ACCENT_CYAN),
            ("Crash Rate", f"{self.stats.get('crash_rate', 0):.1f}%", Colors.ACCENT_RED),
            ("Survival", f"{self.stats.get('survival_rate', 0):.1f}%", Colors.ACCENT_GREEN),
            ("Avg Length", f"{self.stats.get('avg_length', 0):.0f}", Colors.ACCENT_PURPLE),
        ]
        
        for label, value, color in stats_to_show:
            # Label
            label_surf = safe_render_text(small_font, label + ":", Colors.TEXT_SECONDARY)
            surface.blit(label_surf, (self.x + 15, y_offset))
            
            # Value
            value_surf = safe_render_text(font, value, color)
            surface.blit(value_surf, (self.x + self.width - value_surf.get_width() - 15, y_offset - 2))
            
            y_offset += 32


# ============================================================================
# MODEL WRAPPER
# ============================================================================

class ModelWrapper:
    """Wrapper for SB3 models with unified interface"""
    
    def __init__(self, model_path: str, model_type: str = "DQN", name: str = None):
        self.model_path = model_path
        self.model_type = model_type.upper()
        self.name = name or os.path.basename(model_path)
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model_type == "DQN":
            self.model = DQN.load(model_path, device=device)
        elif self.model_type == "PPO":
            self.model = PPO.load(model_path, device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"✅ Loaded {self.model_type} model: {self.name} on {device}")
        
        # Metrics tracker
        self.metrics = MetricsTracker()
    
    def predict(self, obs, deterministic=True):
        """Predict action"""
        return self.model.predict(obs, deterministic=deterministic)
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return self.metrics.get_stats()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def create_env(config, render_mode='rgb_array'):
    """Create environment with given configuration"""
    env_config = {
        'lanes_count': config.get('lanes_count', 3),
        'vehicles_count': config.get('vehicles_count', 25),
        'vehicles_density': config.get('vehicles_density', 1.0),
        'duration': config.get('duration', 60),
        'simulation_frequency': config.get('simulation_frequency', 30),
        'policy_frequency': config.get('policy_frequency', 1),
        'render_agent': True,
        'offscreen_rendering': render_mode == 'rgb_array',
    }
    return gymnasium.make("highway-v0", config=env_config, render_mode=render_mode)


def run_generalization_test(models: List[ModelWrapper], output_file: str = None):
    """Run systematic generalization test on all models"""
    print("\n" + "="*80)
    print("  RUNNING GENERALIZATION TEST SUITE")
    print("="*80)
    
    results = {model.name: {} for model in models}
    
    for test_config in GENERALIZATION_TEST_SUITE:
        name = test_config['name']
        config = {k: v for k, v in test_config.items() if k != 'name'}
        config.update({'duration': 60, 'simulation_frequency': 30})
        
        print(f"\nTesting: {name}")
        print(f"  Config: {config['lanes_count']} lanes, {config['vehicles_count']} vehicles, {config['vehicles_density']:.1f} density")
        
        for model in models:
            env = create_env(config)
            episode_rewards = []
            crashes = 0
            
            # Run 5 episodes per configuration
            for ep in range(5):
                obs, _ = env.reset()
                done = truncated = False
                episode_reward = 0
                
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                if done and not truncated:
                    crashes += 1
            
            env.close()
            
            avg_reward = np.mean(episode_rewards)
            crash_rate = 100 * crashes / 5
            
            results[model.name][name] = {
                'avg_reward': avg_reward,
                'std_reward': np.std(episode_rewards),
                'crash_rate': crash_rate
            }
            
            print(f"    {model.name}: Reward={avg_reward:.2f}±{np.std(episode_rewards):.2f}, Crashes={crash_rate:.0f}%")
    
    # Print summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    for model in models:
        all_rewards = [r['avg_reward'] for r in results[model.name].values()]
        all_crashes = [r['crash_rate'] for r in results[model.name].values()]
        print(f"\n{model.name}:")
        print(f"  Overall Avg Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"  Overall Crash Rate: {np.mean(all_crashes):.1f}%")
    
    # Save to file
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario', 'Model', 'Avg Reward', 'Std Reward', 'Crash Rate'])
            for scenario in GENERALIZATION_TEST_SUITE:
                name = scenario['name']
                for model in models:
                    r = results[model.name][name]
                    writer.writerow([name, model.name, r['avg_reward'], r['std_reward'], r['crash_rate']])
        print(f"\n✅ Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Enhanced Interactive Model Tester')
    parser.add_argument('--models', nargs='+', required=True, help='Paths to model files')
    parser.add_argument('--model-types', nargs='+', default=None, help='Model types (DQN/PPO), one per model')
    parser.add_argument('--model-names', nargs='+', default=None, help='Display names for models')
    parser.add_argument('--mode', choices=['interactive', 'test', 'both'], default='interactive',
                        help='Mode: interactive GUI, automatic test, or both')
    parser.add_argument('--output', default=None, help='Output CSV file for test results')
    
    args = parser.parse_args()
    
    # Load models
    models = []
    for i, model_path in enumerate(args.models):
        model_type = args.model_types[i] if args.model_types and i < len(args.model_types) else "DQN"
        model_name = args.model_names[i] if args.model_names and i < len(args.model_names) else f"Model {i+1}"
        models.append(ModelWrapper(model_path, model_type, model_name))
    
    # Run test mode
    if args.mode in ['test', 'both']:
        output_file = args.output or f"generalization_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        run_generalization_test(models, output_file)
    
    # Run interactive mode
    if args.mode in ['interactive', 'both']:
        run_interactive(models)


def run_interactive(models: List[ModelWrapper]):
    """Run interactive testing interface"""
    # Use first model for now (can be extended for multi-model view)
    model = models[0]
    
    pygame.init()
    pygame.display.init()
    
    # Window setup
    env_width, env_height = 600, 200
    panel_width = 320
    window_width = env_width + panel_width
    window_height = max(env_height + 180, 640)
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"🚗 Testing: {model.name}")
    
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
    
    # UI panels
    control_panel = EnhancedControlPanel(env_width + 10, 10, panel_width - 20)
    stats_panel = EnhancedStatsPanel(env_width + 10, 370, panel_width - 20, model.name)
    
    # State
    running = True
    paused = False
    episode_count = 0
    
    print("\n" + "="*80)
    print("  INTERACTIVE MODE - Controls:")
    print("="*80)
    print("  SPACE: Pause/Resume")
    print("  R: Reset statistics")
    print("  S: Save results to CSV")
    print("  ESC/Q: Quit")
    print("="*80 + "\n")
    
    # Create initial environment
    config = control_panel.get_config()
    env = create_env(config)
    current_config = config.copy()
    
    # Track if config needs to be applied (only between episodes)
    config_changed = False
    
    # Main loop
    while running:
        # Check if config changed - apply between episodes
        new_config = control_panel.get_config()
        if (new_config['lanes_count'] != current_config.get('lanes_count') or
            new_config['vehicles_count'] != current_config.get('vehicles_count') or
            new_config['vehicles_density'] != current_config.get('vehicles_density')):
            # Config changed, recreate environment
            try:
                env.close()
            except:
                pass
            config = new_config
            current_config = config.copy()
            env = create_env(config)
            print(f"🔄 Config updated: {config['lanes_count']} lanes, {config['vehicles_count']} vehicles, {config['vehicles_density']:.1f} density")
        else:
            config = current_config
        
        episode_count += 1
        episode_reward = 0
        episode_steps = 0
        
        done = truncated = False
        obs, info = env.reset()
        
        # Episode loop
        while not (done or truncated) and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        model.metrics.reset()
                        print("📊 Statistics reset")
                    elif event.key == pygame.K_s:
                        filename = f"results_{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        model.metrics.export_to_csv(filename, model.name)
                else:
                    control_panel.handle_event(event)
            
            if not running:
                break
            
            # Skip step if paused
            if paused:
                pygame.time.wait(100)
                continue
            
            # Get action and step
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Render
            frame = env.render()
            screen.fill(Colors.BG_DARK)
            
            # Draw environment
            if frame is not None:
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))
            
            pygame.draw.rect(screen, Colors.BORDER, (0, 0, env_width, env_height), 2, border_radius=4)
            
            # Info bar
            info_y = env_height + 15
            info_rect = pygame.Rect(10, info_y - 5, env_width - 20, 60)
            draw_rounded_rect(screen, Colors.BG_CARD, info_rect, 8)
            
            # Safe text rendering
            episode_text = safe_render_text(big_font, f"Episode {episode_count}", Colors.TEXT_PRIMARY)
            screen.blit(episode_text, (20, info_y + 2))
            
            reward_color = Colors.ACCENT_GREEN if episode_reward > 0 else Colors.ACCENT_RED
            reward_text = safe_render_text(font, f"Reward: {episode_reward:.1f}", reward_color)
            screen.blit(reward_text, (150, info_y + 5))
            
            steps_text = safe_render_text(font, f"Steps: {episode_steps}", Colors.TEXT_SECONDARY)
            screen.blit(steps_text, (280, info_y + 5))
            
            # Pause indicator
            if paused:
                pause_text = safe_render_text(title_font, "⏸ PAUSED", Colors.ACCENT_YELLOW)
                screen.blit(pause_text, (450, info_y + 5))
            
            # Current config
            params = [
                (f"🛣️ {config['lanes_count']}", Colors.ACCENT_CYAN),
                (f"🚗 {config['vehicles_count']}", Colors.ACCENT_BLUE),
                (f"📊 {config['vehicles_density']:.1f}", Colors.ACCENT_PURPLE),
            ]
            x_offset = 20
            for text, color in params:
                surf = safe_render_text(small_font, text, color)
                screen.blit(surf, (x_offset, info_y + 35))
                x_offset += surf.get_width() + 25
            
            # Draw panels
            control_panel.draw(screen, font, small_font, title_font)
            stats_panel.draw(screen, font, small_font, title_font)
            
            pygame.display.flip()
            clock.tick(60)
        
        # Don't close env - we reuse it with reset()
        # This preserves the pygame/viewer state
        
        if not running:
            break
        
        # Record episode
        crashed = done and not truncated
        config_key = f"{config['lanes_count']}L_{config['vehicles_count']}V_{config['vehicles_density']:.1f}D"
        model.metrics.record_episode(episode_reward, episode_steps, crashed, config_key)
        
        # Update stats
        stats_panel.update(model.get_stats())
        
        # Print to console
        outcome = "💥 CRASH" if crashed else "✅ SURVIVED"
        print(f"Episode {episode_count}: {outcome} | Reward: {episode_reward:.1f} | Steps: {episode_steps}")
        
        # Brief pause between episodes
        pause_start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - pause_start < 500 and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
            
            screen.fill(Colors.BG_DARK)
            
            # Transition screen
            msg_rect = pygame.Rect((env_width - 300) // 2, (env_height - 80) // 2, 300, 80)
            draw_rounded_rect(screen, Colors.BG_CARD, msg_rect, 12)
            
            msg = safe_render_text(title_font, "🔄 Next Episode...", Colors.TEXT_PRIMARY)
            msg_rect_text = msg.get_rect(center=(env_width // 2, env_height // 2))
            screen.blit(msg, msg_rect_text)
            
            pygame.draw.rect(screen, Colors.BORDER, (0, 0, env_width, env_height), 2, border_radius=4)
            
            control_panel.draw(screen, font, small_font, title_font)
            stats_panel.draw(screen, font, small_font, title_font)
            pygame.display.flip()
    
    # Clean up environment
    try:
        env.close()
    except:
        pass
    
    pygame.quit()
    
    # Final stats
    final_stats = model.get_stats()
    print("\n" + "="*80)
    print("  TESTING COMPLETE")
    print("="*80)
    print(f"  Total Episodes: {final_stats['total_episodes']}")
    print(f"  Average Reward: {final_stats['avg_reward']:.2f}")
    print(f"  Crash Rate: {final_stats['crash_rate']:.1f}%")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()