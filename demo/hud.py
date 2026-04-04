import pygame
import os

class HUD:
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        pygame.font.init()
        # We use Consolas because it is a rigid monospace font installed on Windows
        self.font_title = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_reg = pygame.font.SysFont("consolas", 14)
        self.font_small = pygame.font.SysFont("consolas", 11)
        
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        try:
            self.icon_batt = pygame.image.load(os.path.join(asset_dir, "battery_icon.png")).convert_alpha()
            self.icon_batt = pygame.transform.smoothscale(self.icon_batt, (20, 20))
            
            self.icon_mem = pygame.image.load(os.path.join(asset_dir, "memory_icon.png")).convert_alpha()
            self.icon_mem = pygame.transform.smoothscale(self.icon_mem, (20, 20))
            
            self.icon_sig = pygame.image.load(os.path.join(asset_dir, "signal_icon.png")).convert_alpha()
            self.icon_sig = pygame.transform.smoothscale(self.icon_sig, (20, 20))
        except:
            self.icon_batt = pygame.Surface((20, 20))
            self.icon_mem = pygame.Surface((20, 20))
            self.icon_sig = pygame.Surface((20, 20))
            
        # Action history filter
        self.last_action_str = "WAITING"
        self.action_hold_frames = 0
        
        self.last_mode_str = "N/A"
        self.mode_hold_frames = 0
            
    def draw_text(self, surface, text, font, color, x, y):
        render = font.render(text, True, color)
        surface.blit(render, (x, y))

    def draw_bar(self, surface, x, y, width, height, pct, color):
        pygame.draw.rect(surface, (40, 40, 50), (x, y, width, height))
        fill_width = int(width * (pct / 100.0))
        if fill_width > 0:
            pygame.draw.rect(surface, color, (x, y, fill_width, height))
        pygame.draw.rect(surface, (150, 150, 150), (x, y, width, height), 2)

    def draw(self, surface, state_debug):
        # Transparent HUD Panel (Square bottom-left)
        panel_width = 300
        panel_height = 360
        start_y = self.height - panel_height - 20
        
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((15, 20, 35, 210))
        surface.blit(panel, (20, start_y))
        
        x = 35
        y = start_y + 20
        
        self.draw_text(surface, "SYSTEM OVERVIEW", self.font_title, (255, 255, 255), x, y)
        pygame.draw.line(surface, (100, 100, 100), (x, y+28), (x+250, y+28), 2)
        y += 40
        
        mode_raw = str(state_debug.get('spacecraft_mode', 'N/A'))
        if mode_raw.startswith("SpacecraftMode."): 
            mode_raw = mode_raw.split(".")[-1] # Clean enum string
            
        if mode_raw != self.last_mode_str and self.mode_hold_frames <= 0:
            self.last_mode_str = mode_raw
            self.mode_hold_frames = 30
            
        if self.mode_hold_frames > 0:
            self.mode_hold_frames -= 1
            
        self.draw_text(surface, f"MODE:{self.last_mode_str:12s}", self.font_reg, (100, 220, 255), x, y)
        y += 22
        sim_time = state_debug.get('sim_time_min', 0)
        # Pad width to 6 chars so string length never changes
        self.draw_text(surface, f"SIM TIME: {sim_time:6.1f} min", self.font_reg, (200, 200, 200), x, y)
        y += 35
        
        # Battery Block
        surface.blit(self.icon_batt, (x, y-2))
        batt_pct = state_debug.get('battery_soc_pct', 0)
        c_batt = (50, 255, 50) if batt_pct > 60 else (255, 200, 50) if batt_pct > 25 else (255, 50, 50)
        # Pad to 5 chars length
        self.draw_text(surface, f"BATTERY:{batt_pct:6.1f}%", self.font_reg, (255, 255, 255), x+30, y)
        y += 22
        self.draw_bar(surface, x, y, 220, 12, batt_pct, c_batt)
        y += 35
        
        # Storage Block
        surface.blit(self.icon_mem, (x, y-2))
        mem_used = state_debug.get('downlink_queue_mb', 0)
        # Scale to max roughly 2000 for visual sake
        mem_pct = min(100.0, max(0.0, (mem_used / 2000.0) * 100))
        # Pad to 6 chars length
        self.draw_text(surface, f"STORAGE:{mem_used:7.1f} MB", self.font_reg, (255, 255, 255), x+30, y)
        y += 22
        self.draw_bar(surface, x, y, 220, 12, mem_pct, (50, 150, 255))
        y += 35
        
        # Signals Block
        surface.blit(self.icon_sig, (x, y))
        link = state_debug.get('link_quality', 'NONE')
        c_link = (100, 255, 100) if link != "NONE" else (150, 150, 150)
        self.draw_text(surface, f"COMMS: {link}", self.font_reg, c_link, x+30, y)
        y += 40
        
        # Actions
        pygame.draw.line(surface, (100, 100, 100), (x, y), (x+250, y), 2)
        y += 15
        self.draw_text(surface, "LAST ISSUED COMMAND:", self.font_small, (150, 150, 150), x, y)
        y += 20
        action_raw = str(state_debug.get('last_action', 'WAITING'))
        if action_raw.startswith("Action."): 
            action_raw = action_raw.split(".")[-1]
            
        # Temporal hold filter
        if action_raw != self.last_action_str and self.action_hold_frames <= 0:
            self.last_action_str = action_raw
            self.action_hold_frames = 30
            
        if self.action_hold_frames > 0:
            self.action_hold_frames -= 1
            
        self.draw_text(surface, f"> {self.last_action_str:15s}", self.font_reg, (255, 200, 50), x, y)
