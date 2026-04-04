import pygame
import math
import os
import random

class Renderer:
    def __init__(self, screen_width, screen_height):
        self.width = screen_width
        self.height = screen_height
        self.center_x = screen_width // 2
        self.center_y = screen_height // 2
        
        # Distances
        self.earth_orbit_radius = self.width * 0.36 # massive sweeping orbit
        self.cubesat_orbit_radius = 115 # distance from earth center
        
        # Create Deep Space Background with Stars
        self.bg_surface = pygame.Surface((self.width, self.height))
        self.bg_surface.fill((3, 5, 12)) # Darkest deep space blue
        for _ in range(350): # 350 random stars
            sx = random.randint(0, self.width)
            sy = random.randint(0, self.height)
            size = random.choice([1, 1, 1, 2]) # Mostly tiny, some slightly larger
            brightness = random.randint(100, 255)
            # Mix pure white, glowing blue, and pale yellow stars
            color = random.choice([
                (brightness, brightness, brightness),
                (brightness, brightness, brightness),
                (max(0, brightness-40), max(0, brightness-40), brightness), # Pale blue
                (brightness, brightness, max(0, brightness-60)),            # Pale yellow
            ])
            pygame.draw.circle(self.bg_surface, color, (sx, sy), size)
            
        # Load images
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")
        
        # Check for custom space background image
        for bg_name in ["sun.png", "space.webp", "space.jpeg", "space.png", "space.jpg"]:
            bg_path = os.path.join(asset_dir, bg_name)
            if os.path.exists(bg_path):
                try:
                    custom_bg = pygame.image.load(bg_path).convert()
                    self.bg_surface = pygame.transform.smoothscale(custom_bg, (self.width, self.height))
                    break # Stop if we found and loaded one successfully
                except Exception as e:
                    print(f"Warning: Could not load {bg_name}: {e}")
        
        def crop_circle(img_path, size):
            # Load and scale
            img = pygame.image.load(img_path).convert_alpha()
            scaled = pygame.transform.smoothscale(img, (size, size))
            # Create transparent mask layer
            mask = pygame.Surface((size, size), pygame.SRCALPHA)
            # Draw solid circle
            pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2)
            # Multiply pixel channels to apply the circular mask
            mask.blit(scaled, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            return mask

        try:
            self.img_earth = crop_circle(os.path.join(asset_dir, "earth.jpg"), 160)
            self.img_gs = crop_circle(os.path.join(asset_dir, "groundstation.jpeg"), 40)
            
            # The cubesat sprite should remain its native shape
            self.img_cubesat = pygame.image.load(os.path.join(asset_dir, "cubesat.png")).convert_alpha()
            self.img_cubesat = pygame.transform.smoothscale(self.img_cubesat, (35, 35))

        except Exception as e:
            print(f"Warning: Could not load assets: {e}")
            self.img_earth = pygame.Surface((120, 120))
            self.img_cubesat = pygame.Surface((35, 35))
            self.img_sun = pygame.Surface((280, 280))
            self.img_gs = pygame.Surface((25, 25))

    def draw(self, surface, state_debug):
        # Draw starry background space
        surface.blit(self.bg_surface, (0, 0)) 
        
        # --- Time tracking for visuals ---
        sim_time = state_debug.get("sim_time_s", 0.0)
        
        # Earth revolving around sun visually (fast-forwarded map)
        earth_revolve_angle = (sim_time * 0.002) % 360  
        earth_rev_rad = math.radians(earth_revolve_angle)
        
        # Earth rotating on its own axis visually
        earth_rot_angle = (sim_time * 0.02) % 360
        earth_rot_rad = math.radians(earth_rot_angle)
        
        # Draw Earth Orbit Ring
        pygame.draw.circle(surface, (50, 60, 80), (self.center_x, self.center_y), int(self.earth_orbit_radius), 1)
        
        # Calculate Earth center position
        ex = self.center_x + self.earth_orbit_radius * math.cos(earth_rev_rad)
        ey = self.center_y + self.earth_orbit_radius * math.sin(earth_rev_rad)
        
        # --- EARTH ---
        # Draw CubeSat orbit ring around earth
        pygame.draw.circle(surface, (70, 80, 100), (ex, ey), self.cubesat_orbit_radius, 1)

        # Rotate and Draw Earth
        rotated_earth = pygame.transform.rotate(self.img_earth, -earth_rot_angle)
        earth_rect = rotated_earth.get_rect(center=(ex, ey))
        surface.blit(rotated_earth, earth_rect)
        
        # --- GROUND STATION ---
        # Stick it to the surface of the earth (radius visual is ~80px) and spin it with the earth!
        earth_visual_radius = 80
        gx = ex + earth_visual_radius * math.cos(earth_rot_rad)
        gy = ey + earth_visual_radius * math.sin(earth_rot_rad)
        
        gs_rect = self.img_gs.get_rect(center=(gx, gy))
        surface.blit(self.img_gs, gs_rect)
        
        # --- CUBESAT ---
        # Position using physical orbit phase
        orbit_phase = state_debug.get("orbit_phase", 0.0)
        sat_angle_rad = orbit_phase * 2 * math.pi - (math.pi / 2) 
        
        cx = ex + self.cubesat_orbit_radius * math.cos(sat_angle_rad)
        cy = ey + self.cubesat_orbit_radius * math.sin(sat_angle_rad)
        
        # Comms Link: If visible, draw line from GS specifically to Satellite
        if state_debug.get("gs_visible", False):
            pygame.draw.line(surface, (50, 255, 100), (gx, gy), (cx, cy), 3)

        # Science Line: If over target, draw sensor ray downwards to earth's surface
        if state_debug.get("over_target", False):
            tx = ex + earth_visual_radius * math.cos(sat_angle_rad)
            ty = ey + earth_visual_radius * math.sin(sat_angle_rad)
            pygame.draw.line(surface, (255, 50, 100), (tx, ty), (cx, cy), 3)

        # Rotate and draw CubeSat sprite
        rotated_cubesat = pygame.transform.rotate(self.img_cubesat, -math.degrees(sat_angle_rad) - 90)
        cubesat_rect = rotated_cubesat.get_rect(center=(cx, cy))
        surface.blit(rotated_cubesat, cubesat_rect)
