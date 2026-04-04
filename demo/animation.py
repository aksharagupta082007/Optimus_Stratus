import pygame

class AnimationApp:
    def __init__(self, env, title="CubeSat RL Demo"):
        self.env = env
        pygame.init()
        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        
        from demo.renderer import Renderer
        from demo.hud import HUD
        
        self.renderer = Renderer(self.width, self.height)
        self.hud = HUD(self.width, self.height)
        
        self.running = True
        self.paused = False
        
    def render_frame(self, state_debug):
        # Handle Window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

        # Clear and Draw
        self.renderer.draw(self.screen, state_debug)
        self.hud.draw(self.screen, state_debug)
        
        # Draw pausing text over everything
        if self.paused:
            font = pygame.font.SysFont("Courier New", 42, bold=True)
            text = font.render("PAUSED [Press Spacebar to resume]", True, (255, 255, 100))
            rect = text.get_rect(center=(self.width//2, self.height//2))
            padding = 15
            bg_rect = rect.inflate(padding*2, padding*2)
            
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(text, rect)

        # Update display
        pygame.display.flip()
        
        # 60 FPS capped loop
        self.clock.tick(60) 
