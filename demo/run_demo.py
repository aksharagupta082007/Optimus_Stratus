import sys
import os

# Ensure the root project path is available for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from env.cubesat_env import CubeSatEnv
from demo.animation import AnimationApp
from rl.baseline_policy import BaselinePolicy

def run():
    print("Initializing Simulation Environment...")
    env = CubeSatEnv()
    obs, info = env.reset()
    
    print("Loading Baseline AI Policy...")
    policy = BaselinePolicy()
    
    print("Starting Pygame Animation Window...")
    app = AnimationApp(env)
    
    # Track the last debug state to render it correctly while paused
    last_debug_state = info.get("debug", env.render())
    
    while app.running:
        if not app.paused:
            # 1. Ask the AI what to do based on current valid actions
            valid_actions = env.get_valid_actions()
            
            # Use the rule-based baseline policy to fly the CubeSat intelligently!
            decision = policy.decide(state=env.state, valid_actions=valid_actions)
            action = decision.action
            
            # 2. Step the Environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            last_debug_state = info.get("debug", env.render())
            
            # If the satellite dies or completes limit, reset.
            if terminated or truncated:
                print("Episode Ended. Restarting...")
                obs, info = env.reset()
                last_debug_state = info.get("debug", env.render())
                
        # 3. Draw the frame 
        app.render_frame(last_debug_state)

    print("Demo terminated.")

if __name__ == "__main__":
    run()
