import sys
import os
import numpy as np
import time

# Ensure the root project path is available for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from env.cubesat_env import CubeSatEnv
from demo.animation import AnimationApp
from models.enums import Action
from network.sim_client import PiSimClient

def load_random_tfrecord_image():
    """
    Attempts to pull a real image from the local TFRecords to send to the Pi.
    Falls back to a generated dummy image if TF isn't available or records are missing.
    """
    try:
        import tensorflow as tf
        from classifier.preprocess import parse_tfrecord
        import glob
        
        # Look for tfrecords anywhere in the project data folder
        tfrecord_files = glob.glob(os.path.join(project_root, "**", "*.tfrecord"), recursive=True)
        if tfrecord_files:
            # Pick a random file and take one image
            import random
            file_choice = random.choice(tfrecord_files)
            ds = tf.data.TFRecordDataset(file_choice)
            ds = ds.map(parse_tfrecord)
            for img, label in ds.take(1):
                img_np = img.numpy()
                # Normalize to 0-255 uint8 if it's float
                if img_np.max() <= 1.0:
                    img_np = img_np * 255
                return img_np.astype(np.uint8)
    except Exception as e:
        print(f"Note: Could not load real TFRecord ({e}). Using simulated camera data.")
        
    # Fallback: 96x96x3 random image simulating noise/clouds
    return np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)

def run():
    print("--- HIL Simulation Mode ---")
    
    # Configure the IP of your Raspberry Pi here
    PI_IP = input("Enter the Raspberry Pi IP Address (e.g., 192.168.1.100): ").strip()
    if not PI_IP:
        PI_IP = "127.0.0.1" # local fallback for testing
        
    print(f"Connecting to Pi at {PI_IP}:8000...")
    client = PiSimClient(pi_ip=PI_IP)
    
    env = CubeSatEnv()
    obs, info = env.reset()
    
    app = AnimationApp(env)
    last_debug_state = info.get("debug", env.render())
    
    # Hybrid Power state tracking
    base_power_draw = 5.0 # W
    max_cpu_power_draw = 15.0 # W additional draw when Pi CPU is at 100%
    
    while app.running:
        if not app.paused:
            valid_actions = env.get_valid_actions()
            
            # --- Capture Image Hook ---
            raw_frame = None
            if Action.CAPTURE_IMAGE in valid_actions and env.state.orbit.over_target:
                # If the environment is capable of capturing right now, we prepare a frame just in case
                # the policy decides to capture it this step.
                raw_frame = load_random_tfrecord_image()

            # --- Network Call to Raspberry Pi ---
            action_val, telemetry = client.get_decision(env.state, valid_actions, raw_frame)
            action = Action(action_val)
            
            # --- Inject Pi Telemetry ---
            if telemetry:
                # Update rendering logic
                env.state.cdh.memory_used_mb = telemetry['memory_used_mb']
                
                # Hybrid Power Model: Adjust power draw based on Pi's real CPU load
                # (Assuming nominal CPU is 10%, scale up to max_cpu_power_draw)
                cpu_load = telemetry['cpu_load']
                dynamic_draw = base_power_draw + (max_cpu_power_draw * (cpu_load / 100.0))
                env.state.eps.load_output_w = dynamic_draw
                
                # Print status
                print(f"Pi CPU: {cpu_load}% | Pi RAM: {telemetry['memory_used_mb']:.1f}MB | Action: {action.name}")
            else:
                print("Warning: Did not receive telemetry from Pi. Used NO_OP.")

            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            last_debug_state = info.get("debug", env.render())
            
            if terminated or truncated:
                print("Episode Ended. Restarting...")
                obs, info = env.reset()
                last_debug_state = info.get("debug", env.render())
                
        # Draw the frame
        app.render_frame(last_debug_state)

    print("HIL Demo terminated.")

if __name__ == "__main__":
    run()
