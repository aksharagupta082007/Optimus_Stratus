import sys
import os
import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from rl.baseline_policy import BaselinePolicy
from subsystems.hardware_monitor import HardwareMonitor
from classifier.preprocess import preprocess_frame
from classifier.infer_tflite import CloudClassifier
from models.enums import Action

app = FastAPI()
policy = BaselinePolicy()
hardware = HardwareMonitor()

try:
    classifier = CloudClassifier()
    HAS_CLASSIFIER = True
except Exception as e:
    print(f"Warning: Could not load classifier: {e}")
    HAS_CLASSIFIER = False

class DecisionRequest(BaseModel):
    state: Dict[str, Any]
    valid_actions: List[int]
    laptop_battery_pct: float
    raw_frame_list: Optional[List[List[List[int]]]] = None

@app.post("/decide")
def decide(req: DecisionRequest):
    state = req.state
    
    # 1. Read Real Pi Hardware
    hw_telemetry = hardware.get_telemetry()
    
    # 2. Inject Real Hardware & Laptop Battery into State
    # Ensure nested dicts exist
    for key in ['eps', 'cdh', 'thermal', 'payload']:
        if key not in state:
            state[key] = {}
            
    state['eps']['battery_soc_pct'] = req.laptop_battery_pct
    state['cdh']['memory_used_mb'] = hw_telemetry['memory_used_mb']
    state['thermal']['payload_temp_c'] = hw_telemetry['temperature_c']
    state['thermal']['battery_temp_c'] = hw_telemetry['temperature_c']
    
    # 3. Handle Image Classification if a frame was sent
    if req.raw_frame_list is not None and HAS_CLASSIFIER:
        print("Received raw image frame! Running classifier...")
        raw_frame = np.array(req.raw_frame_list, dtype=np.uint8)
        preprocessed = preprocess_frame(raw_frame)
        result = classifier.predict(preprocessed)
        
        state['payload']['classifier_confidence'] = result['classifier_confidence']
        state['payload']['current_frame_cloud_prob'] = result['current_frame_cloud_prob']
        state['payload']['current_frame_usefulness'] = result['current_frame_usefulness']
        state['payload']['classifier_success'] = result['classifier_success']
        # Map usefulness to frame class (2=CLEAR, 0=CLOUDY)
        state['payload']['frame_class'] = 2 if not result['is_cloudy'] else 0
        print(f"Classification Result: {result}")

    # 4. Run Policy
    valid_actions_enum = [Action(a) for a in req.valid_actions]
    decision = policy.decide(state, valid_actions=valid_actions_enum)
    
    print(f"Decision: {decision.action.name} (Reason: {decision.reason})")
    
    # Return decision and hardware telemetry
    return {
        "action": decision.action.value,
        "reason": decision.reason,
        "telemetry": hw_telemetry
    }

if __name__ == "__main__":
    print("Starting Pi HIL Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
