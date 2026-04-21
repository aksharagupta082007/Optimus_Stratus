import requests
import psutil
from models.state_models import SpacecraftState

class PiSimClient:
    def __init__(self, pi_ip: str, port: int = 8000):
        self.url = f"http://{pi_ip}:{port}/decide"

    def _serialize_state(self, state: SpacecraftState) -> dict:
        """
        Converts the SpacecraftState object into a nested dictionary
        that matches the paths expected by BaselinePolicy._get_path().
        """
        return {
            "eps": {
                "battery_soc_pct": state.eps.battery_soc_pct
            },
            "thermal": {
                "payload_temp_c": state.thermal.payload_temp_c,
                "battery_temp_c": state.thermal.battery_temp_c
            },
            "cdh": {
                "memory_used_mb": state.cdh.memory_used_mb,
                "memory_capacity_mb": state.cdh.memory_capacity_mb,
                "downlink_queue_mb": state.cdh.downlink_queue_mb
            },
            "spacecraft": {
                "mode": state.mode.value
            },
            "payload": {
                "mode": state.payload.mode.value,
                "has_frame": state.payload.has_frame,
                "current_frame_class": state.payload.current_frame_class.value
            },
            "orbit": {
                "ground_pass_state": state.orbit.ground_pass_state.value if hasattr(state.orbit, 'ground_pass_state') else 0,
                "target_opportunity": state.orbit.target_opportunity.value
            },
            "comms": {
                "link_quality": state.comms.link_quality.value if hasattr(state.comms, 'link_quality') else 0,
                "pass_state": getattr(state.comms, 'pass_state', 0) if hasattr(state.comms, 'pass_state') and hasattr(getattr(state.comms, 'pass_state'), 'value') else 0
            },
            "adcs": {
                "pointing_quality": state.adcs.pointing_quality.value if hasattr(state.adcs, 'pointing_quality') else 0,
                "wheel_saturation": getattr(state.adcs, 'wheels_saturated', False),
                "needs_detumble": getattr(state.adcs, 'needs_detumble', False),
                "mode": state.adcs.mode.value
            },
            "faults": {
                "max_fault_level": state.faults.highest_fault_level.value,
                "active_faults": [],
                "safe_mode_latched": state.faults.safe_mode_latched
            }
        }

    def get_decision(self, state: SpacecraftState, valid_actions: list, raw_frame=None):
        # Read laptop battery
        battery = psutil.sensors_battery()
        laptop_battery_pct = battery.percent if battery else 100.0

        state_dict = self._serialize_state(state)
        action_ints = [a.value for a in valid_actions]

        payload = {
            "state": state_dict,
            "valid_actions": action_ints,
            "laptop_battery_pct": float(laptop_battery_pct)
        }
        
        if raw_frame is not None:
            # raw_frame should be a numpy array of shape (H, W, 3) uint8
            payload["raw_frame_list"] = raw_frame.tolist()

        try:
            response = requests.post(self.url, json=payload, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            return data['action'], data['telemetry']
        except requests.exceptions.Timeout:
            print("Error: Pi took too long to respond.")
            return 0, None
        except Exception as e:
            print(f"Error communicating with Pi: {e}")
            # Fallback to no-op (Action(0))
            return 0, None
