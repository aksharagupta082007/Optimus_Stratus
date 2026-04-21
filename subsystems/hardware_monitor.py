import psutil
import os

class HardwareMonitor:
    """
    Reads physical hardware telemetry from the Raspberry Pi.
    Uses psutil for cross-platform memory/CPU, and sysfs for Pi temperatures.
    """
    def __init__(self):
        # Call once to initialize the non-blocking CPU percent calculation
        psutil.cpu_percent(interval=None)

    def get_telemetry(self) -> dict:
        # Memory: Convert bytes to MB
        mem = psutil.virtual_memory()
        memory_used_mb = (mem.total - mem.available) / (1024 * 1024)
        
        # CPU Load: Non-blocking percentage since last call
        cpu_load = psutil.cpu_percent(interval=None)

        # CPU Temperature (Linux/Raspberry Pi specific)
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = float(f.read().strip()) / 1000.0
        except FileNotFoundError:
            # Fallback if testing this server on a Windows/Mac machine
            temp_c = 45.0 + (cpu_load * 0.2) # Fake temperature based on load

        return {
            "memory_used_mb": float(memory_used_mb),
            "cpu_load": float(cpu_load),
            "temperature_c": float(temp_c)
        }
