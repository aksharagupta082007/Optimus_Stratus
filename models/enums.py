from enum import Enum, IntEnum, auto


class SpacecraftMode(IntEnum):
    """
    Top-level spacecraft operational mode.
    This is the coarse mission/FSW mode.
    """
    BOOT = 0
    SAFE = 1
    DETUMBLE = 2
    NOMINAL = 3
    SCIENCE = 4
    DOWNLINK = 5
    ECLIPSE_POWER_SAVE = 6
    FAULT_RECOVERY = 7
    SURVIVAL = 8


class AttitudeMode(IntEnum):
    """
    ADCS pointing/control mode.
    Chosen to reflect common smallsat GNC mission profiles.
    """
    UNKNOWN = 0
    DETUMBLE = 1
    SUN_POINTING = 2
    NADIR_POINTING = 3
    GROUND_TRACKING = 4
    INERTIAL_HOLD = 5
    SLEW_MANEUVER = 6
    MOMENTUM_DUMP = 7
    SAFE_SUN_ACQUIRE = 8


class PayloadMode(IntEnum):
    """
    Payload/imager operating state.
    """
    OFF = 0
    STANDBY = 1
    WARMUP = 2
    READY = 3
    IMAGING = 4
    PROCESSING = 5
    ERROR = 6


class CommsMode(IntEnum):
    """
    Communications subsystem state.
    """
    OFF = 0
    LISTEN = 1
    BEACON = 2
    LOW_RATE_TX = 3
    HIGH_RATE_TX = 4
    RX_ONLY = 5
    TXRX = 6
    PASS_PREP = 7
    ERROR = 8


class PowerMode(IntEnum):
    """
    EPS power policy / operating posture.
    """
    POWER_POSITIVE = 0
    NOMINAL = 1
    POWER_SAVE = 2
    CHARGE_PRIORITY = 3
    ECLIPSE_CONSERVE = 4
    CRITICAL_LOW_POWER = 5


class ThermalMode(IntEnum):
    """
    Thermal-control posture.
    """
    NOMINAL = 0
    COLD_SOAK = 1
    HOT_SOAK = 2
    BATTERY_HEATING = 3
    PAYLOAD_HEATING = 4
    THERMAL_PROTECT = 5
    SURVIVAL = 6


class DataHandlingMode(IntEnum):
    """
    CDH / onboard data handling mode.
    """
    IDLE = 0
    BUFFERING = 1
    CLASSIFYING = 2
    COMPRESSING = 3
    QUEUEING = 4
    DOWNLINKING = 5
    PURGING = 6
    RECOVERY = 7


class Action(IntEnum):
    """
    RL / operations action space.

    These are explicit commands the agent may issue.
    They are intentionally mode-based, closer to real
    spacecraft operations than simple toy actions.
    """
    NO_OP = 0
    ENTER_SAFE_MODE = 1
    DETUMBLE = 2

    SUN_POINT_CHARGE = 10
    NADIR_POINT_STANDBY = 11
    SLEW_TO_GROUND = 12
    HOLD_INERTIAL = 13
    DESATURATE_WHEELS = 14

    PAYLOAD_WARMUP = 20
    CAPTURE_IMAGE = 21
    RUN_CLASSIFIER = 22
    STORE_FRAME = 23
    DISCARD_FRAME = 24
    COMPRESS_DATA = 25

    PREPARE_DOWNLINK = 30
    DOWNLINK_LOW_RATE = 31
    DOWNLINK_HIGH_RATE = 32
    SEND_BEACON = 33

    ENABLE_BATTERY_HEATER = 40
    ENABLE_PAYLOAD_HEATER = 41
    DISABLE_HEATERS = 42

    FAULT_RECOVERY = 50
    RESET_PAYLOAD = 51
    RESET_COMMS = 52
    RESET_ADCS = 53
    RESET_CDH = 54


class FaultLevel(IntEnum):
    """
    Fault severity.
    """
    NONE = 0
    WARNING = 1
    LIMIT = 2
    CRITICAL = 3
    FATAL = 4


class FaultType(IntEnum):
    """
    Broad subsystem fault taxonomy.
    """
    NONE = 0
    LOW_BATTERY = 1
    BATTERY_OVERTEMP = 2
    BATTERY_UNDERTEMP = 3
    SOLAR_GENERATION_LOSS = 4

    ADCS_POINTING_LOST = 10
    WHEEL_SATURATION = 11
    SENSOR_DEGRADED = 12

    PAYLOAD_STUCK = 20
    PAYLOAD_OVERTEMP = 21
    CLASSIFIER_FAILURE = 22

    COMMS_LINK_LOSS = 30
    COMMS_TIMEOUT = 31
    DOWNLINK_ABORTED = 32

    MEMORY_FULL = 40
    STORAGE_CORRUPTION = 41
    CDH_RESET_LOOP = 42

    THERMAL_LIMIT_HOT = 50
    THERMAL_LIMIT_COLD = 51

    UNKNOWN = 99


class SunlightState(IntEnum):
    ECLIPSE = 0
    SUNLIT = 1


class GroundPassState(IntEnum):
    """
    Visibility quality for DTE contact.
    """
    NONE = 0
    ACQUIRE = 1
    LOW_ELEVATION = 2
    MID_ELEVATION = 3
    HIGH_ELEVATION = 4
    LOSS = 5


class LinkQuality(IntEnum):
    NONE = 0
    POOR = 1
    MARGINAL = 2
    GOOD = 3
    EXCELLENT = 4


class TargetOpportunity(IntEnum):
    """
    Imaging opportunity quality.
    """
    NONE = 0
    POOR_LIGHT = 1
    VALID = 2
    HIGH_VALUE = 3


class FrameClass(IntEnum):
    """
    Output of onboard cloud/usefulness screening.
    """
    NONE = 0
    CLOUDY = 1
    PARTLY_CLOUDY = 2
    CLEAR = 3
    HIGH_VALUE_CLEAR = 4


class MemoryPressure(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class BatteryBand(IntEnum):
    CRITICAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    FULL = 4


class TemperatureBand(IntEnum):
    TOO_COLD = 0
    COLD = 1
    NOMINAL = 2
    WARM = 3
    TOO_HOT = 4


class PointingQuality(IntEnum):
    INVALID = 0
    COARSE = 1
    USABLE = 2
    PRECISE = 3


class EpisodeEndReason(IntEnum):
    NOT_DONE = 0
    MAX_STEPS = 1
    BATTERY_DEPLETED = 2
    THERMAL_FAILURE = 3
    FATAL_FAULT = 4
    STORAGE_FAILURE = 5
    MANUAL_ABORT = 6


class RewardEvent(Enum):
    """
    Semantic reward tags for logging/debugging.
    String Enum not required; regular Enum is enough.
    """
    USEFUL_IMAGE_CAPTURED = auto()
    CLOUDY_IMAGE_CAPTURED = auto()
    CLASSIFICATION_CORRECT = auto()
    CLASSIFICATION_WRONG = auto()
    FRAME_STORED = auto()
    FRAME_DISCARDED = auto()
    SUCCESSFUL_DOWNLINK = auto()
    FAILED_DOWNLINK = auto()
    POWER_POSITIVE_STEP = auto()
    POWER_NEGATIVE_STEP = auto()
    THERMAL_PROTECTED = auto()
    THERMAL_VIOLATION = auto()
    MEMORY_OVERFLOW = auto()
    SAFE_MODE_ENTRY = auto()
    FAULT_RECOVERED = auto()


class SubsystemName(Enum):
    EPS = "eps"
    ADCS = "adcs"
    PAYLOAD = "payload"
    COMMS = "comms"
    CDH = "cdh"
    THERMAL = "thermal"


class ResetCause(IntEnum):
    NONE = 0
    COMMAND = 1
    WATCHDOG = 2
    BROWNOUT = 3
    FAULT_PROTECTION = 4
    UNKNOWN = 5