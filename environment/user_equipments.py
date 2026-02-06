import config
import numpy as np


class UE:
    all_ids: np.ndarray
    global_ranks: np.ndarray
    id_to_rank_map: dict[int, int]

    @classmethod
    def initialize_ue_class(cls) -> None:
        cls.all_ids = np.arange(config.NUM_FILES)  # Assume IDs 0 to NUM_SERVICES-1 are Services, rest are Contents
        cls.global_ranks = np.arange(1, config.NUM_FILES + 1)
        np.random.shuffle(cls.global_ranks)  # Currently random ranks assigned
        cls.id_to_rank_map = dict(zip(cls.all_ids, cls.global_ranks))  # Mapping from ID to rank
        zipf_denom: float = np.sum(1 / cls.global_ranks**config.ZIPF_BETA)
        cls.global_probs: np.ndarray = (1 / cls.global_ranks**config.ZIPF_BETA) / zipf_denom

    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0])
        self.battery_level: float = np.random.uniform(0.6, 1.0) * config.UE_BATTERY_CAPACITY  # Start at capacity between 60% to 100%

        self.current_request: tuple[int, int, int] = (0, 0, 0)  # Request : (req_type, req_size, req_id)
        self.latency_current_request: float = 0.0  # Latency for the current request
        self.assigned: bool = False

        # Random Waypoint Model
        self._waypoint: np.ndarray
        self._wait_time: int
        self._set_new_waypoint()  # Initialize first waypoint

        # Fairness Tracking
        self._successful_requests: int = 0
        self.service_coverage: float = 0.0

    def update_position(self) -> None:
        """Updates the UE's position for one time slot as per the Random Waypoint model."""
        if self._wait_time > 0:
            self._wait_time -= 1
            return

        direction_vec: np.ndarray = self._waypoint - self.pos[:2]
        distance_to_waypoint: float = float(np.linalg.norm(direction_vec))

        if config.UE_MAX_DIST >= distance_to_waypoint:  # Reached the waypoint
            self.pos[:2] = self._waypoint
            self._set_new_waypoint()
        else:  # Move towards the waypoint
            move_vector = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
            self.pos[:2] += move_vector

    def generate_request(self) -> None:
        """Generates a new request tuple for the current time slot."""

        # Check for Emergency Energy Request
        if self.battery_level < config.UE_CRITICAL_THRESHOLD:
            self.current_request = (2, 0, 0)
            self.latency_current_request = 0.0
            self.assigned = False
            return

        req_id: int = np.random.choice(UE.all_ids, p=UE.global_probs)
        req_type: int = 0 if req_id < config.NUM_SERVICES else 1
        req_size: int = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE) if req_type == 0 else 0
        self.current_request = (req_type, req_size, req_id)
        self.latency_current_request = 0.0
        self.assigned = False

    def update_service_coverage(self, current_time_step_t: int) -> None:
        """Updates the fairness metric based on service outcome in the current slot."""
        if self.assigned and self.latency_current_request <= config.TIME_SLOT_DURATION:
            self._successful_requests += 1

        assert current_time_step_t > 0
        self.service_coverage = self._successful_requests / current_time_step_t

    def _set_new_waypoint(self):
        """Set a new destination, speed, and wait time as per the Random Waypoint model."""
        self._waypoint = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)])
        self._wait_time = np.random.randint(0, config.UE_MAX_WAIT_TIME + 1)

    def update_battery(self, harv_energy: float, ue_transmit_time: float) -> None:
        """Updates battery level based on consumption and harvesting."""
        consumed_energy: float = config.UE_STATIC_POWER * config.TIME_SLOT_DURATION
        consumed_energy += config.TRANSMIT_POWER * ue_transmit_time
        self.battery_level = min(config.UE_BATTERY_CAPACITY, self.battery_level - consumed_energy + harv_energy)
        self.battery_level = max(0.0, self.battery_level)
