import config
import numpy as np


class UE:
    service_ids: np.ndarray
    content_ids: np.ndarray
    service_zipf_probabilities: np.ndarray
    content_zipf_probabilities: np.ndarray

    @classmethod
    def initialize_ue_class(cls) -> None:
        # Service Zipf distribution
        cls.service_ids = np.arange(0, config.NUM_SERVICES)
        service_ranks: np.ndarray = np.arange(1, config.NUM_SERVICES + 1)
        service_zipf_denom: float = np.sum(1 / service_ranks**config.ZIPF_BETA)
        cls.service_zipf_probabilities = (1 / service_ranks**config.ZIPF_BETA) / service_zipf_denom

        # Content Zipf distribution
        cls.content_ids = np.arange(config.NUM_SERVICES, config.NUM_FILES)
        content_ranks: np.ndarray = np.arange(1, config.NUM_CONTENTS + 1)
        content_zipf_denom: float = np.sum(1 / content_ranks**config.ZIPF_BETA)
        cls.content_zipf_probabilities = (1 / content_ranks**config.ZIPF_BETA) / content_zipf_denom

    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0])

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
        # Determine request type: 0=service, 1=content
        req_type: int = np.random.choice([0, 1])

        req_id: int = -1
        # Select file ID based on request type and corresponding Zipf probabilities
        if req_type == 0:  # Service request
            req_id = np.random.choice(UE.service_ids, p=UE.service_zipf_probabilities)
        else:  # Content request
            req_id = np.random.choice(UE.content_ids, p=UE.content_zipf_probabilities)

        # Determine input data size (L_m(t))
        req_size: int = 0
        if req_type == 0:
            req_size = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE)

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
