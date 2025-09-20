import config
import numpy as np
from typing import Tuple


class UE:
    num_files: int
    file_ids: np.ndarray
    zipf_probabilities: np.ndarray

    @classmethod
    def initialize_ue_class(cls) -> None:
        cls.num_files = config.NUM_SERVICES + config.NUM_CONTENTS
        ranks: np.ndarray = np.arange(1, cls.num_files + 1)  # Keep ranks 1-based for Zipf calculation
        zipf_denom: float = np.sum(1 / ranks**config.ZIPF_BETA)
        cls.zipf_probabilities = (1 / ranks**config.ZIPF_BETA) / zipf_denom
        cls.file_ids = np.arange(0, cls.num_files)  # Changed to 0-based indexing

    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0])

        # Request : (req_type, req_size, req_id)
        self.current_request: Tuple[int, int, int]
        # Latency for the current request in the ongoing time slot
        self.latency_current_request: float = 0.0

        # Random Waypoint Model
        self.waypoint: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)])


        # Fairness Tracking
        self._successful_service_ticks: int = 0
        self.service_coverage: float = 0.0

    def update_position(self) -> None:
        """
        Updates the UE's position for one time slot as per the Random Waypoint model.
        """
        direction_vec: np.ndarray = self.waypoint - self.pos[:2]
        # Next destination
        self.waypoint = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)])
        distance_to_waypoint: float = float(np.linalg.norm(direction_vec))
        if distance_to_waypoint == 0:
            return
        move_vector: np.ndarray = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
        self.pos[:2] += move_vector

    def generate_request(self) -> None:
        """
        Generates a new request tuple λ_m(t) for the current time slot.
        The request type (service/content) is chosen, and the specific file ID is selected based on the Zipf popularity distribution.
        Returns:
            Tuple of (req_type, req_size, req_id) - req_id is now 0-indexed
        """
        # Determine request type: 0=service, 1=content
        req_type: int = np.random.choice([0, 1])

        # Select file ID based on Zipf probabilities (now 0-indexed)
        req_id: int = np.random.choice(UE.file_ids, p=UE.zipf_probabilities)

        # Determine input data size (L_m(t))
        if req_type == 0:  # Service request
            req_size: int = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE)
        else:  # Content request
            req_size = 0

        self.current_request = (req_type, req_size, req_id)

    def update_service_coverage(self, current_time_t: int) -> None:
        """
        Updates the fairness metric based on service outcome in the current slot.
        Args:
            current_time_t: The current time slot index (t).
        """
        if self.latency_current_request <= config.TIME_SLOT_DURATION:
            self._successful_service_ticks += 1

        if current_time_t > 0:
            self.service_coverage = self._successful_service_ticks / current_time_t
