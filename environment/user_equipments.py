import config
import numpy as np
from typing import Optional, Tuple


class UE:
    num_files: int = 0
    file_ids: np.ndarray = None
    zipf_probabilities: np.ndarray = None

    @classmethod
    def initialize_ue_class(cls):
        cls.num_files = config.NUM_SERVICES + config.NUM_CONTENTS
        ranks = np.arange(1, cls.num_files + 1)
        zipf_denom = np.sum(1 / ranks**config.ZIPF_BETA)
        cls.zipf_probabilities = (1 / ranks**config.ZIPF_BETA) / zipf_denom
        cls.file_ids = np.arange(1, cls.num_files + 1)

    def __init__(self, ue_id: int):
        self.id: int = ue_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0])

        # Request : (req_type, req_size, req_id)
        self.current_request: Optional[Tuple[int, int, int]] = None

        # Random Waypoint Model
        self.waypoint: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)])

        # Fairness Tracking
        self._successful_service_ticks: int = 0
        self.service_coverage: float = 0.0

    def update_position(self) -> None:
        """
        Updates the UE's position for one time slot as per the Random Waypoint model.
        """
        direction_vec = self.waypoint - self.pos[:2]
        # Next destination
        self.waypoint = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)])
        distance_to_waypoint = np.linalg.norm(direction_vec)
        if distance_to_waypoint == 0:
            return
        move_vector = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
        self.pos[:2] += move_vector

    def generate_request(self) -> Tuple[int, int, int]:
        """
        Generates a new request tuple λ_m(t) for the current time slot.
        The request type (service/content) is chosen, and the specific file ID is selected based on the Zipf popularity distribution.
        Returns:
            Tuple of (req_type, req_size, req_id)
        """
        # Determine request type: 0=service, 1=content
        req_type = np.random.choice([0, 1])

        # Select file ID based on Zipf probabilities
        req_id = np.random.choice(UE.file_ids, p=UE.zipf_probabilities)

        # Determine input data size (L_m(t))
        if req_type == 0:  # Service request
            req_size = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE)
        else:  # Content request
            req_size = 0

        self.current_request = (req_type, req_size, req_id)
        return self.current_request

    def update_service_coverage(self, current_time_t: int, served_successfully: bool) -> None:
        """
        Updates the fairness metric based on service outcome in the current slot.
        Args:
            current_time_t: The current time slot index (t).
            served_successfully: A boolean indicating if the UE's request was
                                 covered and completed within the deadline τ.
        """
        if served_successfully:
            self._successful_service_ticks += 1

        if current_time_t > 0:
            self.service_coverage = self._successful_service_ticks / current_time_t
