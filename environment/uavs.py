from __future__ import annotations
from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np


def _get_computing_latency_and_energy(uav: UAV, cpu_cycles: int) -> tuple[float, float]:
    """Calculate computing latency and energy for a UAV processing request."""
    assert uav._current_service_request_count != 0
    computing_capacity_per_request: float = config.UAV_COMPUTING_CAPACITY[uav.id] / uav._current_service_request_count
    latency: float = cpu_cycles / computing_capacity_per_request
    energy: float = config.K_CPU * cpu_cycles * (computing_capacity_per_request**2)
    return latency, energy


def _try_add_file_to_cache(uav: UAV, file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    used_space: int = np.sum(uav._working_cache * config.FILE_SIZES)
    if used_space + config.FILE_SIZES[file_id] <= config.UAV_STORAGE_CAPACITY[uav.id]:
        uav._working_cache[file_id] = True


class UAV:
    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), config.UAV_ALTITUDE])

        self._dist_moved: float = 0.0  # Distance moved in the current time slot
        self._current_covered_ues: list[UE] = []
        self._neighbors: list[UAV] = []
        self._current_collaborator: UAV | None = None
        self._current_service_request_count: int = 0
        self._energy_current_slot: float = 0.0  # Energy consumed for this time slot
        self.collision_violation: bool = False  # Track if UAV has violated minimum separation
        self.boundary_violation: bool = False  # Track if UAV has gone out of bounds

        # Cache and request tracking
        self._current_requested_files: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self.cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._working_cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts = np.zeros(config.NUM_FILES)  # For GDSF caching policy
        self._ema_scores = np.zeros(config.NUM_FILES)

        # Communication rates
        self._uav_uav_rate: float = 0.0
        self._uav_mbs_rate: float = 0.0

    @property
    def energy(self) -> float:
        return self._energy_current_slot

    @property
    def current_covered_ues(self) -> list[UE]:
        return self._current_covered_ues

    @property
    def neighbors(self) -> list[UAV]:
        return self._neighbors

    @property
    def current_collaborator(self) -> UAV | None:
        return self._current_collaborator

    def reset_for_next_step(self) -> None:
        """Reset UAV state for a new step."""
        self._current_covered_ues = []
        self._neighbors = []
        self._current_collaborator = None
        self._current_service_request_count = 0
        self._current_requested_files = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts = np.zeros(config.NUM_FILES)
        self._energy_current_slot = 0.0
        self.collision_violation = False
        self.boundary_violation = False

    def update_position(self, next_pos: np.ndarray) -> None:
        """Update the UAV's position to the new location chosen by the MARL agent."""
        new_pos: np.ndarray = np.append(next_pos, config.UAV_ALTITUDE)
        self._dist_moved = float(np.linalg.norm(new_pos - self.pos))
        self.pos = new_pos

    def set_neighbors(self, all_uavs: list[UAV]) -> None:
        """Set neighboring UAVs within sensing range for this UAV."""
        self._neighbors = []
        for other_uav in all_uavs:
            if other_uav.id != self.id:
                distance = float(np.linalg.norm(self.pos - other_uav.pos))
                if distance <= config.UAV_SENSING_RANGE:
                    self._neighbors.append(other_uav)

    def set_current_requested_files(self) -> None:
        """Update the current requested files based on the UEs covered by this UAV."""
        for ue in self._current_covered_ues:
            if ue.current_request:
                _, _, req_id = ue.current_request
                self._current_requested_files[req_id] = True

    def select_collaborator(self) -> None:
        """Choose a single collaborating UAV from its list of neighbours."""
        if not self._neighbors:
            self._set_rates()
            return

        best_collaborators: list[UAV] = []
        missing_requested_files: np.ndarray = self._current_requested_files & (~self.cache)
        max_missing_overlap: int = -1

        # Find neighbors with maximum overlap
        for neighbor in self._neighbors:
            overlap: int = int(np.sum(missing_requested_files & neighbor.cache))
            if overlap > max_missing_overlap:
                max_missing_overlap = overlap
                best_collaborators = [neighbor]
            elif overlap == max_missing_overlap:
                best_collaborators.append(neighbor)

        # If only one best collaborator, select it
        if len(best_collaborators) == 1:
            self._current_collaborator = best_collaborators[0]
            self._set_rates()
            return

        # If tie in overlap, select closest one(s)
        min_distance: float = float("inf")
        closest_collaborators: list[UAV] = []

        for collaborator in best_collaborators:
            distance: float = float(np.linalg.norm(self.pos - collaborator.pos))

            if distance < min_distance:
                min_distance = distance
                closest_collaborators = [collaborator]
            elif distance == min_distance:
                closest_collaborators.append(collaborator)

        # If still tied, select randomly
        if len(closest_collaborators) == 1:
            self._current_collaborator = closest_collaborators[0]
        else:
            self._current_collaborator = closest_collaborators[np.random.randint(0, len(closest_collaborators))]

        # Set communication rates once collaborator is selected
        self._set_rates()

    def set_freq_counts(self) -> None:
        """Set the request count for current slot based on cache availability."""
        for ue in self._current_covered_ues:
            req_type, _, req_id = ue.current_request
            self._freq_counts[req_id] += 1
            if self.cache[req_id]:
                if req_type == 0:
                    self._current_service_request_count += 1
            elif self._current_collaborator:
                self._current_collaborator._freq_counts[req_id] += 1
                if req_type == 0 and self._current_collaborator.cache[req_id]:
                    self._current_collaborator._current_service_request_count += 1

    def process_requests(self) -> None:
        """Process requests from UEs covered by this UAV."""
        self._working_cache = self.cache.copy()
        for ue in self._current_covered_ues:
            ue_uav_rate = comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos, self.pos), len(self._current_covered_ues))
            if ue.current_request[0] == 0:  # Service Request
                self._process_service_request(ue, ue_uav_rate)
            else:  # Content Request
                self._process_content_request(ue, ue_uav_rate)

    def _set_rates(self) -> None:
        """Set communication rates for UAV-MBS and UAV-UAV links."""
        self._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(self.pos, config.MBS_POS))
        if self._current_collaborator:
            self._uav_uav_rate = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, self._current_collaborator.pos))

    def _process_service_request(self, ue: UE, ue_uav_rate: float) -> None:
        """Process a service request from a UE."""
        _, req_size, req_id = ue.current_request
        assert req_id < config.NUM_SERVICES

        ue_assoc_uav_latency = req_size / ue_uav_rate
        cpu_cycles: int = config.CPU_CYCLES_PER_BYTE[req_id] * req_size

        if self.cache[req_id]:
            # Serve locally
            comp_latency, comp_energy = _get_computing_latency_and_energy(self, cpu_cycles)
            ue.latency_current_request = ue_assoc_uav_latency + comp_latency
            self._energy_current_slot += comp_energy
        elif self._current_collaborator:
            uav_uav_latency = req_size / self._uav_uav_rate
            if self._current_collaborator.cache[req_id]:
                # Served by collaborator
                comp_latency, comp_energy = _get_computing_latency_and_energy(self._current_collaborator, cpu_cycles)
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency + comp_latency
                self._current_collaborator._energy_current_slot += comp_energy
            else:
                # Served by MBS through collaborator
                uav_mbs_latency = req_size / self._current_collaborator._uav_mbs_rate
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency + uav_mbs_latency
                _try_add_file_to_cache(self._current_collaborator, req_id)
            _try_add_file_to_cache(self, req_id)
        else:
            # Offload to MBS directly
            uav_mbs_latency = req_size / self._uav_mbs_rate
            ue.latency_current_request = ue_assoc_uav_latency + uav_mbs_latency
            _try_add_file_to_cache(self, req_id)

    def _process_content_request(self, ue: UE, ue_uav_rate: float) -> None:
        """Process a content request from a UE."""
        _, _, req_id = ue.current_request
        assert req_id >= config.NUM_SERVICES

        file_size = config.FILE_SIZES[req_id]
        ue_assoc_uav_latency = file_size / ue_uav_rate

        if self.cache[req_id]:
            # Serve locally
            ue.latency_current_request = ue_assoc_uav_latency
        elif self._current_collaborator:
            uav_uav_latency = file_size / self._uav_uav_rate
            if self._current_collaborator.cache[req_id]:
                # Served by collaborator
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency
            else:
                # Served by MBS through collaborator
                uav_mbs_latency = file_size / self._current_collaborator._uav_mbs_rate
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency + uav_mbs_latency
                _try_add_file_to_cache(self._current_collaborator, req_id)
            _try_add_file_to_cache(self, req_id)
        else:
            # Offload to MBS directly
            uav_mbs_latency = file_size / self._uav_mbs_rate
            ue.latency_current_request = ue_assoc_uav_latency + uav_mbs_latency
            _try_add_file_to_cache(self, req_id)

    def update_ema_and_cache(self) -> None:
        """Update EMA scores and cache reactively."""
        self._ema_scores = config.GDSF_SMOOTHING_FACTOR * self._freq_counts + (1 - config.GDSF_SMOOTHING_FACTOR) * self._ema_scores
        self.cache = self._working_cache.copy()  # Update cache after processing all requests of all UAVs

    def gdsf_cache_update(self) -> None:
        """Update cache using the GDSF caching policy at a longer timescale."""
        priority_scores = self._ema_scores / config.FILE_SIZES
        sorted_file_ids = np.argsort(-priority_scores)
        self.cache = np.zeros(config.NUM_FILES, dtype=bool)
        used_space = 0.0
        for file_id in sorted_file_ids:
            file_size = config.FILE_SIZES[file_id]
            if used_space + file_size <= config.UAV_STORAGE_CAPACITY[self.id]:
                self.cache[file_id] = True
                used_space += file_size
            else:
                break

    def update_energy_consumption(self) -> None:
        """Update UAV energy consumption for the current time slot."""
        time_moving = self._dist_moved / config.UAV_SPEED
        time_hovering = config.TIME_SLOT_DURATION - time_moving
        fly_energy = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering
        self._energy_current_slot += fly_energy
