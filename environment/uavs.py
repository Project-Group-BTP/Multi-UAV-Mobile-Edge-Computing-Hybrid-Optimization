from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np
from typing import List, Optional, Tuple


def _get_computing_latency_and_energy(uav: "UAV", cpu_cycles: int) -> Tuple[float, float]:
    """Calculate computing latency and energy for a UAV processing request."""
    assert uav._current_service_request_count != 0
    computing_capacity_per_request = config.UAV_COMPUTING_CAPACITY[uav.id] / uav._current_service_request_count
    latency = cpu_cycles / computing_capacity_per_request
    energy = config.K_CPU * cpu_cycles * (computing_capacity_per_request**2)
    return latency, energy


def _try_add_file_to_cache(uav: "UAV", file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    used_space = np.sum(uav.cache * config.FILE_SIZES)
    if used_space + config.FILE_SIZES[file_id] <= config.UAV_STORAGE_CAPACITY[uav.id]:
        uav.cache[file_id] = True


class UAV:
    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), config.UAV_ALTITUDE])

        self._dist_moved: float = 0.0  # Distance moved in the current time slot
        self._current_covered_ues: List[UE] = []
        self._current_collaborator: Optional["UAV"] = None
        self._current_service_request_count: int = 0
        self._energy_current_slot: float = 0.0  # Energy consumed for this time slot

        # Cache and request tracking
        self._current_requested_files: np.ndarray = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)
        self.cache: np.ndarray = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)

        # Communication rates
        self._uav_uav_rate: float = 0.0
        self._uav_mbs_rate: float = 0.0

    def reset_for_time_slot(self) -> None:
        """Reset UAV state for a new time slot."""
        self._current_covered_ues = []
        self._current_collaborator = None
        self._current_service_request_count = 0
        self._current_requested_files = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)
        self._energy_current_slot = 0.0

    def update_position(self, next_pos: np.ndarray) -> None:
        """Update the UAV's position to the new location chosen by the MARL agent."""
        new_pos = np.append(next_pos, config.UAV_ALTITUDE)
        self._dist_moved = float(np.linalg.norm(new_pos - self.pos))
        self.pos = new_pos

    def get_neighbors(self, all_uavs: List["UAV"]) -> List["UAV"]:
        """Get neighboring UAVs within sensing range for this UAV."""
        neighbors: List["UAV"] = []
        for other_uav in all_uavs:
            if other_uav.id != self.id:
                distance = float(np.linalg.norm(self.pos - other_uav.pos))
                if distance <= config.UAV_SENSING_RANGE:
                    neighbors.append(other_uav)
        return neighbors

    def _set_covered_ues(self, ues: List[UE]) -> None:
        """Set the list of UEs covered by this UAV."""
        self._current_covered_ues = [ue for ue in ues if np.linalg.norm(self.pos[:2] - ue.pos[:2]) <= config.UAV_COVERAGE_RADIUS]

    def set_current_requested_files(self, ues: List[UE]) -> None:
        """Update the current requested files based on the UEs covered by this UAV."""
        self._set_covered_ues(ues)
        self._current_requested_files = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)

        for ue in self._current_covered_ues:
            if ue.current_request:
                _, _, req_id = ue.current_request
                self._current_requested_files[req_id] = True

    def select_collaborator(self, neighbors: List["UAV"]) -> None:
        """Choose a single collaborating UAV from the list of neighbours."""
        if not neighbors:
            return

        best_collaborators: List["UAV"] = []
        max_overlap: int = -1

        # Find neighbors with maximum overlap
        for neighbor in neighbors:
            overlap = int(np.sum(self._current_requested_files & neighbor.cache))

            if overlap > max_overlap:
                max_overlap = overlap
                best_collaborators = [neighbor]
            elif overlap == max_overlap:
                best_collaborators.append(neighbor)

        # If only one best collaborator, select it
        if len(best_collaborators) == 1:
            self._current_collaborator = best_collaborators[0]
            self._set_rates()
            return

        # If tie in overlap, select closest one(s)
        min_distance: float = float("inf")
        closest_collaborators: List["UAV"] = []

        for collaborator in best_collaborators:
            distance = float(np.linalg.norm(self.pos - collaborator.pos))

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

    def _set_rates(self) -> None:
        """Set communication rates for UAV-MBS and UAV-UAV links."""
        self._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(self.pos, config.MBS_POS))
        if self._current_collaborator:
            self._uav_uav_rate = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, self._current_collaborator.pos))

    def set_current_service_request_count(self) -> None:
        """Set the request count for current slot based on cache availability."""
        for ue in self._current_covered_ues:
            req_type, _, req_id = ue.current_request
            if req_type == 0:  # Service Request
                if self.cache[req_id]:
                    self._current_service_request_count += 1
                elif self._current_collaborator:
                    self._current_collaborator._current_service_request_count += 1

    def process_requests(self) -> None:
        """Process requests from UEs covered by this UAV."""
        for ue in self._current_covered_ues:
            if ue.current_request:
                ue_uav_rate = comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos, self.pos), len(self._current_covered_ues))

                if ue.current_request[0] == 0:  # Service Request
                    self._process_service_request(ue, ue_uav_rate)
                else:  # Content Request
                    self._process_content_request(ue, ue_uav_rate)

    def _process_service_request(self, ue: UE, ue_uav_rate: float) -> None:
        """Process a service request from a UE."""
        _, req_size, req_id = ue.current_request
        assert req_id < config.NUM_SERVICES

        ue_assoc_uav_latency = req_size / ue_uav_rate
        cpu_cycles = config.CPU_CYCLES_PER_BYTE[req_id] * req_size
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

    def update_energy_consumption(self) -> None:
        """Update UAV energy consumption for the current time slot."""
        time_moving = self._dist_moved / config.UAV_SPEED
        time_hovering = config.TIME_SLOT_DURATION - time_moving
        fly_energy = config.POWER_MOVE * time_moving + config.POWER_HOVER * time_hovering
        self._energy_current_slot += fly_energy
