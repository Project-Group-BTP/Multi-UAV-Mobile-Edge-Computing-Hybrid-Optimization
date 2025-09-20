from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np
from typing import List, Optional, Tuple


def _get_computing_latency_and_energy(uav: "UAV", cpu_cycles: int) -> Tuple[float, float]:
    assert uav.current_slot_request_count != 0  # rethink
    computing_capacity_per_request = uav.computing_capacity / uav.current_slot_request_count
    return cpu_cycles / computing_capacity_per_request, config.K_CPU * cpu_cycles * (computing_capacity_per_request ** 2)


def _try_add_file_to_cache(uav: "UAV", file_id: int) -> None:
    used_space = np.sum(uav.cache * config.FILE_SIZES)
    if used_space + config.FILE_SIZES[file_id] <= uav.storage_capacity:
        uav.cache[file_id] = True


class UAV:
    def __init__(self, uav_id: int) -> None:
        self.id: int = uav_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), config.UAV_ALTITUDE])
        self.dist_moved: float = 0.0  # Distance moved in the current time slot
        self.computing_capacity: int = config.UAV_COMPUTING_CAP
        self.storage_capacity: int = config.UAV_STORAGE_CAP
        self.coverage_radius: float = config.UAV_COVERAGE_RADIUS
        self.sensing_range: float = config.UAV_SENSING_RANGE

        self.current_covered_ues: List[UE] = []
        self.current_collaborator: Optional["UAV"] = None  # Rethink optional
        self.current_slot_request_count: int = 0

        # Energy consumed for this time slot
        self.energy_current_slot: float = 0.0

        # Caching State : A numpy array of booleans indicating if a file is cached (0-indexed)
        self.current_requested_files: np.ndarray = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)
        self.cache: np.ndarray = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)

        # Rates
        self.uav_uav_rate: float = 0.0
        self.uav_mbs_rate: float = 0.0

    def update_position(self, next_pos: np.ndarray) -> None:
        """
        Updates the UAV's position to the new location chosen by the MARL agent.
        """
        new_pos = np.append(next_pos, config.UAV_ALTITUDE)
        self.dist_moved = float(np.linalg.norm(new_pos - self.pos))
        self.pos = new_pos

    def _set_covered_ues(self, ues: List[UE]) -> None:
        """
        Returns a list of UEs covered by this UAV.
        """
        self.current_covered_ues = [ue for ue in ues if np.linalg.norm(self.pos[:2] - ue.pos[:2]) <= self.coverage_radius]

    def set_current_requested_files(self, ues: List[UE]) -> None:
        """
        Updates the current requested files based on the UEs covered by this UAV.
        """
        self.current_slot_request_count = 0
        self._set_covered_ues(ues)
        self.current_requested_files = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)
        for ue in self.current_covered_ues:
            if ue.current_request:
                _, _, req_id = ue.current_request
                self.current_requested_files[req_id] = True

    def get_neighbors(self, all_uavs: List["UAV"]) -> List["UAV"]:
        """
        Get neighboring UAVs within sensing range for this UAV.
        """
        neighbors: List["UAV"] = []
        for other_uav in all_uavs:
            if other_uav.id != self.id:
                distance = float(np.linalg.norm(self.pos - other_uav.pos))
                if distance <= self.sensing_range:
                    neighbors.append(other_uav)
        return neighbors

    def select_collaborator(self, neighbors: List["UAV"]) -> None:
        """
        Chooses a single collaborating UAV from the list of neighbours.
        """
        if not neighbors:
            return None  # Rethink what to do for none
        best_collaborators: List["UAV"] = []
        max_overlap: int = -1

        # Find neighbors with maximum overlap
        for neighbor in neighbors:
            # Calculate overlap using bitwise AND on boolean arrays
            overlap = int(np.sum(self.current_requested_files & neighbor.cache))

            if overlap > max_overlap:
                max_overlap = overlap
                best_collaborators = [neighbor]
            elif overlap == max_overlap:
                best_collaborators.append(neighbor)

        # If only one best collaborator, return it
        if len(best_collaborators) == 1:
            self.current_collaborator = best_collaborators[0]
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
            self.current_collaborator = closest_collaborators[0]
        else:
            self.current_collaborator = closest_collaborators[np.random.randint(0, len(closest_collaborators))]

    def set_current_slot_request_count(self) -> None:
        for ue in self.current_covered_ues:
            _, _, req_id = ue.current_request
            if self.cache[req_id]:
                self.current_slot_request_count += 1
            elif self.current_collaborator:  # Rethink for None
                self.current_collaborator.current_slot_request_count += 1

    def process_requests(self) -> None:
        """
        Processes requests from UEs covered by this UAV.
        Either serves from cache or offloads to collaborators/MBS.
        """
        # Request from Associated UE
        self.set_rates()
        self.energy_current_slot = 0.0
        for ue in self.current_covered_ues:
            ue_uav_rate = comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos, self.pos), len(self.current_covered_ues))
            if ue.current_request[0] == 0:  # Service Request
                self.process_service_request(ue, ue_uav_rate)
            else:  # Content Request
                self.process_content_request(ue, ue_uav_rate)
            ue.update_service_coverage

    def set_rates(self) -> None:
        self.uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(self.pos, config.MBS_POS))
        if self.current_collaborator:  # rethink for None
            self.uav_uav_rate = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, self.current_collaborator.pos))

    def process_service_request(self, ue: UE, ue_uav_rate: float) -> None:
        _, req_size, req_id = ue.current_request
        ue_assoc_uav_latency = req_size / ue_uav_rate
        cpu_cycles = config.CPU_CYCLES_PER_BYTE[req_id] * req_size
        if self.cache[req_id]:
            # Serve locally
            comp_latency, comp_energy = _get_computing_latency_and_energy(self, cpu_cycles)
            ue.latency_current_request = ue_assoc_uav_latency + comp_latency
            self.energy_current_slot += comp_energy
        elif self.current_collaborator:
            uav_uav_latency = req_size / self.uav_uav_rate
            if self.current_collaborator.cache[req_id]:
                # Served by collaborator
                comp_latency, comp_energy = _get_computing_latency_and_energy(self.current_collaborator, cpu_cycles)
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency + comp_latency
                self.current_collaborator.energy_current_slot += comp_energy
            else:
                # Served by MBS
                uav_mbs_latency = req_size / self.current_collaborator.uav_mbs_rate
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency + uav_mbs_latency
                _try_add_file_to_cache(self.current_collaborator, req_id)
        else:  # Rethink this path
            # Offload to MBS directly
            uav_mbs_latency = req_size / self.uav_mbs_rate
            ue.latency_current_request = ue_assoc_uav_latency + uav_mbs_latency
            _try_add_file_to_cache(self, req_id)

    def process_content_request(self, ue: UE, ue_uav_rate: float) -> None:
        _, _, req_id = ue.current_request
        file_size = config.FILE_SIZES[req_id]
        ue_assoc_uav_latency = file_size / ue_uav_rate
        if self.cache[req_id]:
            # Serve locally
            ue.latency_current_request = ue_assoc_uav_latency
        elif self.current_collaborator:
            uav_uav_latency = file_size / self.uav_uav_rate
            if self.current_collaborator.cache[req_id]:
                # Served by collaborator
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency
            else:
                # Served by MBS
                uav_mbs_latency = file_size / self.current_collaborator.uav_mbs_rate
                ue.latency_current_request = ue_assoc_uav_latency + uav_uav_latency + uav_mbs_latency
                _try_add_file_to_cache(self.current_collaborator, req_id)
        else:  # Rethink this path
            # Offload to MBS directly
            uav_mbs_latency = file_size / self.uav_mbs_rate
            ue.latency_current_request = ue_assoc_uav_latency + uav_mbs_latency
            _try_add_file_to_cache(self, req_id)

    def update_energy_consumption(self) -> None:
        time_moving = self.dist_moved / config.UAV_SPEED
        fly_energy = config.POWER_MOVE * time_moving + config.POWER_HOVER * (config.TIME_SLOT_DURATION - time_moving)
        self.energy_current_slot += fly_energy
