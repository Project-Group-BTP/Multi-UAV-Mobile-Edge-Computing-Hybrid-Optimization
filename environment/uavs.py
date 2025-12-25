from __future__ import annotations
from environment.user_equipments import UE
from environment import comm_model as comms
import config
import numpy as np


def _get_belief_probability(file_id: int, neighbor_id: int) -> float:
    """Returns the estimated probability P_{v,i} that a neighbor has file_i."""
    rank: int = file_id
    if file_id >= config.NUM_SERVICES:
        rank = file_id - config.NUM_SERVICES
    c_hat_v: float = config.UAV_STORAGE_CAPACITY[neighbor_id] / config.AVG_FILE_SIZE
    exponent: float = config.PROB_GAMMA * (rank - c_hat_v)
    probability: float = 1.0 / (1.0 + np.exp(exponent))
    return probability


def _get_computing_latency_and_energy(uav: UAV, cpu_cycles: int) -> tuple[float, float]:
    """Calculate computing latency and energy for a UAV processing request."""
    assert uav._current_service_request_count > 0
    computing_capacity_per_request: float = config.UAV_COMPUTING_CAPACITY[uav.id] / uav._current_service_request_count
    latency: float = cpu_cycles / computing_capacity_per_request
    energy: float = config.K_CPU * cpu_cycles * (computing_capacity_per_request**2)
    return latency, energy


def _try_add_file_to_cache(uav: UAV, file_id: int) -> None:
    """Try to add a file to UAV cache if there's enough space."""
    if uav._working_cache[file_id]:
        return  # Already in cache
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
        self._current_service_request_count: int = 0
        self._energy_current_slot: float = 0.0  # Energy consumed for this time slot
        self.collision_violation: bool = False  # Track if UAV has violated minimum separation
        self.boundary_violation: bool = False  # Track if UAV has gone out of bounds

        # Cache and request tracking
        self.cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._working_cache: np.ndarray = np.zeros(config.NUM_FILES, dtype=bool)
        self._freq_counts: np.ndarray = np.zeros(config.NUM_FILES)
        self._ema_scores: np.ndarray = np.zeros(config.NUM_FILES)

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

    def reset_for_next_step(self) -> None:
        """Reset UAV state for a new step."""
        self._current_covered_ues = []
        self._neighbors = []
        self._current_service_request_count = 0
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

    def calculate_initial_load(self) -> None:
        for ue in self._current_covered_ues:
            if ue.current_request[0] == 0:  # Service
                self._current_service_request_count += 1

    def process_requests(self) -> None:
        """Process Requests using Probabilistic Decisions with Optimistic Relief."""
        self._working_cache = self.cache.copy()
        self._uav_mbs_rate = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(self.pos, config.MBS_POS))

        shuffled_indices: np.ndarray = np.random.permutation(len(self._current_covered_ues))

        for idx in shuffled_indices:
            ue: UE = self._current_covered_ues[idx]

            req_type, _, req_id = ue.current_request
            if req_type == 2:
                self._process_energy_request(ue)
                continue

            ue_uav_rate: float = comms.calculate_ue_uav_rate(comms.calculate_channel_gain(ue.pos, self.pos), len(self._current_covered_ues))

            best_target_idx, best_target_uav = self._decide_offloading_target(ue.current_request, ue_uav_rate)

            self._freq_counts[req_id] += 1  # I got a request for this file
            if best_target_idx == 1 and best_target_uav is not None:  # Request also seen by collaborating UAV
                best_target_uav._freq_counts[req_id] += 1

            if req_type == 0:
                if best_target_idx != 0:
                    # OPTIMISTIC RELIEF: I was counted in 'calculate_initial_load', but I am leaving. Decrement so next user sees smaller queue.
                    self._current_service_request_count = max(0, self._current_service_request_count - 1)
                    if best_target_idx == 1 and best_target_uav is not None:
                        best_target_uav._current_service_request_count += 1

            if req_type == 0:
                self._process_service_request(ue, ue_uav_rate, best_target_idx, best_target_uav)
            else:
                self._process_content_request(ue, ue_uav_rate, best_target_idx, best_target_uav)

    def _decide_offloading_target(self, current_req: tuple[int, int, int], ue_uav_rate: float) -> tuple[int, UAV | None]:
        """Returns (target_idx, target_uav_obj); Id: 0 = Local, 1 = Collaborating UAV, 2 = MBS"""
        req_type, req_size, req_id = current_req
        file_size: int = config.FILE_SIZES[req_id]
        cpu_cycles: int = config.CPU_CYCLES_PER_BYTE[req_id] * req_size if req_type == 0 else -1

        # Associated UAV (Local) Expected Latency
        p_local: float = 1.0 if self.cache[req_id] else 0.0
        ue_uav_upload_latency: float = req_size / ue_uav_rate  # For service
        ue_uav_download_latency: float = file_size / ue_uav_rate  # For content
        exp_fetch_latency: float = (1.0 - p_local) * (file_size / self._uav_mbs_rate)  # For both
        exp_local_latency: float = exp_fetch_latency + ue_uav_download_latency  # For content
        if req_type == 0:  # Service
            assert self._current_service_request_count > 0
            est_comp_latency: float = cpu_cycles / (config.UAV_COMPUTING_CAPACITY[self.id] / self._current_service_request_count)
            exp_local_latency = ue_uav_upload_latency + exp_fetch_latency + est_comp_latency  # Overwrite for service

        best_exp_latency: float = exp_local_latency
        best_target_idx: int = 0
        best_target_uav: UAV | None = None

        # MBS Offloading Expected Latency
        uav_mbs_download_latency: float = file_size / self._uav_mbs_rate
        exp_mbs_latency: float = uav_mbs_download_latency + ue_uav_download_latency  # For content
        if req_type == 0:
            uav_mbs_upload_latency: float = req_size / self._uav_mbs_rate
            exp_mbs_latency = ue_uav_upload_latency + uav_mbs_upload_latency  # Overwrite for service

        if exp_mbs_latency < best_exp_latency:
            best_exp_latency = exp_mbs_latency
            best_target_idx = 2

        # Collaborating UAV Expected Latency
        for neighbor in self._neighbors:
            belief_prob: float = _get_belief_probability(req_id, neighbor.id)

            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, neighbor.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(neighbor.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate
            exp_neighbor_fetch_latency: float = (1.0 - belief_prob) * (file_size / uav_mbs_rate)  # For both
            exp_neighbour_latency: float = exp_neighbor_fetch_latency + uav_uav_download_latency + ue_uav_download_latency  # For content
            if req_type == 0:  # Service
                # Neighbor Load: They broadcasted 'initial_load'. We add +1 because "If I come, I add to the pile."
                neigh_load: int = neighbor._current_service_request_count + 1
                assert neigh_load > 0
                est_comp_latency: float = cpu_cycles / (config.UAV_COMPUTING_CAPACITY[neighbor.id] / neigh_load)
                uav_uav_upload_latency: float = req_size / uav_uav_rate
                exp_neighbour_latency = ue_uav_upload_latency + uav_uav_upload_latency + exp_neighbor_fetch_latency + est_comp_latency  # Overwrite for service

            if exp_neighbour_latency < best_exp_latency:
                best_exp_latency = exp_neighbour_latency
                best_target_idx = 1
                best_target_uav = neighbor

        return best_target_idx, best_target_uav

    def _process_service_request(self, ue: UE, ue_uav_rate: float, target_idx: int, target_uav: UAV | None) -> None:
        _, req_size, req_id = ue.current_request
        assert req_id < config.NUM_SERVICES
        cpu_cycles: int = config.CPU_CYCLES_PER_BYTE[req_id] * req_size
        file_size: int = config.FILE_SIZES[req_id]

        ue_uav_upload_latency: float = req_size / ue_uav_rate
        ue.update_battery(0.0, ue_uav_upload_latency)
        if target_idx == 0:  # Associated UAV
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                _try_add_file_to_cache(self, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(self, cpu_cycles)
            ue.latency_current_request = ue_uav_upload_latency + fetch_latency + comp_latency
            self._energy_current_slot += comp_energy

        elif target_idx == 1:  # Collaborating UAV
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_upload_latency: float = req_size / uav_uav_rate

            fetch_latency: float = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                _try_add_file_to_cache(target_uav, req_id)

            comp_latency, comp_energy = _get_computing_latency_and_energy(target_uav, cpu_cycles)
            ue.latency_current_request = ue_uav_upload_latency + uav_uav_upload_latency + fetch_latency + comp_latency
            target_uav._energy_current_slot += comp_energy
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

        else:  # MBS
            uav_mbs_upload_latency: float = req_size / self._uav_mbs_rate
            ue.latency_current_request = ue_uav_upload_latency + uav_mbs_upload_latency
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

    def _process_content_request(self, ue: UE, ue_uav_rate: float, target_idx: int, target_uav: UAV | None) -> None:
        req_id: int = ue.current_request[2]
        assert req_id >= config.NUM_SERVICES
        file_size: int = config.FILE_SIZES[req_id]

        ue_uav_download_latency: float = file_size / ue_uav_rate
        ue.update_battery(0.0, ue_uav_download_latency)
        if target_idx == 0:  # Associated UAV
            fetch_latency: float = 0.0
            if not self.cache[req_id]:
                fetch_latency = file_size / self._uav_mbs_rate
                _try_add_file_to_cache(self, req_id)

            ue.latency_current_request = fetch_latency + ue_uav_download_latency

        elif target_idx == 1:  # Collaborating UAV
            assert target_uav is not None
            uav_uav_rate: float = comms.calculate_uav_uav_rate(comms.calculate_channel_gain(self.pos, target_uav.pos))
            uav_mbs_rate: float = comms.calculate_uav_mbs_rate(comms.calculate_channel_gain(target_uav.pos, config.MBS_POS))
            uav_uav_download_latency: float = file_size / uav_uav_rate

            fetch_latency: float = 0.0
            if not target_uav.cache[req_id]:
                fetch_latency = file_size / uav_mbs_rate
                _try_add_file_to_cache(target_uav, req_id)

            ue.latency_current_request = fetch_latency + uav_uav_download_latency + ue_uav_download_latency
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

        else:  # MBS
            uav_mbs_download_latency: float = file_size / self._uav_mbs_rate
            ue.latency_current_request = uav_mbs_download_latency + ue_uav_download_latency
            _try_add_file_to_cache(self, req_id)  # Since it was a miss, try to add to associated UAV's cache as well in background

    def _process_energy_request(self, ue: UE) -> None:
        """Process an emergency energy request from a UE."""
        channel_gain: float = comms.calculate_channel_gain(self.pos, ue.pos)
        harv_energy: float = config.WPT_EFFICIENCY * config.WPT_TRANSMIT_POWER * channel_gain * config.TIME_SLOT_DURATION
        ue.update_battery(harv_energy, 0.0)
        ue.latency_current_request = 0.0  # No latency deadline for energy requests

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
        has_energy_request = any(ue.current_request[0] == 2 for ue in self._current_covered_ues)
        if has_energy_request:
            self._energy_current_slot += config.WPT_TRANSMIT_POWER * config.TIME_SLOT_DURATION
