from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np
from typing import List


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: List[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: List[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0

    @property
    def time_step(self) -> int:
        """Get current time step."""
        return self._time_step

    # Temporary properties, can be removed later if not needed
    @property
    def uavs(self) -> List[UAV]:
        return self._uavs

    @property
    def ues(self) -> List[UE]:
        return self._ues

    def _update_positions(self, actions: List[np.ndarray]) -> None:
        """Update positions of UAVs and UEs."""
        for uav, action in zip(self._uavs, actions):
            uav.update_position(action[:2])
        for ue in self._ues:
            ue.update_position()

    def step(self, actions: List[np.ndarray]) -> None:
        """Execute one time step of the simulation."""
        for uav in self._uavs:
            uav.reset_for_time_slot()

        self._time_step += 1
        self._update_positions(actions)

        for ue in self._ues:
            ue.generate_request()

        for uav in self._uavs:
            uav.set_current_requested_files(self._ues)
            uav.select_collaborator(uav.get_neighbors(self._uavs))

        for uav in self._uavs:
            uav.set_freq_counts()

        for uav in self._uavs:
            uav.process_requests()

        for ue in self._ues:
            ue.update_service_coverage(self._time_step)

        for uav in self._uavs:
            uav.update_ema_and_cache()
            uav.update_energy_consumption()

        # MARL model used at this point

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

    # def get_obs(self) -> List[np.ndarray]:
    #     """
    #     Gets the local observation for each UAV agent.
    #     As per the paper, this includes:
    #     - Own position
    #     - Positions of neighbor UAVs
    #     - Caching status of self and neighbors
    #     - (Simplified) Aggregate info about associated UEs: count and total requested data size.
    #     """
    #     all_obs: List[np.ndarray] = []
    #     for uav in self._uavs:
    #         # 1. Own Position (normalized)
    #         own_pos_obs = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT])

    #         # 2. Neighbor Information (positions and cache status)
    #         neighbors = uav.get_neighbors(self._uavs)
    #         neighbor_positions = np.zeros((config.NUM_UAVS - 1, 2))
    #         neighbor_caches = np.zeros((config.NUM_UAVS - 1, config.NUM_FILES))

    #         # Pad with zeros if fewer neighbors than max
    #         for i, neighbor in enumerate(neighbors):
    #             if i < config.NUM_UAVS - 1:
    #                 neighbor_positions[i] = neighbor.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT])
    #                 neighbor_caches[i] = neighbor.cache

    #         # 3. Associated UE Information (simplified for fixed-size observation)
    #         associated_ue_count = len(uav.current_covered_ues)
    #         total_req_size = sum(ue.current_request[1] for ue in uav.current_covered_ues)
    #         ue_info = np.array([associated_ue_count, total_req_size])

    #         # 4. Own Cache Status
    #         own_cache_obs = uav.cache.astype(np.float32)

    #         # Concatenate all parts to form the final observation vector
    #         obs = np.concatenate([own_pos_obs, neighbor_positions.flatten(), neighbor_caches.flatten(), ue_info, own_cache_obs]).astype(np.float32)

    #         all_obs.append(obs)

    #     # Determine and set the observation dimension in the config if not already set
    #     if not hasattr(config, "OBS_DIM_SINGLE"):
    #         config.OBS_DIM_SINGLE = len(all_obs[0])

    #     return all_obs

    # def get_reward(self) -> float:
    #     """Calculates the global reward based on the system state."""
    #     total_latency: float = sum(ue.latency for ue in self._ues)
    #     total_energy: float = sum(uav.energy for uav in self._uavs)
    #     sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues if ue.service_coverage > 0])
    #     sum_sc: float = np.sum(sc_metrics)
    #     sum_sq_sc: float = np.sum(sc_metrics**2)
    #     jfi: float = (sum_sc**2) / (len(sc_metrics) * sum_sq_sc) if sum_sq_sc > 0 else 0.0
    #     reward: float = -(config.ALPHA_1 * total_latency + config.ALPHA_2 * total_energy - config.ALPHA_3 * jfi)
    #     return reward
