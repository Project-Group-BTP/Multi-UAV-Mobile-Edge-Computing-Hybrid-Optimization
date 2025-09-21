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
            uav.update_ema_scores()
            uav.update_energy_consumption()

        # MARL model used at this point

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        total_latency = sum(ue.latency for ue in self._ues)
        total_energy = sum(uav.energy for uav in self._uavs)
        sc_metrics = np.array([ue.service_coverage for ue in self._ues])
        sum_sc = np.sum(sc_metrics)
        sum_sq_sc = np.sum(sc_metrics**2)
        jfi = (sum_sc**2) / (config.NUM_UES * sum_sq_sc) if sum_sq_sc > 0 else 0.0
