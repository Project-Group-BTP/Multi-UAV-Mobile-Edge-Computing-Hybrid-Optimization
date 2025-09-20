from environment.user_equipments import UE
from environment.uavs import UAV
from environment import comm_model as comms
import config
import numpy as np
from typing import List


class Env:
    """
    The main environment class, acting as a central mediator.

    It manages the state of all UAVs and UEs, handles their interactions,
    calculates the system-wide objectives (latency, energy, fairness),
    and provides observations and rewards to the MARL agents.
    """

    def __init__(self) -> None:
        self.mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self.ues: List[UE] = [UE(i) for i in range(config.NUM_UES)]
        self.uavs: List[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self.time_step: int = 0

    def update_positions(self, actions: List[np.ndarray]) -> None:
        for uav, action in zip(self.uavs, actions):
            uav.update_position(action[:2])
        for ue in self.ues:
            ue.update_position()

    def step(self, actions: List[np.ndarray]) -> None:
        self.time_step += 1
        self.update_positions(actions)
        for ue in self.ues:
            ue.generate_request()
        for uav in self.uavs:
            uav.set_current_requested_files(self.ues)
            uav_neighbors = uav.get_neighbors(self.uavs)
            uav.select_collaborator(uav_neighbors)
        for uav in self.uavs:
            uav.set_rates()
            uav.set_current_slot_request_count()

        for uav in self.uavs:
            uav.process_requests()

        for ue in self.ues:
            ue.update_service_coverage(self.time_step)
        for uav in self.uavs:
            uav.update_energy_consumption()

