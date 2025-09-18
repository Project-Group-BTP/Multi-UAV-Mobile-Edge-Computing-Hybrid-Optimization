from .user_equipments import UE
from .uavs import UAV
from . import comm_model as comms
import config
import numpy as np
from typing import Dict, List


class UAVEnv:
    """
    The main environment class, acting as a central mediator.

    It manages the state of all UAVs and UEs, handles their interactions,
    calculates the system-wide objectives (latency, energy, fairness),
    and provides observations and rewards to the MARL agents.
    """

    def __init__(self):
        self.mbs_pos = config.MBS_POS
        UE.initialize_ue_class()
        self.ues = [UE(i) for i in range(config.NUM_UES)]
        self.uavs = [UAV(i) for i in range(config.NUM_UAVS)]

        self.time_step = 0
