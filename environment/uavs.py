import config
import numpy as np


class UAV:
    def __init__(self, uav_id: int):
        self.id = uav_id
        self.pos = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), config.UAV_ALTITUDE])
        self.computing_capacity = config.UAV_COMPUTING_CAP
        self.storage_capacity = config.UAV_STORAGE_CAP
        self.coverage_radius = config.UAV_COVERAGE_RADIUS
        self.sensing_range = config.UAV_SENSING_RANGE

        # Caching State : A numpy array of booleans indicating if a file is cached
        self.cache: np.ndarray = np.zeros(config.NUM_CONTENTS + config.NUM_SERVICES, dtype=bool)

    def update_position(self, next_pos: np.ndarray):
        """
        Updates the UAV's position to the new location chosen by the MARL agent.
        """
        self.pos = np.append(next_pos, config.UAV_ALTITUDE)

    def get_cached_files_size(self) -> float:
        """Helper to get the total size of all files in the cache."""
        return sum(self.cache.values())
