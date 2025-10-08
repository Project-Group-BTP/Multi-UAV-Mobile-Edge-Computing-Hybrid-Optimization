# pending
# scale_actions, apply_actions_to_env
# add penalty per uav

from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self.reset()

    def reset(self) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0

        # Set initial requested files and collaborators for observation
        for uav in self._uavs:
            uav.set_current_requested_files(self._ues)
            uav.select_collaborator(uav.get_neighbors(self._uavs))

        return self._get_obs()

    @property
    def time_step(self) -> int:
        """Get current time step."""
        return self._time_step

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def _update_positions(self, actions: np.ndarray) -> None:
        """Update positions of UAVs and UEs."""
        for uav, action in zip(self._uavs, actions):
            uav.update_position(action[:2])
        for ue in self._ues:
            ue.update_position()

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], list[float], tuple[float, float, float]]:
        """
        Execute one time step of the simulation.
        Returns:
            A tuple of (next_observations, rewards, done_flag).
        """
        for uav in self._uavs:
            uav.reset_for_time_slot()
        self._time_step += 1

        scaled_actions = self._scale_actions(actions)
        # self._apply_actions_to_env(scaled_actions)
        # self._update_positions(scaled_actions)

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

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        next_obs: list[np.ndarray] = self._get_obs()
        rewards, metrics = self._get_rewards_and_metrics()
        return next_obs, rewards, metrics

    def _get_obs(self) -> list[np.ndarray]:
        """
        Gets the local observation for each UAV agent.
        """
        all_obs: list[np.ndarray] = []
        for uav in self._uavs:
            # 1. Own Position (normalized)
            own_pos_obs = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT])

            # 2. Neighbor Information (positions and cache status)
            neighbors = uav.get_neighbors(self._uavs)
            neighbor_positions = np.zeros((config.NUM_UAVS - 1, 2))
            neighbor_caches = np.zeros((config.NUM_UAVS - 1, config.NUM_FILES))

            # Pad with zeros if fewer neighbors than max
            for i, neighbor in enumerate(neighbors):
                if i < config.NUM_UAVS - 1:
                    neighbor_positions[i] = neighbor.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT])
                    neighbor_caches[i] = neighbor.cache

            # 3. Associated UE Information (simplified for fixed-size observation)
            associated_ue_count = len(uav.current_covered_ues)
            total_req_size = sum(ue.current_request[1] for ue in uav.current_covered_ues if ue.current_request)
            ue_info = np.array([associated_ue_count, total_req_size])

            # 4. Own Cache Status
            own_cache_obs = uav.cache.astype(np.float32)

            # Concatenate all parts to form the final observation vector
            obs = np.concatenate([own_pos_obs, neighbor_positions.flatten(), neighbor_caches.flatten(), ue_info, own_cache_obs]).astype(np.float32)

            all_obs.append(obs)

        # Determine and set the observation dimension in the config if not already set
        # if not hasattr(config, "OBS_DIM_SINGLE"):
        #     config.OBS_DIM_SINGLE = len(all_obs[0])
        #     config.STATE_DIM = config.NUM_UAVS * config.OBS_DIM_SINGLE

        return all_obs
    
    def _scale_actions(self, actions: np.ndarray) -> list[np.ndarray]:
        """
        Scales the actions from the actor's output range [-1, 1] to the environment's
        expected range [0, AREA_WIDTH] and [0, AREA_HEIGHT].
        This also enforces the max speed constraint.
        """
        scaled_actions = []
        # This assumes a fixed uav_speed for all UAVs, which is the case in our config
        max_dist = config.UAV_SPEED * config.TIME_SLOT_DURATION

        for action in actions:
            # Action is expected to be [delta_x, delta_y] in range [-1, 1]
            # We scale this to represent a vector of movement
            move_vec = action * max_dist
            scaled_actions.append(move_vec)

        return scaled_actions

    def _apply_actions_to_env(self, unscaled_actions: np.ndarray) -> list[np.ndarray]:
        """
        Applies the scaled actions to the environment state to get the next positions.
        This function respects the area boundaries.
        """
        current_positions = np.array([uav.pos[:2] for uav in self._uavs])
        scaled_moves = self._scale_actions(unscaled_actions)

        next_positions = current_positions + scaled_moves

        # Clip to ensure UAVs stay within the area boundaries
        next_positions[:, 0] = np.clip(next_positions[:, 0], 0, config.AREA_WIDTH)
        next_positions[:, 1] = np.clip(next_positions[:, 1], 0, config.AREA_HEIGHT)

        # Simple collision avoidance: if too close, don't move. A more sophisticated
        # method could be used, but this is a start.
        for i in range(len(next_positions)):
            for j in range(i + 1, len(next_positions)):
                if np.linalg.norm(next_positions[i] - next_positions[j]) < config.MIN_UAV_SEPARATION:
                    # On collision, revert the second UAV to its original position
                    next_positions[j] = current_positions[j]

        return [pos for pos in next_positions]


    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float]]:
        """Returns the global reward and other metrics."""
        total_latency: float = sum(ue.latency for ue in self._ues)
        total_energy: float = sum(uav.energy for uav in self._uavs)
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues if ue.service_coverage > 0])
        sum_sc: float = np.sum(sc_metrics)
        sum_sq_sc: float = np.sum(sc_metrics**2)
        jfi: float = (sum_sc**2) / (len(sc_metrics) * sum_sq_sc) if len(sc_metrics) > 0 and sum_sq_sc > 0 else 0.0
        reward: float = -(config.ALPHA_1 * total_latency + config.ALPHA_2 * total_energy - config.ALPHA_3 * jfi)
        rewards = [reward] * config.NUM_UAVS
        self._apply_penalties(rewards)
        return rewards, (total_latency, total_energy, jfi)

    def _apply_penalties(self, rewards: list[float]) -> None:
        """Applies penalties to the rewards based on certain conditions."""
        pass
        # for i, reward in enumerate(rewards):
        #     if self._ues[i].latency > config.LATENCY_THRESHOLD:
        #         rewards[i] -= config.LATENCY_PENALTY
        #     if self._uavs[i].energy > config.ENERGY_THRESHOLD:
        #         rewards[i] -= config.ENERGY_PENALTY
        # return rewards
