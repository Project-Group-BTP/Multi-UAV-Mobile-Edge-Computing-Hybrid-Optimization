from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def reset(self) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step = 0
        return self._get_obs()

    def step(self, actions: np.ndarray, visualize: bool = False) -> tuple[list[np.ndarray], list[float], tuple[float, float, float]]:
        """Execute one time step of the simulation."""
        self._time_step += 1

        for uav in self._uavs:
            uav.process_requests()

        for ue in self._ues:
            ue.update_service_coverage(self._time_step)

        for uav in self._uavs:
            uav.update_ema_and_cache()
            uav.update_energy_consumption()

        rewards, metrics = self._get_rewards_and_metrics()

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # For next time step
        for ue in self._ues:
            ue.update_position()

        for uav in self._uavs:
            uav.reset_for_next_step()

        if visualize:
            for uav, action in zip(self._uavs, actions):  # only for visualize script
                uav.update_position(action)
        else:
            self._apply_actions_to_env(actions)

        next_obs: list[np.ndarray] = self._get_obs()
        return next_obs, rewards, metrics

    def _get_obs(self) -> list[np.ndarray]:
        """Gets the local observation for each UAV agent."""
        # For new time step
        for ue in self._ues:
            ue.generate_request()
        self._associate_ues_to_uavs()
        for uav in self._uavs:
            uav.set_current_requested_files()
            uav.set_neighbors(self._uavs)
        for uav in self._uavs:
            uav.select_collaborator()
        for uav in self._uavs:
            uav.set_freq_counts()

        all_obs: list[np.ndarray] = []
        for uav in self._uavs:
            # Part 1: Own state (position and cache status)
            own_pos: np.ndarray = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT])
            own_cache: np.ndarray = uav.cache.astype(np.float32)
            own_state: np.ndarray = np.concatenate([own_pos, own_cache])

            # Part 2: Neighbors state (positions and cache status)
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, 2 + config.NUM_FILES))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_cache: np.ndarray = neighbor.cache.astype(np.float32)
                neighbor_states[i, :] = np.concatenate([relative_pos, neighbor_cache])

            # Part 3: State of associated UEs
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, 2 + 3))
            ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])))[: config.MAX_ASSOCIATED_UES]
            for i, ue in enumerate(ues):
                delta_pos: np.ndarray = (ue.pos[:2] - uav.pos[:2]) / config.AREA_WIDTH
                req_type, req_size, req_id = ue.current_request
                norm_id: float = float(req_id) / float(config.NUM_FILES)
                norm_size: float = float(req_size) / float(config.MAX_INPUT_SIZE)
                request_info: np.ndarray = np.array([req_type, norm_size, norm_id], dtype=np.float32)
                ue_states[i, :] = np.concatenate([delta_pos, request_info])

            # Part 4: Combine all parts into a single, flat observation vector
            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        return all_obs

    def _apply_actions_to_env(self, actions: np.ndarray) -> None:
        """Calculates next positions and resolves potential collisions iteratively."""
        current_positions: np.ndarray = np.array([uav.pos[:2] for uav in self._uavs])
        max_dist: float = config.UAV_SPEED * config.TIME_SLOT_DURATION

        # Interpret actions as a direct (x, y) vector
        delta_vec_raw: np.ndarray = np.array(actions, dtype=np.float32)

        # Calculate the magnitude (distance) of this raw vector
        raw_magnitude: np.ndarray = np.linalg.norm(delta_vec_raw, axis=1, keepdims=True)

        # Clip the magnitude to be at most 1.0
        clipped_magnitude: np.ndarray = np.minimum(raw_magnitude, 1.0)
        distances: np.ndarray = clipped_magnitude * max_dist
        denom: np.ndarray = raw_magnitude + float(config.EPSILON)
        directions: np.ndarray = delta_vec_raw / denom
        delta_pos: np.ndarray = directions * distances

        proposed_positions: np.ndarray = current_positions + delta_pos

        min_boundary_gap: float = config.UAV_COVERAGE_RADIUS / 2.0
        for i, uav in enumerate(self._uavs):
            if not (min_boundary_gap <= proposed_positions[i, 0] <= config.AREA_WIDTH - min_boundary_gap and min_boundary_gap <= proposed_positions[i, 1] <= config.AREA_HEIGHT - min_boundary_gap):
                uav.boundary_violation = True
        next_positions: np.ndarray = np.clip(proposed_positions, [min_boundary_gap, min_boundary_gap], [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])

        min_sep_sq: float = config.MIN_UAV_SEPARATION**2
        for _ in range(config.COLLISION_AVOIDANCE_ITERATIONS + 1):
            collision_detected_in_iter: bool = False
            for i in range(config.NUM_UAVS):
                for j in range(i + 1, config.NUM_UAVS):
                    pos_i: np.ndarray = next_positions[i]
                    pos_j: np.ndarray = next_positions[j]
                    dist_sq: float = np.sum((pos_i - pos_j) ** 2)
                    if dist_sq < min_sep_sq:
                        self._uavs[i].collision_violation = True
                        self._uavs[j].collision_violation = True
                        collision_detected_in_iter = True
                        dist: float = np.sqrt(dist_sq) if dist_sq > 0 else config.EPSILON
                        overlap: float = config.MIN_UAV_SEPARATION - dist
                        direction: np.ndarray = (pos_i - pos_j) / dist
                        next_positions[i] += direction * overlap * 0.5
                        next_positions[j] -= direction * overlap * 0.5
            if not collision_detected_in_iter:
                break

        final_positions: np.ndarray = np.clip(next_positions, [min_boundary_gap, min_boundary_gap], [config.AREA_WIDTH - min_boundary_gap, config.AREA_HEIGHT - min_boundary_gap])
        for i, uav in enumerate(self._uavs):
            uav.update_position(final_positions[i])

    def _associate_ues_to_uavs(self) -> None:
        """Assigns each UE to at most one UAV, resolving overlaps by choosing the closest UAV."""
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2]))
                if distance <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance))

            if not covering_uavs:
                continue
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True

    def _get_rewards_and_metrics(self) -> tuple[list[float], tuple[float, float, float]]:
        """Returns the reward and other metrics."""
        total_latency: float = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues)
        total_energy: float = sum(uav.energy for uav in self._uavs)
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))

        r_fairness: float = config.ALPHA_3 * np.log(jfi + config.EPSILON)
        r_latency: float = config.ALPHA_1 * np.log(total_latency + config.EPSILON)
        r_energy: float = config.ALPHA_2 * np.log(total_energy + config.EPSILON)
        reward: float = r_fairness - r_latency - r_energy
        rewards: list[float] = [reward] * config.NUM_UAVS
        for uav in self._uavs:
            if uav.collision_violation:
                rewards[uav.id] -= config.COLLISION_PENALTY
            if uav.boundary_violation:
                rewards[uav.id] -= config.BOUNDARY_PENALTY
        rewards = [r * config.REWARD_SCALING_FACTOR for r in rewards]
        return rewards, (total_latency, total_energy, jfi)
