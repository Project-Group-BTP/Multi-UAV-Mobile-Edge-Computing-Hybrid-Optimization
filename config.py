import numpy as np

# Training Parameters
MODEL: str = "maddpg"  # options: 'maddpg', 'matd3', 'mappo', 'masac', 'random'
SEED: int = 1234  # random seed for reproducibility
STEPS_PER_EPISODE: int = 1000  # total T
LOG_FREQ: int = 10  # episodes
IMG_FREQ: int = 100  # steps
TEST_LOG_FREQ: int = 1  # episodes (for testing)
TEST_IMG_FREQ: int = 1  # steps (for testing)
LEARN_FREQ: int = 5  # steps

# Simulation Parameters
MBS_POS: np.ndarray = np.array([0.0, 0.0, 0.0])  # (X_mbs, Y_mbs, Z_mbs) in meters
NUM_UAVS: int = 10  # U
NUM_UES: int = 200  # M
AREA_WIDTH: int = 1000  # X_max in meters
AREA_HEIGHT: int = 1000  # Y_max in meters
TIME_SLOT_DURATION: int = 1  # tau in seconds
UE_MAX_DIST: int = 50  # d_max^UE in meters
UE_MAX_WAIT_TIME: int = 5  # in time slots

# UAV Parameters
UAV_ALTITUDE: int = 100  # H in meters
UAV_SPEED: int = 50  # v^UAV in m/s
UAV_STORAGE_CAPACITY: np.ndarray = np.random.choice(np.arange(1e6, 10.1e6, 1e5), size=NUM_UAVS)  # S_u in bytes (1MB to 10MB in 100KB steps)
UAV_COMPUTING_CAPACITY: np.ndarray = np.random.choice(np.arange(1e9, 10.1e9, 1e8), size=NUM_UAVS)  # F_u in cycles/sec (1GHz to 10GHz in 100M steps)
UAV_SENSING_RANGE: float = 300.0  # R^sense in meters
UAV_COVERAGE_RADIUS: float = 100.0  # R in meters

# Collision Avoidance and Penalties
COLLISION_AVOIDANCE_ITERATIONS: int = 10  # number of iterations to resolve collisions
COLLISION_PENALTY: float = 10.0  # penalty per collision
BOUNDARY_PENALTY: float = 5.0  # penalty for going out of bounds
# IMPORTANT : Reconfigurable, should try for various values including : NUM_UAVS - 1 and NUM_UES
MAX_UAV_NEIGHBORS: int = NUM_UAVS // 2
MAX_ASSOCIATED_UES: int = NUM_UES // NUM_UAVS
assert MAX_UAV_NEIGHBORS >= 1 and MAX_UAV_NEIGHBORS <= NUM_UAVS - 1
assert MAX_ASSOCIATED_UES >= 1 and MAX_ASSOCIATED_UES <= NUM_UES

MIN_UAV_SEPARATION: float = 200.0  # d_min in meters
assert UAV_COVERAGE_RADIUS * 2 <= MIN_UAV_SEPARATION

POWER_MOVE: float = 150.0  # P_move in Watts
POWER_HOVER: float = 100.0  # P_hover in Watts

# Request Parameters
NUM_SERVICES: int = 50  # S
NUM_CONTENTS: int = 100  # K
NUM_FILES: int = NUM_SERVICES + NUM_CONTENTS  # S + K
CPU_CYCLES_PER_BYTE: np.ndarray = np.random.randint(500, 1500, size=NUM_SERVICES)  # omega_s_m
FILE_SIZES: np.ndarray = np.random.randint(1_000, 1_000_000, size=NUM_FILES)  # 1KB to 1MB
MIN_INPUT_SIZE: int = 1_000  # 1KB
MAX_INPUT_SIZE: int = 1_000_000  # 1MB
ZIPF_BETA: float = 0.8  # beta^Zipf
K_CPU: float = 1e-9  # CPU capacitance coefficient

# Caching Parameters
T_CACHE_UPDATE_INTERVAL: int = 10  # T_cache
GDSF_SMOOTHING_FACTOR: float = 0.5  # beta^gdsf

# Communication Parameters
G_CONSTS_PRODUCT: float = 2.2846 * 1e-3  # G_0 * g_0
TRANSMIT_POWER: float = 0.1  # P in Watts
AWGN: float = 1e-10  # sigma^2
BANDWIDTH_INTER: int = 20 * 10**6  # B^inter in Hz
BANDWIDTH_EDGE: int = 20 * 10**6  # B^edge in Hz
BANDWIDTH_BACKHAUL: int = 50 * 10**6  # B^backhaul in Hz

# Model Parameters

ALPHA_1 = 0.4  # for latency
ALPHA_2 = 0.4  # for energy
ALPHA_3 = 0.2  # for fairness
assert round(ALPHA_1 + ALPHA_2 + ALPHA_3, 3) == 1.0

OBS_DIM_SINGLE: int = 2 + NUM_FILES + (MAX_UAV_NEIGHBORS * (2 + NUM_FILES)) + (MAX_ASSOCIATED_UES * (2 + 3))
# own state: pos (2) + cache (NUM_FILES) + Neighbors: pos (2) + cache (NUM_FILES) + UEs: pos (2) + request_tuple (3)

ACTION_DIM: int = 2  # angle, distance from [-1, 1]
STATE_DIM: int = NUM_UAVS * OBS_DIM_SINGLE
MLP_HIDDEN_DIM: int = 160

ACTOR_LR: float = 0.002
CRITIC_LR: float = 0.001
DISCOUNT_FACTOR: float = 0.99  # gamma
UPDATE_FACTOR: float = 0.01  # tau
MAX_GRAD_NORM: float = 0.5  # maximum norm for gradient clipping to prevent exploding gradients
LOG_STD_MAX: float = 2  # maximum log standard deviation for stochastic policies
LOG_STD_MIN: float = -20  # minimum log standard deviation for stochastic policies
EPSILON: float = 1e-9  # small value to prevent division by zero

# Off-policy algorithm hyperparameters
REPLAY_BUFFER_SIZE: int = 1_000_000  # B
REPLAY_BATCH_SIZE: int = 32  # minibatch size
INITIAL_RANDOM_STEPS: int = 10  # steps of random actions for exploration

# Gaussian Noise Parameters (for MADDPG and MATD3)
INITIAL_NOISE_SCALE: float = 1.0
MIN_NOISE_SCALE: float = 0.1
NOISE_DECAY_RATE: float = 0.995

# MATD3 Specific Hyperparameters
POLICY_UPDATE_FREQ: int = 2  # delayed policy update frequency
TARGET_POLICY_NOISE: float = 0.2  # standard deviation of target policy smoothing noise.
NOISE_CLIP: float = 0.5  # range to clip target policy smoothing noise

# MAPPO Specific Hyperparameters
PPO_ROLLOUT_LENGTH: int = 2048  # number of steps to collect per rollout before updating
PPO_GAE_LAMBDA: float = 0.95  # lambda parameter for GAE
PPO_EPOCHS: int = 10  # number of epochs to run on the collected rollout data
PPO_BATCH_SIZE: int = 64  # size of mini-batches to use during the update step
PPO_CLIP_EPS: float = 0.2  # clipping parameter (epsilon) for the PPO surrogate objective
PPO_ENTROPY_COEF: float = 0.01  # coefficient for the entropy bonus to encourage exploration

# MASAC Specific Hyperparameters
ALPHA_LR: float = 3e-4  # learning rate for the entropy temperature alpha
