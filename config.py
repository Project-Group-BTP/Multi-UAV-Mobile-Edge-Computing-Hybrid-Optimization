import numpy as np

# Training Parameters
NUM_EPISODES: int = 500
MODEL: str = "mappo"  # Options: 'maddpg', 'matd3', 'mappo', 'random', 'greedy'
SAVE_FREQ = NUM_EPISODES // 10
if NUM_EPISODES < 1000:
    SAVE_FREQ = 100
BATCH_SIZE: int = 32
STEPS_PER_EPISODE: int = 1000  # Total T
LOG_FREQ: int = 10
IMG_FREQ: int = 100  # save image every 100 episodes
LEARN_FREQ: int = 5  # learn every 5 steps

RESUME_DIRECTORY: str = ""  # path to saved model directory to resume training from

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
ZIPF_BETA: float = 0.8
K_CPU: float = 1e-9  # CPU capacitance coefficient
T_CACHE_UPDATE_INTERVAL: int = 10  # T_cache
GDSF_SMOOTHING_FACTOR: float = 0.5  # beta^gdsf

# Communication Parameters
G_CONSTS_PRODUCT: float = 2.2846 * 1e-3  # G_0 * g_0
TRANSMIT_POWER: float = 0.1  # P in Watts
AWGN: float = 1e-10  # sigma^2
BANDWIDTH_INTER: int = 20 * 10**6  # B^inter in Hz
BANDWIDTH_EDGE: int = 20 * 10**6  # B^edge in Hz
BANDWIDTH_BACKHAUL: int = 50 * 10**6  # B^backhaul in Hz

# -- MARL Parameters --

ALPHA_1 = 0.4  # for latency
ALPHA_2 = 0.4  # for energy
ALPHA_3 = 0.2  # for fairness
assert ALPHA_1 + ALPHA_2 + ALPHA_3 == 1.0

# Model Parameters

OBS_DIM: int = 5
ACTION_DIM: int = 2
MLP_HIDDEN_DIM: int = 160

LEARNING_RATE: float = 0.001  # alpha
DISCOUNT_FACTOR: float = 0.99  # gamma
UPDATE_FACTOR: float = 0.01  # tau

# MADDPG Specific Hyperparameters
REPLAY_BUFFER_SIZE: int = 1_000_000  # B
MAX_GRAD_NORM: float = 1.0  # for gradient clipping
INITIAL_RANDOM_STEPS: int = 10  # steps of random actions for exploration


EPSILON: float = 1e-9

# Gaussian Noise Parameters
INITIAL_NOISE_SCALE: float = 1.0
MIN_NOISE_SCALE: float = 0.1
NOISE_DECAY_RATE: float = 0.995

# --- MATD3 Specific Hyperparameters ---
# The 'd' parameter from the paper: update policy and targets every 'd' critic updates.
POLICY_UPDATE_FREQ = 2

# The standard deviation 'sigma' for the noise added to target policy actions.
TARGET_POLICY_NOISE = 0.2

# The clipping value 'c' for the target policy noise.
NOISE_CLIP = 0.5

# --- MAPPO Specific Hyperparameters ---
LOG_STD_MAX = 2
LOG_STD_MIN = -20
# Learning rates for the Actor and Critic networks
PPO_ACTOR_LR = 3e-4
PPO_CRITIC_LR = 1e-3

# The number of training epochs to run on the collected rollout data
PPO_EPOCHS = 10

# The size of mini-batches to use during the PPO update step
PPO_MINIBATCH_SIZE = 64

# The clipping parameter (epsilon) for the PPO surrogate objective function
PPO_CLIP_EPS = 0.2

# The coefficient for the entropy bonus, encouraging exploration
PPO_ENTROPY_COEF = 0.01

# The coefficient for the value function loss in the total loss calculation
PPO_VALUE_COEF = 0.5

PPO_MAX_GRAD_NORM = 0.5  # for gradient clipping
