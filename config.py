import numpy as np

# Simulation Parameters
NUM_UAVS: int = 5
NUM_UES: int = 20
AREA_WIDTH: int = 1000  # X_max
AREA_HEIGHT: int = 1000  # Y_max
SIM_TIME_SLOTS: int = 1000  # Total T
TIME_SLOT_DURATION: int = 1  # tau in seconds
UE_MAX_DIST: int = 10

# UAV Parameters
UAV_ALTITUDE: int = 100  # H
UAV_SPEED: int = 50  # v^UAV in m/s
UAV_STORAGE_CAP: int = 10**9  # S_u in bytes (e.g., 1 GB)
UAV_COMPUTING_CAP: int = 10**9  # F_u in cycles/sec (e.g., 1 GHz)
UAV_SENSING_RANGE: float = 300.0  # R^sense
UAV_COVERAGE_RADIUS: float = 200.0  # R
MIN_UAV_SEPARATION: float = 50.0  # d_min
POWER_MOVE: float = 150.0  # P_move in Watts
POWER_HOVER: float = 100.0  # P_hover in Watts

# Communication Parameters
G_0: float = 1e-3  # G_0
g_0: float = 1e-3  # g_0
TRANSMIT_POWER: float = 0.1  # P_tx in Watts
NOISE_POWER: float = 1e-10  # sigma^2 in Watts
BANDWIDTH_INTER: int = 20 * 10**6  # B^inter in Hz
BANDWIDTH_EDGE: int = 20 * 10**6  # B^edge in Hz
BANDWIDTH_BACKHAUL: int = 50 * 10**6  # B^backhaul in Hz

# MARL Hyperparameters
LEARNING_RATE: float = 0.001
DISCOUNT_FACTOR: float = 0.99  # gamma
BUFFER_SIZE: int = 1_000_000

MIN_INPUT_SIZE: int = 1_000_000
MAX_INPUT_SIZE: int = 1_000_000_000
ZIPF_BETA: float = 0.8
NUM_SERVICES: int = 50
NUM_CONTENTS: int = 100
CPU_CYCLES_PER_BYTE: np.ndarray = np.concatenate((np.random.randint(500, 1500, size=NUM_SERVICES), np.zeros(NUM_CONTENTS)))  # C_s for services, 0 for contents
FILE_SIZES: np.ndarray = np.random.randint(1_000_000, 100_000_000, size=NUM_SERVICES + NUM_CONTENTS)  # in bytes
K_CPU : float = 1e-9 # CPU capacitance coefficient

# Caching Parameters
T_CACHE_UPDATE_INTERVAL: int = 10  # T_cache
GDSF_SMOOTHING_FACTOR: float = 0.5  # beta^gdsf

MBS_POS: np.ndarray = np.array([0.0, 0.0, 0.0])  # (X_mbs, Y_mbs, Z_mbs)


# Gaussian Noise Parameters
INITIAL_NOISE_SCALE: float = 1.0
MIN_NOISE_SCALE: float = 0.1
NOISE_DECAY_RATE: float = 0.995

# MLP
MLP_HIDDEN_DIM: int = 160
ACTION_DIM: int = 2  # 2D continuous actions (delta_x, delta_y)

# Training Parameters
NUM_EPISODES: int = 500
MAX_STEPS: int = 300
BATCH_SIZE: int = 32
LOG_FREQ: int = 10
IMG_FREQ: int = 100  # save image every 100 episodes
LEARN_FREQ: int = 5  # learn every 5 steps

RESUME_DIRECTORY: str = ""  # path to saved model directory to resume training from


# Small no
EPSILON: float = 1e-9
