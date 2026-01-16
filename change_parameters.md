**Configs and hyperparameters that can be varied for experimentation and hyperparameter tuning (Important ones specially highlighted).**

NOTE : Please do try to design a better reward function and improve algorithms/models/buffers etc. 

- MODEL : Try across different models
- STEPS_PER_EPISODE : Can try different episode lengths
- LOG, IMG FREQS : Change as per convenience

- MBS_POS : Change MBS position to see its effect on performance
- **(IMP)** NUM_UAVS and NUM_UES : Study effect of varying number of UAVs and users

- **(IMP)** Try varying starting positions of UAVs and UEs, like concentrating UEs around **some hotspots**
- UAV_SPEED: Vary speed to vary how much area UAVs are allowed to cover

- UAV_STORAGE_CAPACITY and UAV_COMPUTING_CAPACITY : Vary capacity to see their effect on latency
- **(IMP)** NUM_SERVICES, NUM_CONTENTS : Vary number of services and contents to simulate different scenarios. Also vary their popularity distribution (can try something different from Zipf)
- CPU_CYCLES_PER_BYTE, FILE_SIZES, MIN_INPUT_SIZE, MAX_INPUT_SIZE : Vary to simulate different service requirements

Basically all the above parameters affect each other, so on changing one you may have to adjust others too.

- COLLISION_AVOIDANCE_ITERATIONS : Increase to fine-tune collision avoidance
- **(IMP)** COLLISION, BOUNDARY and NON_SERVED_LATENCY PENALTIES : Adjust their values

- **(IMP)** MAX_UAV_NEIGHBORS and MAX_ASSOCIATED_UES : Vary to see their effect on reward

- **(IMP)** T_CACHE_UPDATE_INTERVAL and GDSF_SMOOTHING_FACTOR : Tune these hyperparameters for better caching performance

- UE_BATTERY_CAPACITY and UE_CRITICAL_THRESHOLD : Vary to emulate different UE battery scenarios

MARL Hyperparameters :

Some model improvement related suggestions, can try as per requirement:
- Changing F.mse_loss to F.huber_loss in all update() functions.
- Using Running Mean/Std Normalizer
- Trying LR decay, particularly for on-policy. Adding LR scheduler.

Though you can try changing/fixing all of them, still the below are the most important ones for hyperparameter tuning:

- **(IMP)** ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4, REWARD_SCALING_FACTOR : Adjust weights for different components of reward function
- **(IMP)** MLP Structure : Change number of layers and units per layer, maybe try adding attention or other improvements. Can vary MLP_HIDDEN_DIM
- **(IMP)** Learning Rates : Experiment with different learning rates for actor and critic networks (ACTOR_LR, CRITIC_LR)
- **(IMP)** DISCOUNT_FACTOR and MAX_GRAD_NORM : Can vary their effect as well
- **(IMP)** REPLAY_BUFFER_SIZE, REPLAY_BATCH_SIZE, INITIAL_RANDOM_STEPS, LEARN_FREQ : Tune these to improve learning stability and efficiency for off-policy algorithms
- **(IMP)** MAPPO, MATD3, MASAC Specific Hyperparameters : Tune these hyperparameters specific to the chosen MARL algorithm for optimal performance
- **(IMP)** ATTN_HIDDEN_DIM, ATTN_NUM_HEADS : Current values should work fine, but can try varying them.

**IMP NOTE**: Explore normalization of reward terms by different methods. Also take a final call on reward scaling and non-served latency penalty and include them in docs.
