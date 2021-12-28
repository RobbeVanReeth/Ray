from ray import tune
from ray.tune.registry import register_env

from cartpole_custom import CartPoleEnv

# False: Basic CartPole environment | True: Target (adjusted) CartPole environment
target = True

register_env("CartPole-v0-custom", lambda env_config: CartPoleEnv(target)) 

tune.run(
    "PPO",
    # Input your checkpoint path here!
    restore="C:/Users/Robbe boss/results/Domain_rando_model/"
            "PPO_Multi-CartPole-v0_adf9d_00000_0_2021-11-22_19-54-54/checkpoint_000017/checkpoint-17",
    verbose=0,
    config={
        "env": "CartPole-v0-custom",
        "num_workers": 0,
        "render_env": True,
        "framework": "torch",
        "rollout_fragment_length": 5e5,
        "train_batch_size": 5e5,
    }
)