from ray import tune
from ray.tune.registry import register_env
import ray

from cartpole_custom import CartPoleEnv
ray.shutdown()  # Stop all processes from ray, in case any were present
ray.init(
  num_cpus=4,  # Make use of 4 cpus inside my computer
  include_dashboard=False,
  ignore_reinit_error=True,
  log_to_driver=False,
)
# False: Basic CartPole environment | True: Target (adjusted) CartPole environment
target = False

checkpoint_path = "C:/Users/RobbeVR/ray_results/Sim2Real"
register_env("CartPole-v0", lambda env_config: CartPoleEnv(target))
stop = {"episode_reward_mean": 195}  # Run until mean reward is close to 200, as stated in assignment
config = {
    "num_workers": 3,
    "env": "CartPole-v0",
    "framework": "torch",
    "lr": 0.0001,
    "logger_config": None,
}

analysis = tune.run(
  "PPO",
  checkpoint_freq=1,  # Checkpoints at each run, saves at ./results/test_experiment/
  config=config,
  checkpoint_at_end=True,
  stop=stop,
  local_dir="./results",  # Save data at this dir
  name="base_model"
)


trial = analysis.get_best_logdir("episode_reward_mean", "max")
checkpoint = analysis.get_best_checkpoint(
  trial,
  "training_iteration",
  "max",
)
print(f'The best model is found at: {checkpoint}')
