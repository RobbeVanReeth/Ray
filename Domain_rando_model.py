from ray import tune
from ray.tune.registry import register_env
import ray
from multi_env import MultiEnv

checkpoint_path = "C:/Users/RobbeVR/ray_results/Sim2Real"

ray.shutdown()  # Stop all processes from ray, in case any were present
ray.init(
  num_cpus=4,  # Make use of 4 cpus inside my computer
  include_dashboard=False,
  ignore_reinit_error=True,
  log_to_driver=False,
)
target = True

register_env("Multi-CartPole-v0", lambda env_config: MultiEnv(env_config=env_config))
stop = {"episode_reward_mean": 195}  # Run until mean reward is close to 200, as stated in assignment
analysis = tune.run(
        "PPO",
        config={
            "num_workers": 3,
            "env": "Multi-CartPole-v0",
            "framework": "torch",
            "logger_config": None,
            "env_config": {
                "worker_index": 1
            }
        },
          checkpoint_at_end=True,
          stop=stop,
          local_dir="~/results",  # Save data at this dir
          name="Domain_rando_model"

)


trial = analysis.get_best_logdir("episode_reward_mean", "max")
checkpoint = analysis.get_best_checkpoint(
  trial,
  "training_iteration",
  "max",
)
print(f'The best model is found at: {checkpoint}')
