import gym
import random
from cartpole_dr import CartPoleEnv


class MultiEnv(gym.Env):

    def __init__(self, env_config):
        self.gravity = None
        self.cart_mass = None
        self.pole_mass = None
        self.force = None
        self.env_config = env_config

        self.randomizeParameters()
        self.printInfo()

        # Create new Cartpole instance with specified parameters here
        self.env = CartPoleEnv(self.gravity, self.cart_mass, self.pole_mass, self.force)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def randomizeParameters(self):
        # Randomize your parameters here!
        self.gravity = random.randint(1, 20)  # From the moon to mars!
        self.cart_mass = random.random() * 3
        self.pole_mass = random.random() * 3
        self.force = random.randint(1, 100)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def printInfo(self):
        print(
            f"Starting environment for worker {self.env_config.worker_index} with: gravity {self.gravity} "
            f"| cart mass: {self.cart_mass} | pole mass: {self.pole_mass} | force: {self.force}")
