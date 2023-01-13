""""
    Some important packages:
    - pytorch: 1.13.0
    - gym-super-mario-bros
    - stable-baselines3[extra]
"""

import gym_super_mario_bros as mario
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

import numpy as np
from gym.wrappers import TimeLimit, FrameStack

from stable_baselines3 import PPO  # model to train
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # wrapper vectorized env for multi processing
from stable_baselines3.common.callbacks import BaseCallback  # callbacks to create checkpoints and stuffs
from stable_baselines3.common.env_util import make_vec_env  # function to create vec env from wrapper
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results  # plotting stuffs
from stable_baselines3.common.utils import set_random_seed  # RNG fer da win
from stable_baselines3.common.monitor import Monitor  # monitor performance in envs
from stable_baselines3.common.vec_env import VecMonitor  # monitor fer vec envs
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame  # some wrappers to preprocess observation
import os


# Callback to save best model, taken from documentations
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, save_freq=100000):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # save model every 100k steps
            if self.n_calls > 0:
                self.model.save(os.path.join(self.log_dir, f'model_step_{self.n_calls}'))

        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True


class TrainMarioUtil:
    def __init__(self, policy="MlpPolicy", model_name="MarioPPO", mario_world="-1-1",
                 log_dir="logs", tensorboard="board", learning_rate=0.00003,
                 n_steps=2048, batch_size=512, num_env=1, multiprocess=False, device="auto", n_epochs=8, gamma=0.99,
                 ent_coef=0.001, clip_range=0.17):

        self.log_dir = log_dir
        self.tensorboard = tensorboard
        self.model_name = model_name
        self.mario_world = mario_world
        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.num_env = num_env
        self.multiprocess = multiprocess
        self.device = device
        os.makedirs(log_dir, exist_ok=True)

    def make_env(self, rank, seed=0):
        """
        Utility function for multiprocessed env.
        Return a callable to generate environments with random seed with specified rank
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """

        def _init():
            env = mario.make(f"SuperMarioBros{self.mario_world}-v0")
            # wrap env in joypad space
            env = JoypadSpace(env, COMPLEX_MOVEMENT)
            # skip 4 frames
            env = MaxAndSkipEnv(env, 4)
            # gray scale + change to squared frames
            env = WarpFrame(env)
            env = FrameStack(env, num_stack=4)
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    def train_model(self, steps=1000000, save_freq=50000, log_freq=1000):
        # Create the vectorized environment
        # env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(cores)]), "logs/TestMonitor")

        # Create normal env if number of envs is not specified
        if self.num_env == 0:
            env = Monitor(env=self.make_env(rank=1)(),
                          filename=os.path.join(self.log_dir, self.model_name, "TestMonitor"))
        else:
            if self.multiprocess:
                env = VecMonitor(SubprocVecEnv([self.make_env(i) for i in range(self.num_env)]),
                                 os.path.join(self.log_dir, self.model_name, "TestMonitor"))
            else:
                env = VecMonitor(DummyVecEnv([self.make_env(i) for i in range(self.num_env)]),
                                 os.path.join(self.log_dir, self.model_name, "TestMonitor"))

        #     define model
        model = PPO(self.policy,
                    env=env,
                    learning_rate=self.learning_rate,
                    verbose=1,
                    tensorboard_log=f"./{self.tensorboard}/",
                    n_epochs=self.n_epochs,
                    gamma=self.gamma,
                    batch_size=self.batch_size,
                    clip_range=self.clip_range,
                    ent_coef=self.ent_coef,
                    device=self.device
                    )

        print("------- Start training process ---------")
        callback = SaveOnBestTrainingRewardCallback(check_freq=log_freq, save_freq=save_freq,
                                                    log_dir=os.path.join(self.log_dir, self.model_name))
        model.learn(total_timesteps=steps, callback=callback, tb_log_name=self.model_name)
        model.save(os.path.join("trained_models", self.model_name))
        print("------------- Done Learning -------------")

    def cont_train(self, steps=1000000, save_freq=50000, log_freq=1000):
        # Create normal env
        if self.num_env == 1:
            env = Monitor(env=self.make_env(rank=1)(),
                          filename=os.path.join(self.log_dir, f"{self.model_name}_CONT", "TestMonitor"))
        else:
            if self.multiprocess:
                env = VecMonitor(SubprocVecEnv([self.make_env(i) for i in range(self.num_env)]),
                                 os.path.join(self.log_dir, f"{self.model_name}_CONT", "TestMonitor"))
            else:
                env = VecMonitor(DummyVecEnv([self.make_env(i) for i in range(self.num_env)]),
                                 os.path.join(self.log_dir, f"{self.model_name}_CONT", "TestMonitor"))
        #     define model
        model = PPO.load(os.path.join("trained_models", f"{self.model_name}.zip"), env=env)
        print("------- Start training process ---------")
        callback = SaveOnBestTrainingRewardCallback(check_freq=log_freq, save_freq=save_freq,
                                                    log_dir=f"logs/{self.model_name}_CONT")
        model.learn(total_timesteps=steps, callback=callback, tb_log_name=f"{self.model_name}_CONT")
        model.save(os.path.join("trained_models", f"{self.model_name}_CONT"))
        print("------------- Done Learning -------------")

    def run_model(self, timesteps=2000):
        # make env for testing (with same configuration as training envs)
        env = mario.make(f"SuperMarioBros{self.mario_world}-v0")
        # wrap env in joypad space
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        # skip 4 frames
        # env = MaxAndSkipEnv(env, 4)
        # gray scale + change to squared frames
        env = WarpFrame(env)
        env = FrameStack(env, num_stack=4)
        # load model
        obs = env.reset()
        model = PPO.load(f"{self.model_name}.zip", env=env)

        for _ in range(timesteps):
            action, _states = model.predict(obs)
            # print(action)
            obs, reward, done, info = env.step(int(action))
            if done:
                env.reset()
            env.render()
