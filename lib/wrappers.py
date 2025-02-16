import cv2
import gymnasium as gym
import collections
import numpy as np

class FireResetEnv(gym.Wrapper):
    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs1, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs2, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs2, info  # Возвращаем актуальный info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = np.max(np.stack([self._obs_buffer[-1], obs]), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    # wrappers.py (ускорение ProcessFrame84)
    @staticmethod
    def process(frame):
        # Быстрая обрезка и ресайз
        img = frame[34:194]  # Обрезать 80% экрана (только игровое поле)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.reshape(84, 84, 1)


# wrappers.py
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        self.n_steps = n_steps
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.repeat(old_space.low, n_steps, axis=0).reshape(n_steps * old_space.shape[0], *old_space.shape[1:]),
            high=np.repeat(old_space.high, n_steps, axis=0).reshape(n_steps * old_space.shape[0], *old_space.shape[1:]),
            dtype=dtype
        )

    def reset(self, **kwargs):
        self.buffer = np.zeros((self.n_steps, *self.env.observation_space.shape), dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        for i in range(self.n_steps-1):  # Инициализируем буфер первым кадром
            self.buffer[i] = obs
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = observation
        return self.buffer.reshape(-1, 84, 84)

class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)          # Output shape: (84, 84, 1)
    env = ImageToPytorch(env)          # Convert to (1, 84, 84)
    env = BufferWrapper(env, 4)        # Now shape: (4, 84, 84)
    env = ScaledFloatFrame(env)
    return env
