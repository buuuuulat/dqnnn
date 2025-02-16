import cv2
import gymnasium as gym
import collections
import numpy as np

class FireResetEnv(gym.Wrapper):
    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        # Начальный ресет, чтобы получить корректное начальное состояние
        obs, info = self.env.reset(**kwargs)
        # Выполняем действие "FIRE" (1)
        step_result = self.env.step(1)
        if len(step_result) == 4:
            obs1, reward, done, info = step_result
            terminated, truncated = done, False
        else:
            obs1, reward, terminated, truncated, info = step_result
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        # Выполняем следующее действие (обычно 2)
        step_result = self.env.step(2)
        if len(step_result) == 4:
            obs2, reward, done, info = step_result
            terminated, truncated = done, False
        else:
            obs2, reward, terminated, truncated, info = step_result
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs2, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                term, trunc = done, False
            else:
                obs, reward, term, trunc, info = step_result
            total_reward += reward
            self._obs_buffer.append(obs)
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
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

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            raise ValueError("Unknown resolution.")
        # Преобразуем в оттенки серого
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=old_space.low.repeat(n_steps, axis=0),
            high=old_space.high.repeat(n_steps, axis=0),
            dtype=dtype
        )

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

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
    env = ProcessFrame84(env)
    env = ImageToPytorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env
