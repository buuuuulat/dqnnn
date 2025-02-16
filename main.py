from lib import wrappers, dqn_model

import argparse
import time
import numpy as np
import collections
import ale_py
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state']
)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, np.uint8), np.array(next_states))


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state, info = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state])
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, terminated or truncated, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if terminated or truncated:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def calc_loss(self, batch, net, tgt_net, device='cpu'):
        states, actions, rewards, dones, next_states = batch
        states_v = torch.as_tensor(np.array(states)).to(device)
        next_states_v = torch.as_tensor(np.array(next_states)).to(device)
        actions_v = torch.as_tensor(actions).to(device)
        rewards_v = torch.as_tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)
        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mps', default=False, action='store_true', help='Enable mps')
    parser.add_argument('--env', default=DEFAULT_ENV_NAME,
                        help='Name of the environment, default=' + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device('mps' if args.mps else 'cpu')

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment='-' + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            m_reward = np.mean(total_rewards[-100:])
            print(f'{frame_idx}: done {len(total_rewards)} games, reward {m_reward}, eps {epsilon}, speed {speed}')
            writer.add_scalar('epsilon', epsilon, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('reward_100', m_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                env_name = os.path.basename(args.env)  # Extract the base name
                torch.save(net.state_dict(), env_name + '-best_%.0f.dat' % m_reward)
                if best_m_reward is not None:
                    print('Best reward updated %.3f -> %.3f' % (best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print(f'Solved in {frame_idx} frames!')
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = agent.calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
