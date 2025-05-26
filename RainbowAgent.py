import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque, namedtuple
import random

# === Noisy Linear Layer ===
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features).sign() * torch.randn(self.in_features).abs().sqrt()
        epsilon_out = torch.randn(self.out_features).sign() * torch.randn(self.out_features).abs().sqrt()
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# === Rainbow Network (Dueling + Noisy) ===
class RainbowNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(RainbowNetwork, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, action_size)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for layer in self.modules():
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

# === Prioritized Replay Buffer with N-Step Returns ===
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, n_step=3, gamma=0.99):
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])
        
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for _, _, r, n_s, d in reversed(list(self.n_step_buffer)[:-1]):
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]
        experience = self.experience(state, action, reward, next_state, done)

        max_prio = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience

        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, beta=0.4, device="cpu"):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

        batch = self.experience(*zip(*experiences))

        states = torch.tensor(np.vstack(batch.state), dtype=torch.float32).to(device)
        actions = torch.tensor(np.vstack(batch.action), dtype=torch.long).to(device)
        rewards = torch.tensor(np.vstack(batch.reward), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32).to(device)
        dones = torch.tensor(np.vstack(batch.done).astype(np.uint8), dtype=torch.float32).to(device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5

    def __len__(self):
        return len(self.memory)

# === RainbowAgent ===
class RainbowAgent:
    def __init__(self, state_size, action_size, device,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99,
                 lr=1e-4, tau=1e-3, update_every=4, alpha=0.6, beta_start=0.4,
                 beta_frames=100000, n_step=3):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.n_step = n_step

        self.policy_net = RainbowNetwork(state_size, action_size).to(device)
        self.target_net = RainbowNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, alpha, n_step, gamma)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
            experiences = self.memory.sample(beta=beta, device=self.device)
            self.learn(experiences)
            self.frame += 1

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, indices, weights = experiences
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        expected_q = self.policy_net(states).gather(1, actions)
        td_errors = target_q - expected_q
        loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        self.soft_update(self.policy_net, self.target_net, self.tau)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
