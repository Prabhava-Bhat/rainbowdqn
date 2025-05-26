import numpy as np
from collections import deque, namedtuple
import random
import torch

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6, n_step=3, gamma=0.99):
        self.capacity = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action", "reward", "next_state", "done"])
        
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
