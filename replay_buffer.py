import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.index = 0
        self.capacity = int(capacity)

        self.actions = np.empty((self.capacity,), dtype=np.int32)
        self.states = np.empty((self.capacity,) + state_shape, dtype=np.float32)
        self.states_ = np.empty((self.capacity,) + state_shape, dtype=np.float32)
        self.rewards = np.empty((self.capacity,), dtype=np.float32)
        self.dones = np.empty((self.capacity,), dtype=np.float32)

    def store_transition(self, state, action, state_, reward, done):
        self.states[self.index % self.capacity] = state
        self.states_[self.index % self.capacity] = state_
        self.actions[self.index % self.capacity] = action
        self.rewards[self.index % self.capacity] = reward
        self.dones[self.index % self.capacity] = 1 - np.float32(done)

        self.index += 1

    def sample_batch(self, batch_size):
        if self.index < batch_size:
            return None

        indexes = np.random.choice(min([self.index, self.capacity]), batch_size)

        s = self.states[indexes]
        s_ = self.states_[indexes]
        a = self.actions[indexes]
        r = self.rewards[indexes]
        d = self.dones[indexes]

        return s, a, s_, r, d
