import numpy as np

class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self._buffer = []
        self._position = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)
        self._capacity = capacity
        self._alpha = alpha

    def __len__(self):
        len_buffer = len(self._buffer)
        return len_buffer

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self._priorities[idx] = prio
   
    def sample(self, batch_size, beta=0.4):
        prios = self._priorities if len(self._buffer) == self._capacity else self._priorities[:self._position]
        probs = (prios ** self._alpha) / (prios ** self._alpha).sum()
        indices = np.random.choice(len(self._buffer), batch_size, p=probs)
        samples = []
        for idx in indices:
            samples.append(self._buffer[idx])
        weights = np.array((len(self._buffer) * probs[indices]) ** (-beta) / 
                           ((len(self._buffer) * probs[indices]) ** (-beta)).max(), dtype=np.float32)

        batch = list(zip(*samples))
        # states, actions, rewards, next_states, dones = [np.concatenate(batch[i]) for i in range(5)]
        dones = batch[4]
        actions = batch[1]
        rewards = batch[2]
        states = np.concatenate(batch[0])
        next_states = np.concatenate(batch[3])
        return states, actions, rewards, next_states, dones, indices, weights

    def push(self, state, action, reward, next_state, done):
        len_buffer = len(self._buffer)
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        batch = (state, action, reward, next_state, done)
        if self._buffer:
            max_prio = self._priorities.max()
        else:
            max_prio = 1.0
        self._priorities[self._position] = max_prio
        # self._buffer[self._position] = batch if len_buffer >= self._capacity else self._buffer.append(batch)
        if len_buffer >= self._capacity:
            self._buffer[self._position] = batch
        else:
            self._buffer.append(batch)
        self._position = (self._position + 1) % self._capacity