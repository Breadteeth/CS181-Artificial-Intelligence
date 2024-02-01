class TrainInformation:
    def __init__(self):
        self._average = 0.0
        self._best_reward = float('-inf')
        self._best_average = float('-inf')
        self._rewards = []
        self._average_range = 100
        self._index = 0
        self._new_best_counter = 0

    def _get_best_reward(self):
        return self._best_reward

    def _get_best_average(self):
        return self._best_average

    def _get_average(self):
        avg_range = -self._average_range
        rewards_slice = self._rewards[avg_range:]
        return sum(rewards_slice) / len(rewards_slice) if rewards_slice else 0.0

    def _get_index(self):
        return self._index

    def _get_new_best_counter(self):
        return self._new_best_counter

    best_reward = property(_get_best_reward)
    best_average = property(_get_best_average)
    average = property(_get_average)
    index = property(_get_index)
    new_best_counter = property(_get_new_best_counter)

    def update_best_counter(self):
        self._new_best_counter += 1

    def _update_best_reward(self, episode_reward):
        if episode_reward > self._best_reward:
            self._best_reward = episode_reward
            return True
        return False

    def _update_best_average(self):
        if self._average > self._best_average:
            self._best_average = self._average
            return True
        return False

    def update_rewards(self, episode_reward):
        self._rewards.append(episode_reward)
        x = self._update_best_reward(episode_reward)
        y = self._update_best_average()
        if x or y:
            self.update_best_counter()
        return x or y

    def update_index(self):
        self._index += 1
