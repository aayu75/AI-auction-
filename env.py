import numpy as np

class AuctionEnv:
    def __init__(self, auctions=None, n_agents=5):
        self.auctions = auctions
        self.n_agents = n_agents

    def reset(self):
        if self.auctions:
            self.current = np.random.choice(self.auctions)
            self.bids = self.current["bids"]
            self.valuations = self.current["valuations"]
        else:
            self.valuations = np.random.uniform(50, 200, self.n_agents)
            self.bids = self.valuations - np.random.uniform(0, 30, self.n_agents)

        return self._get_state()

    def _get_state(self):
        return np.array([
            np.max(self.bids),
            np.mean(self.bids),
            len(self.bids)
        ], dtype=np.float32)

    def step(self, action):
        bids_sorted = np.sort(self.bids)
        highest = bids_sorted[-1]
        second = bids_sorted[-2] if len(bids_sorted) > 1 else highest

        w1, w2 = action
        payment = w1 * highest + w2 * second

        winner_index = np.argmax(self.bids)
        true_winner = np.argmax(self.valuations)

        efficiency = 1 if winner_index == true_winner else 0
        reward = payment + 10 * efficiency

        return self._get_state(), reward, highest, second, efficiency
