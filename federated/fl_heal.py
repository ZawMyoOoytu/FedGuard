import numpy as np

class FLHealer:

    def aggregate(self, trust, attack_score):

        weights = np.array(trust) * np.exp(-attack_score)
        weights = weights / (weights.sum() + 1e-8)

        return np.mean(weights), weights.tolist()