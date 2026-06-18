import numpy as np

class HybridTrustSystem:
    """
    Central Trust Engine (Single Source of Truth)
    """

    def __init__(self):
        self.history = []

    def compute(self,
                attack=0.0,
                node_health=1.0,
                recovery=0.5,
                llm_confidence=0.5,
                anomaly=0.0):

        # MAIN TRUST FORMULA (balanced + sensitive)
        trust = (
            0.55 * (1.0 - attack) +
            0.20 * node_health +
            0.15 * recovery +
            0.05 * llm_confidence -
            0.05 * anomaly
        )

        # clamp
        trust = float(np.clip(trust, 0.0, 1.0))

        # store history (for graphs / paper)
        self.history.append(trust)

        return trust

    def trend(self, window=10):
        if len(self.history) < window:
            return None
        return np.mean(self.history[-window:])

    def reset(self):
        self.history = []