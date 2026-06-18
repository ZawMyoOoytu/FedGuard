# security/attack.py
import numpy as np

def attack(state):
    """
    Adversarial attack module:
    - modifies mask
    - injects risk
    - affects reward
    """

    level = np.random.rand()
    state["attack_level"] = level

    # high attack scenario
    if level > 0.6:
        state["mask"] = np.array([1, 0, 0, 0, 0])  # strong disruption
        state["reward"] -= 0.5
    elif level > 0.3:
        state["mask"] = np.array([1, 1, 0, 0, 0])
        state["reward"] -= 0.2
    else:
        state["mask"] = np.ones(5)
        state["reward"] += 0.05

    return state