# blockchain/ledger.py

def update_trust(state):
    """
    Blockchain-style trust memory system
    - penalizes attack
    - reinforces safe behavior
    """

    attack = state.get("attack_level", 0.0)

    # trust decay logic
    penalty = attack * 0.2
    state["trust"] -= penalty

    # clamp trust
    state["trust"] = max(0.1, state["trust"])

    # trust affects reward
    state["reward"] += state["trust"] * 0.05

    return state