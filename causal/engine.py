import numpy as np

def causal_update(state):
    """
    Causal inference module (clean version)

    OUTPUT:
        causal_bias
        causal_signal
        risk_score

    NOTE:
        DOES NOT modify reward directly
    """

    obs = state.get("obs", np.array([0.0]))
    attack = state.get("attack_level", 0.0)

    obs_mean = float(np.mean(obs))

    # =========================
    # causal risk estimation
    # =========================
    risk_score = (
        0.6 * attack +
        0.4 * (1.0 - obs_mean)
    )

    # =========================
    # causal bias (directional effect)
    # =========================
    if risk_score > 0.6:
        causal_bias = -0.4
    elif risk_score > 0.3:
        causal_bias = -0.1
    else:
        causal_bias = 0.2

    # =========================
    # causal signal (for RL)
    # =========================
    causal_signal = obs_mean - attack

    return {
        "causal_bias": causal_bias,
        "causal_signal": causal_signal,
        "risk_score": risk_score
    }