# =========================
# FIX: System Stability + OpenBLAS + Reproducibility
# =========================
import os
import sys
import warnings
import gc
import json
import random
import time
import numpy as np
import torch
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Thread control (IMPORTANT for OpenBLAS crash fix)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

# Reduce fragmentation risk
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

gc.collect()
load_dotenv()

# =========================
# Imports (project modules)
# =========================
from config import Config
from rl.dqn import DQNAgent, FederatedLearning, BlockchainLedger
from env.spectrum_env import SpectrumEnvironment
from llm.safe_space import LLMSafeSpace
from utils.logger import Logger


# =========================
# Seed for reproducibility
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# SAFE TRUST HELPER (IMPORTANT FIX)
# =========================
def safe_trust(state):
    """Avoid missing trust crash"""
    if isinstance(state, dict) and "trust" in state:
        return float(state["trust"])
    return 0.0


# =========================
# MAIN
# =========================
def main():

    Config.print_config()
    set_seed(Config.SEED)

    config_dict = {
        'state_dim': Config.STATE_SIZE,
        'action_dim': Config.ACTION_SPACE,
        'max_steps': Config.MAX_STEPS,
        'num_clients': Config.NUM_CLIENTS,
        'fl_frequency': Config.FL_ROUNDS,
        'gamma': Config.GAMMA,
        'learning_rate': Config.LEARNING_RATE,
        'epsilon_start': Config.EPSILON_START,
        'epsilon_min': Config.EPSILON_MIN,
        'epsilon_decay': Config.EPSILON_DECAY,
        'batch_size': Config.BATCH_SIZE,
        'memory_size': Config.MEMORY_SIZE,
        'target_update': Config.TARGET_UPDATE,
        'device': Config.DEVICE,
        'trust_critical_threshold': Config.TRUST_CRITICAL_THRESHOLD,
        'attack_high_threshold': Config.ATTACK_HIGH_THRESHOLD,
        'safe_action_ratio': Config.SAFE_ACTION_RATIO,
        'live_llm_enabled': Config.LIVE_LLM_ENABLED,
        'local_llm_enabled': Config.LOCAL_LLM_ENABLED,
        'max_retries': Config.MAX_RETRIES,
    }

    # =========================
    # Core modules
    # =========================
    logger = Logger(log_dir=Config.LOG_DIR)
    blockchain = BlockchainLedger()
    
    # ✅ Logger ကို Blockchain နဲ့ ချိတ်ဆက်မယ်
    logger.set_blockchain(blockchain)
    
    fl = FederatedLearning(config_dict)
    fl.ledger = blockchain
    env = SpectrumEnvironment(config_dict)
    llm = LLMSafeSpace(config_dict)

    print("=" * 60)
    print("SYSTEM INITIALIZED")
    print("=" * 60)

    # =========================
    # Create agents (SAFE)
    # =========================
    agents = []
    for i in range(Config.NUM_CLIENTS):
        try:
            agents.append(
                DQNAgent(
                    state_dim=Config.STATE_SIZE,
                    action_dim=Config.ACTION_SPACE,
                    config=config_dict
                )
            )
        except Exception as e:
            print(f"Agent init failed {i}: {e}")

    if len(agents) == 0:
        print("NO AGENTS CREATED ❌")
        return

    # =========================
    # Tracking
    # =========================
    rewards_hist = []
    trust_hist = []

    llm_stats = {"local": 0, "live": 0, "cache": 0, "fallback": 0}

    episodes = min(Config.EPISODES, 20)

    # =========================
    # TRAIN LOOP
    # =========================
    for ep in range(episodes):

        ep_rewards = []
        ep_trusts = []

        # ✅ Episode စတဲ့အခါ Log ထည့်မယ်
        logger.log_blockchain({
            'event': 'episode_start',
            'episode': ep,
            'timestamp': time.time()
        })

        for agent_id, agent in enumerate(agents):

            state = env.reset(agent_id)
            total_reward = 0

            for step in range(Config.MAX_STEPS):

                try:
                    action, _ = agent.act(state)
                    dqn_action = action
                    llm_override = False

                    # LLM safety gate
                    if not llm.validate_action(state, action):
                        rec = llm.get_recommendation(state)
                        action = rec["action"]
                        llm_override = True

                        # ✅ Override Log ထည့်မယ်
                        logger.log_blockchain({
                            'event': 'action_override',
                            'episode': ep,
                            'step': step,
                            'agent_id': agent_id,
                            'dqn_action': int(dqn_action),
                            'llm_action': int(action),
                            'reason': rec.get('reason', 'unsafe_action'),
                            'provider': rec.get('provider', 'local'),
                            'trust_score': float(safe_trust(state))
                        })

                        provider = rec.get("provider", "local")
                        llm_stats["local" if provider == "local" else "live"] += 1

                        if rec.get("cache_hit"):
                            llm_stats["cache"] += 1
                        if rec.get("fallback_used"):
                            llm_stats["fallback"] += 1

                    next_state, reward, done = env.step(action, agent_id)

                    # ✅ Step Log ထည့်မယ် (ဒီမှာ Data အကုန်ပါတယ်)
                    logger.log_blockchain({
                        'event': 'step_executed',
                        'episode': ep,
                        'step': step,
                        'agent_id': agent_id,
                        'action': int(action),              # ✅ action
                        'reward': float(reward),            # ✅ reward
                        'trust_score': float(safe_trust(state)),  # ✅ trust_score
                        'llm_override': bool(llm_override)  # ✅ llm_override
                    })

                    agent.train(state, action, reward, next_state, done)

                    state = next_state
                    total_reward += reward

                    if done:
                        break

                except Exception as e:
                    print(f"Step error: {e}")
                    break

            ep_rewards.append(total_reward)
            ep_trusts.append(safe_trust(state))

            # ✅ Agent Episode Log ထည့်မယ်
            logger.log_blockchain({
                'event': 'agent_episode_end',
                'episode': ep,
                'agent_id': agent_id,
                'total_reward': float(total_reward),
                'trust_score': float(safe_trust(state))
            })

        # =========================
        # Episode stats
        # =========================
        avg_r = np.mean(ep_rewards)
        avg_t = np.mean(ep_trusts)

        rewards_hist.append(avg_r)
        trust_hist.append(avg_t)

        # ✅ Episode End Log ထည့်မယ်
        logger.log_blockchain({
            'event': 'episode_end',
            'episode': ep,
            'avg_reward': float(avg_r),
            'avg_trust': float(avg_t),
            'llm_stats': llm_stats.copy()
        })

        # FL update
        if ep % Config.FL_ROUNDS == 0 and ep > 0:
            updates = [a.get_federated_update() for a in agents if a]
            updates = [u for u in updates if u]

            if len(updates) > 0:
                global_w = fl.aggregate(updates)

                for a in agents:
                    a.apply_federated_update(global_w)

                print(f"[FL] Round {fl.aggregation_count}")

                # ✅ FL Round Log ထည့်မယ်
                logger.log_blockchain({
                    'event': 'fl_round',
                    'round': fl.aggregation_count,
                    'episode': ep,
                    'num_updates': len(updates)
                })

        # GC fix
        if ep % 5 == 0:
            gc.collect()

        # =========================
        # PRINT
        # =========================
        if ep % 5 == 0:
            print(f"\nEpisode {ep}")
            print(f"Reward: {avg_r:.3f}")
            print(f"Trust:  {avg_t:.3f}")
            print(f"LLM: {llm_stats}")

    # =========================
    # FINAL REPORT
    # =========================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Reward: {rewards_hist[0]:.3f} → {rewards_hist[-1]:.3f}")
    print(f"Trust:  {trust_hist[0]:.3f} → {trust_hist[-1]:.3f}")

    print("\nLLM usage:", llm_stats)

    # ✅ Blockchain Summary
    blockchain_data = logger.get_blockchain_data()
    print(f"\nBlockchain Records: {len(blockchain_data)}")
    if len(blockchain_data) > 0:
        overrides = [b for b in blockchain_data if b.get('event') == 'action_override']
        print(f"  - Action Overrides: {len(overrides)}")
        episodes_logged = [b for b in blockchain_data if b.get('event') == 'episode_end']
        print(f"  - Episodes Logged: {len(episodes_logged)}")
        if len(blockchain_data) > 0:
            last = blockchain_data[-1]
            print(f"  - Last Event: {last.get('event', 'N/A')} (Episode {last.get('episode', 'N/A')})")

    # Save
    logger.save_results()

    os.makedirs("models", exist_ok=True)
    agents[0].save("models/final_model.pth")

    print("\nDONE ✅")


if __name__ == "__main__":
    main()