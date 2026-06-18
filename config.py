# config.py
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """CLSD_FRLS System Configuration"""
    
    # =========================
    # ENVIRONMENT SETTINGS
    # =========================
    STATE_SIZE = 7  # [trust, attack_prob, mask0, mask1, mask2, mask3, mask4]
    ACTION_SPACE = 5  # 0: all on, 1-5: turn off specific defense
    
    EPISODES = 100
    MAX_STEPS = 30
    
    # Spectrum channels
    NUM_CHANNELS = 5
    
    # =========================
    # ATTACK SETTINGS
    # =========================
    JAMMER_PROB = 0.4
    SYBIL_PROB = 0.3
    ATTACK_THRESHOLD = 0.6
    
    JAMMER_WEIGHT = 0.7
    SYBIL_WEIGHT = 0.3
    
    ATTACK_PROB_INCREASE = 0.01
    MAX_ATTACK_PROB = 0.5
    
    # =========================
    # TRUST SYSTEM
    # =========================
    INITIAL_TRUST = [0.9, 0.85, 0.8, 0.7]
    MIN_TRUST = 0.1
    MAX_TRUST = 1.0
    
    TRUST_DECAY = 0.05
    TRUST_BOOST = 0.02
    TRUST_ATTACK_PENALTY = 0.15
    TRUST_DEFENSE_BONUS = 0.05
    
    # =========================
    # LLM SAFE SPACE ENGINE
    # =========================
    HIGH_RISK_LATENCY = 2
    LOW_RISK_LATENCY = 5
    SAFE_ACTION_RATIO = 0.3
    
    TRUST_CRITICAL_THRESHOLD = 0.25
    ATTACK_HIGH_THRESHOLD = 0.35
    
    # =========================
    # HYBRID LLM SETTINGS
    # =========================
    LIVE_LLM_ENABLED = True
    LIVE_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    LIVE_LLM_API_KEY = os.getenv(f"{LIVE_LLM_PROVIDER.upper()}_API_KEY")
    
    LOCAL_LLM_ENABLED = True
    LLM_STRATEGY = "hybrid"
    USE_LIVE_FOR_CRITICAL = True
    
    ENABLE_LLM_CACHE = True
    CACHE_SIZE = 1000
    CACHE_TTL = 3600
    
    LLM_TIMEOUT = 5
    MAX_RETRIES = 3
    
    # =========================
    # DRL SETTINGS (DQN)
    # =========================
    GAMMA = 0.99
    LEARNING_RATE = 0.0005
    
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000
    TARGET_UPDATE = 100
    
    HIDDEN_LAYERS = [128, 128, 64]
    
    # =========================
    # FEDERATED LEARNING
    # =========================
    NUM_CLIENTS = 3
    FL_ROUNDS = 20
    
    ATTACK_INFLUENCE_DECAY = 1.5
    FL_AGGREGATION_WEIGHT = 0.5
    FL_MIN_CLIENTS = 2
    
    # =========================
    # BLOCKCHAIN SETTINGS
    # =========================
    ENABLE_BLOCKCHAIN = True
    STORE_CAUSAL_INFO = True
    STORE_ATTACK_INFO = True
    STORE_ACTION_TRACE = True
    
    BLOCKCHAIN_DIFFICULTY = 4
    BLOCKS_PER_ACTION = 50
    
    # =========================
    # REWARD FUNCTION WEIGHTS
    # =========================
    REWARD_SPECTRUM = 1.0
    REWARD_ATTACK_PENALTY = -1.2
    REWARD_TRUST_BONUS = 0.8
    
    REWARD_TRUST_WEIGHT = 2.0
    REWARD_DEFENSE_PENALTY = -0.05
    REWARD_ATTACK_PENALTY_WEIGHT = -1.0
    REWARD_ATTACK_FAIL_BONUS = 0.2
    REWARD_TRUST_HIGH_BONUS = 1.0
    REWARD_TRUST_LOW_PENALTY = -0.5
    
    # =========================
    # LOGGING
    # =========================
    PRINT_EVERY_STEP = True
    SAVE_LOGS = True
    LOG_DIR = "logs/"
    
    # =========================
    # SYSTEM SETTINGS
    # =========================
    SEED = 42
    DEVICE = "cpu"
    
    # =========================
    # HELPER METHODS
    # =========================
    @classmethod
    def get_state_size(cls):
        """Return state size for environment"""
        return cls.STATE_SIZE
    
    @classmethod
    def get_action_space(cls):
        """Return action space size"""
        return cls.ACTION_SPACE
    
    @classmethod
    def get_reward_weights(cls):
        """Return reward function weights as dict"""
        return {
            'spectrum': cls.REWARD_SPECTRUM,
            'attack_penalty': cls.REWARD_ATTACK_PENALTY_WEIGHT,
            'trust_bonus': cls.REWARD_TRUST_BONUS,
            'defense_penalty': cls.REWARD_DEFENSE_PENALTY,
            'attack_fail_bonus': cls.REWARD_ATTACK_FAIL_BONUS,
            'trust_high_bonus': cls.REWARD_TRUST_HIGH_BONUS,
            'trust_low_penalty': cls.REWARD_TRUST_LOW_PENALTY
        }
    
    @classmethod
    def get_trust_params(cls):
        """Return trust system parameters"""
        return {
            'initial': cls.INITIAL_TRUST,
            'min': cls.MIN_TRUST,
            'max': cls.MAX_TRUST,
            'decay': cls.TRUST_DECAY,
            'boost': cls.TRUST_BOOST,
            'attack_penalty': cls.TRUST_ATTACK_PENALTY,
            'defense_bonus': cls.TRUST_DEFENSE_BONUS
        }
    
    @classmethod
    def get_fl_params(cls):
        """Return federated learning parameters"""
        return {
            'num_clients': cls.NUM_CLIENTS,
            'rounds': cls.FL_ROUNDS,
            'aggregation_weight': cls.FL_AGGREGATION_WEIGHT,
            'min_clients': cls.FL_MIN_CLIENTS
        }
    
    @classmethod
    def get_blockchain_params(cls):
        """Return blockchain parameters"""
        return {
            'enabled': cls.ENABLE_BLOCKCHAIN,
            'difficulty': cls.BLOCKCHAIN_DIFFICULTY,
            'blocks_per_action': cls.BLOCKS_PER_ACTION,
            'store_causal': cls.STORE_CAUSAL_INFO,
            'store_attack': cls.STORE_ATTACK_INFO,
            'store_action': cls.STORE_ACTION_TRACE
        }
    
    @classmethod
    def get_llm_config(cls):
        """Return LLM configuration"""
        return {
            'live_llm_enabled': cls.LIVE_LLM_ENABLED,
            'live_llm_provider': cls.LIVE_LLM_PROVIDER,
            'live_llm_api_key': cls.LIVE_LLM_API_KEY,
            'local_llm_enabled': cls.LOCAL_LLM_ENABLED,
            'strategy': cls.LLM_STRATEGY,
            'use_live_for_critical': cls.USE_LIVE_FOR_CRITICAL,
            'critical_trust_threshold': cls.TRUST_CRITICAL_THRESHOLD,
            'critical_attack_threshold': cls.ATTACK_HIGH_THRESHOLD,
            'enable_cache': cls.ENABLE_LLM_CACHE,
            'cache_size': cls.CACHE_SIZE,
            'cache_ttl': cls.CACHE_TTL,
            'llm_timeout': cls.LLM_TIMEOUT,
            'max_retries': cls.MAX_RETRIES
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*60)
        print("CLSD_FRLS CONFIGURATION")
        print("="*60)
        print(f"State Size: {cls.STATE_SIZE}")
        print(f"Action Space: {cls.ACTION_SPACE}")
        print(f"Episodes: {cls.EPISODES}")
        print(f"Max Steps: {cls.MAX_STEPS}")
        print(f"Gamma: {cls.GAMMA}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epsilon: {cls.EPSILON_START} -> {cls.EPSILON_MIN}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Memory Size: {cls.MEMORY_SIZE}")
        print(f"Target Update: {cls.TARGET_UPDATE}")
        print(f"FL Clients: {cls.NUM_CLIENTS}")
        print(f"FL Rounds: {cls.FL_ROUNDS}")
        print(f"Blockchain: {cls.ENABLE_BLOCKCHAIN}")
        print(f"Blockchain Difficulty: {cls.BLOCKCHAIN_DIFFICULTY}")
        print(f"Trust Critical Threshold: {cls.TRUST_CRITICAL_THRESHOLD}")
        print(f"Attack High Threshold: {cls.ATTACK_HIGH_THRESHOLD}")
        print(f"Live LLM Enabled: {cls.LIVE_LLM_ENABLED}")
        print(f"Live LLM Provider: {cls.LIVE_LLM_PROVIDER}")
        print(f"LLM Strategy: {cls.LLM_STRATEGY}")
        print(f"LLM Cache: {cls.ENABLE_LLM_CACHE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Seed: {cls.SEED}")
        print("="*60)


# =========================
# TESTING
# =========================
if __name__ == "__main__":
    # Test config
    Config.print_config()
    
    print("\nReward Weights:")
    print(Config.get_reward_weights())
    
    print("\nTrust Parameters:")
    print(Config.get_trust_params())
    
    print("\nFL Parameters:")
    print(Config.get_fl_params())
    
    print("\nLLM Config:")
    print(Config.get_llm_config())