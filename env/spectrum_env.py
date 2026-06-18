import numpy as np
from typing import Dict, Tuple, Optional
from config import Config

class SpectrumEnvironment:
    """
    Spectrum sensing environment with multi-agent support
    For CLSD_FRLS: Collaborative Learning for Spectrum Defense
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize spectrum environment
        
        Args:
            config: Configuration dictionary (optional, uses Config class if None)
        """
        # Use Config class if no config provided
        if config is None:
            config = {
                'state_dim': Config.STATE_SIZE,
                'action_dim': Config.ACTION_SPACE,
                'max_steps': Config.MAX_STEPS,
                'num_clients': Config.NUM_CLIENTS
            }
        
        self.config = config
        self.step_count = 0
        self.episode_count = 0
        
        # Spectrum parameters
        self.num_channels = Config.NUM_CHANNELS
        self.channel_states = np.ones(self.num_channels, dtype=np.float32)
        
        # Multi-agent tracking
        self.agents = {}
        self.trust_levels = {}
        self.attack_probs = {}
        self.agent_steps = {}
        
        # Attack parameters
        self.attack_prob_increase = Config.ATTACK_PROB_INCREASE
        self.max_attack_prob = Config.MAX_ATTACK_PROB
        
        # Trust parameters
        self.min_trust = Config.MIN_TRUST
        self.max_trust = Config.MAX_TRUST
        self.trust_attack_penalty = Config.TRUST_ATTACK_PENALTY
        self.trust_defense_bonus = Config.TRUST_DEFENSE_BONUS
        
        # History for richer state
        self.trust_history = {}
        self.attack_history = {}
        self.reward_history = {}
        
        # Statistics
        self.total_attacks = 0
        self.successful_attacks = 0
        
    def reset(self, agent_id: int) -> Dict:
        """
        Reset environment for specific agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Initial state dictionary
        """
        # Initialize agent if new
        if agent_id not in self.trust_levels:
            # Use initial trust from config or default
            initial_trust = Config.INITIAL_TRUST[agent_id % len(Config.INITIAL_TRUST)]
            self.trust_levels[agent_id] = initial_trust
            self.attack_probs[agent_id] = 0.1
            self.agent_steps[agent_id] = 0
            self.trust_history[agent_id] = [initial_trust]
            self.attack_history[agent_id] = [0.1]
            self.reward_history[agent_id] = []
        
        # Reset for this episode
        self.agent_steps[agent_id] = 0
        self.channel_states = np.ones(self.num_channels, dtype=np.float32)
        self.step_count = 0
        
        return self.get_state(agent_id)
    
    def get_state(self, agent_id: int) -> Dict:
        """
        Get current state for agent with enhanced features
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            State dictionary with obs and metadata
        """
        trust = self.trust_levels.get(agent_id, 0.5)
        attack = self.attack_probs.get(agent_id, 0.1)
        
        # Get history for richer state
        trust_history = self.trust_history.get(agent_id, [trust])
        attack_history = self.attack_history.get(agent_id, [attack])
        
        # Calculate trends
        trust_trend = trust - np.mean(trust_history[-5:]) if len(trust_history) >= 5 else 0
        attack_trend = attack - np.mean(attack_history[-5:]) if len(attack_history) >= 5 else 0
        
        # Defense effectiveness
        defense_active = np.mean(self.channel_states)
        
        # Build enhanced state (7 dimensions)
        obs = np.concatenate([
            [trust],                          # 1. Current trust level
            [attack],                         # 2. Current attack probability
            self.channel_states,              # 3-7. Channel states (5 channels)
        ])
        
        # Return state with metadata
        return {
            'obs': obs.astype(np.float32),
            'mask': self.channel_states.copy().astype(np.float32),
            'trust': trust,
            'attack_prob': attack,
            'trust_trend': trust_trend,
            'attack_trend': attack_trend,
            'defense_active': defense_active,
            'agent_id': agent_id,
            'step': self.agent_steps.get(agent_id, 0)
        }
    
    def step(self, action: int, agent_id: int = 0) -> Tuple[Dict, float, bool]:
        """
        Execute action and return next state
        
        Args:
            action: Action to execute (0-5)
                0: All defenses on
                1-5: Turn off specific defense channel
            agent_id: Agent identifier
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        self.step_count += 1
        self.agent_steps[agent_id] = self.agent_steps.get(agent_id, 0) + 1
        
        # =========================
        # 1. UPDATE SPECTRUM BASED ON ACTION
        # =========================
        if action == 0:
            # All channels on (full defense)
            self.channel_states = np.ones(self.num_channels, dtype=np.float32)
        elif 1 <= action <= self.num_channels:
            # Turn off specific channel
            self.channel_states = np.ones(self.num_channels, dtype=np.float32)
            self.channel_states[action - 1] = 0
        else:
            # Invalid action - keep current state
            pass
        
        # =========================
        # 2. SIMULATE ATTACK
        # =========================
        defense_effectiveness = np.mean(self.channel_states)
        attack_prob = self.attack_probs.get(agent_id, 0.1)
        
        # Attack success probability depends on defense and attack probability
        attack_success_prob = attack_prob * (1 - defense_effectiveness)
        attack_success = np.random.random() < attack_success_prob
        
        self.total_attacks += 1
        if attack_success:
            self.successful_attacks += 1
        
        # =========================
        # 3. UPDATE TRUST
        # =========================
        trust = self.trust_levels.get(agent_id, 0.5)
        
        if attack_success:
            # Attack successful - trust decreases
            trust = max(self.min_trust, trust - self.trust_attack_penalty)
        else:
            # Attack failed - trust increases based on defense
            trust_bonus = self.trust_defense_bonus * defense_effectiveness
            trust = min(self.max_trust, trust + trust_bonus)
        
        self.trust_levels[agent_id] = trust
        
        # =========================
        # 4. UPDATE ATTACK PROBABILITY
        # =========================
        # Attack probability increases over time
        current_attack = self.attack_probs.get(agent_id, 0.1)
        new_attack = min(
            self.max_attack_prob,
            current_attack + self.attack_prob_increase * (1 - defense_effectiveness)
        )
        self.attack_probs[agent_id] = new_attack
        
        # =========================
        # 5. UPDATE HISTORY
        # =========================
        if agent_id not in self.trust_history:
            self.trust_history[agent_id] = []
            self.attack_history[agent_id] = []
            self.reward_history[agent_id] = []
        
        self.trust_history[agent_id].append(trust)
        self.attack_history[agent_id].append(new_attack)
        
        # Keep history limited
        if len(self.trust_history[agent_id]) > 20:
            self.trust_history[agent_id].pop(0)
        if len(self.attack_history[agent_id]) > 20:
            self.attack_history[agent_id].pop(0)
        
        # =========================
        # 6. CALCULATE REWARD
        # =========================
        reward = self._calculate_reward(
            trust=trust,
            defense_effectiveness=defense_effectiveness,
            attack_success=attack_success,
            agent_id=agent_id
        )
        
        self.reward_history[agent_id].append(reward)
        if len(self.reward_history[agent_id]) > 20:
            self.reward_history[agent_id].pop(0)
        
        # =========================
        # 7. CHECK DONE CONDITION
        # =========================
        max_steps = self.config.get('max_steps', Config.MAX_STEPS)
        done = (
            self.agent_steps[agent_id] >= max_steps or
            trust <= self.min_trust  # Episode ends if trust is too low
        )
        
        # =========================
        # 8. UPDATE AGENT'S STATE IN ENVIRONMENT
        # =========================
        # Update attack probability for all agents (slightly)
        for aid in self.attack_probs:
            if aid != agent_id:
                self.attack_probs[aid] = min(
                    self.max_attack_prob,
                    self.attack_probs[aid] + 0.002
                )
        
        return self.get_state(agent_id), reward, done
    
    def _calculate_reward(
        self, 
        trust: float, 
        defense_effectiveness: float, 
        attack_success: bool,
        agent_id: int
    ) -> float:
        """
        Calculate reward based on current state
        
        Returns:
            Reward value
        """
        # Get reward weights from Config
        reward_weights = Config.get_reward_weights()
        
        # 1. Trust-based reward (main component)
        # Higher trust = higher reward
        trust_reward = (trust - 0.5) * 2.0  # Range: -1.0 to 1.0
        
        # 2. Defense cost (penalty for turning off defenses)
        # Turning off defenses has a small penalty
        defense_penalty = -(1 - defense_effectiveness) * 0.1
        
        # 3. Attack outcome - FIXED: Initialize both variables
        # Attack successful = bad (negative reward)
        # Attack failed = good (positive reward)
        if attack_success:
            attack_outcome_reward = -0.5  # Penalty for successful attack
            attack_outcome_penalty = 0.0
        else:
            attack_outcome_reward = 0.2   # Bonus for failed attack
            attack_outcome_penalty = 0.0
        
        # 4. Trust bonuses/penalties
        trust_high_bonus = 0.5 if trust > 0.7 else 0.0
        trust_low_penalty = -0.3 if trust < 0.3 else 0.0
        
        # 5. Defense effectiveness bonus
        defense_bonus = 0.3 if defense_effectiveness > 0.8 else 0.0
        
        # 6. Combined reward - FIXED: All variables are defined
        reward = (
            trust_reward +
            defense_penalty +
            attack_outcome_reward +
            trust_high_bonus +
            trust_low_penalty +
            defense_bonus
        )
        
        # Clip reward to reasonable range
        return float(np.clip(reward, -2.0, 2.0))
    
    def render(self, step: int, episode: int = 0):
        """
        Print environment state
        
        Args:
            step: Current step
            episode: Current episode
        """
        # Get first agent's state for display
        agent_id = next(iter(self.trust_levels.keys())) if self.trust_levels else 0
        trust = self.trust_levels.get(agent_id, 0.5)
        attack = self.attack_probs.get(agent_id, 0.1)
        
        print(f"\nSTEP {step}")
        print(f"Reward: {trust - 0.5:.3f}")
        print(f"Trust: {trust:.3f}")
        print(f"Attack: {attack:.3f}")
        print(f"Mask: {self.channel_states}")
        print("-" * 28)
    
    def get_agent_trust(self, agent_id: int) -> float:
        """Get trust level for specific agent"""
        return self.trust_levels.get(agent_id, 0.5)
    
    def get_agent_attack_prob(self, agent_id: int) -> float:
        """Get attack probability for specific agent"""
        return self.attack_probs.get(agent_id, 0.1)
    
    def get_statistics(self) -> Dict:
        """Get environment statistics"""
        return {
            'total_attacks': self.total_attacks,
            'successful_attacks': self.successful_attacks,
            'attack_success_rate': (
                self.successful_attacks / self.total_attacks 
                if self.total_attacks > 0 else 0
            ),
            'num_agents': len(self.trust_levels),
            'avg_trust': np.mean(list(self.trust_levels.values())) if self.trust_levels else 0,
            'avg_attack_prob': np.mean(list(self.attack_probs.values())) if self.attack_probs else 0,
            'defense_active': np.mean(self.channel_states)
        }
    
    def close(self):
        """Clean up environment"""
        pass


# =========================
# TESTING
# =========================
if __name__ == "__main__":
    # Test the environment
    env = SpectrumEnvironment()
    
    print("Testing Spectrum Environment")
    print("=" * 40)
    
    # Test single agent
    agent_id = 0
    state = env.reset(agent_id)
    print(f"Initial state: trust={state['trust']:.2f}, attack={state['attack_prob']:.2f}")
    
    total_reward = 0
    for step in range(10):
        # Random action
        action = np.random.randint(0, 6)
        next_state, reward, done = env.step(action, agent_id)
        total_reward += reward
        
        print(f"Step {step}: action={action}, reward={reward:.2f}, "
              f"trust={next_state['trust']:.2f}, done={done}")
        
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Statistics: {env.get_statistics()}")
    print("=" * 40)
    print("Environment test complete!")