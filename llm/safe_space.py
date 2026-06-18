# llm/safe_space.py
import numpy as np
import logging
from typing import Dict, Optional

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from .unified_llm import UnifiedLLM, LLMStrategy, LLMProvider, LLMResponse

class LLMSafeSpace:
    """LLM Safe Space with Hybrid Strategy"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize Unified LLM
        self.llm = UnifiedLLM({
            'strategy': LLMStrategy.HYBRID,
            'live_llm_enabled': self.config.get('live_llm_enabled', True),
            'live_llm_provider': self.config.get('live_llm_provider', 'groq'),
            'live_llm_api_key': self.config.get('live_llm_api_key'),
            'local_llm_enabled': True,
            'enable_cache': self.config.get('enable_cache', True),
            'critical_trust_threshold': self.config.get('trust_critical_threshold', 0.25),
            'critical_attack_threshold': self.config.get('attack_high_threshold', 0.35),
            'use_live_for_critical': True,
            'llm_timeout': 5,
            'max_retries': 3
        })
        
        # Original thresholds (for validation)
        self.trust_critical_threshold = self.config.get('trust_critical_threshold', 0.25)
        self.attack_high_threshold = self.config.get('attack_high_threshold', 0.35)
        self.safe_action_ratio = self.config.get('safe_action_ratio', 0.3)
        
        # Defensive actions
        self.defensive_actions = [0, 1, 2]
        
        logger.info("[LLMSafeSpace] Initialized with Hybrid LLM")
    
    def validate_action(self, state: Dict, action: int) -> bool:
        """Validate if action is safe"""
        trust = state.get('trust', 0.5)
        attack = state.get('attack_prob', 0.1)
        
        # Rule 1: Critical trust
        if trust < self.trust_critical_threshold:
            return action == 0
        
        # Rule 2: High attack risk
        if attack > self.attack_high_threshold:
            return action in self.defensive_actions
        
        return True
    
    def get_recommendation(self, state: Dict) -> Dict:
        """Get recommendation using hybrid LLM"""
        
        # Get from unified LLM
        try:
            response = self.llm.get_recommendation(state)
        except Exception as e:
            logger.error(f"[LLMSafeSpace] LLM error: {e}")
            # Fallback to safe action
            response = self._get_safe_fallback(state)
        
        # Validate response (safety check)
        if not self.validate_action(state, response.action):
            # Override with safe action
            logger.warning(f"[LLMSafeSpace] Overriding unsafe action {response.action}")
            response = self._get_safe_fallback(state)
        
        return {
            'action': response.action,
            'reason': response.reason,
            'provider': response.provider.value,
            'model': response.model,
            'confidence': response.confidence,
            'cache_hit': response.cache_hit,
            'latency_ms': response.latency_ms,
            'fallback_used': response.fallback_used if hasattr(response, 'fallback_used') else False
        }
    
    def _get_safe_fallback(self, state: Dict):
        """Get safe fallback action"""
        trust = state.get('trust', 0.5)
        attack = state.get('attack_prob', 0.1)
        
        if trust < self.trust_critical_threshold:
            action = 0
            reason = "🛡️ SAFETY OVERRIDE: Critical trust! All defenses ON."
        elif attack > self.attack_high_threshold:
            action = 0
            reason = "🛡️ SAFETY OVERRIDE: High attack! All defenses ON."
        else:
            action = 0
            reason = "🛡️ SAFETY OVERRIDE: Defaulting to all defenses ON."
        
        return LLMResponse(
            action=action,
            reason=reason,
            provider=LLMProvider.LOCAL,
            model="safety-override",
            confidence=1.0,
            fallback_used=True
        )
    
    def get_stats(self) -> Dict:
        """Get LLM statistics"""
        try:
            return self.llm.get_stats()
        except Exception as e:
            logger.error(f"[LLMSafeSpace] Error getting stats: {e}")
            return {
                'strategy': 'hybrid',
                'live_enabled': self.config.get('live_llm_enabled', False),
                'local_enabled': True,
                'critical_trust': self.trust_critical_threshold,
                'critical_attack': self.attack_high_threshold
            }


# =========================
# TESTING
# =========================
if __name__ == "__main__":
    # Test LLM Safe Space
    config = {
        'live_llm_enabled': True,
        'live_llm_provider': 'groq',
        'live_llm_api_key': 'test_key',
        'trust_critical_threshold': 0.25,
        'attack_high_threshold': 0.35
    }
    
    llm = LLMSafeSpace(config)
    
    # Test states
    test_states = [
        {'trust': 0.1, 'attack_prob': 0.5, 'mask': [1,1,1,1,1]},
        {'trust': 0.5, 'attack_prob': 0.2, 'mask': [1,1,1,1,1]},
        {'trust': 0.9, 'attack_prob': 0.05, 'mask': [1,1,1,1,1]}
    ]
    
    for state in test_states:
        print(f"\nState: trust={state['trust']}, attack={state['attack_prob']}")
        result = llm.get_recommendation(state)
        print(f"  Action: {result['action']}")
        print(f"  Reason: {result['reason'][:100]}...")
        print(f"  Provider: {result['provider']}")