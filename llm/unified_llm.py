# llm/unified_llm.py
import os
import time
import json
import hashlib
import logging
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# JSON Encoder for numpy types
# =========================
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj

# =========================
# Enums
# =========================
class LLMProvider(Enum):
    LOCAL = "local"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"
    GROQ = "groq"

class LLMStrategy(Enum):
    LOCAL_ONLY = "local_only"
    LIVE_ONLY = "live_only"
    HYBRID = "hybrid"
    HYBRID_WITH_FALLBACK = "hybrid_with_fallback"

# =========================
# Models
# =========================
@dataclass
class LLMRequest:
    """Request to LLM"""
    state: Dict
    context: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 150

@dataclass
class LLMResponse:
    """Response from LLM"""
    action: int
    reason: str
    provider: LLMProvider
    model: str
    confidence: float = 0.8
    source: str = "llm"
    latency_ms: float = 0
    cache_hit: bool = False
    fallback_used: bool = False

# =========================
# LLM Cache
# =========================
class LLMCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: Dict[str, Dict] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        
    def _hash_request(self, state: Dict, context: str = "") -> str:
        """Create hash from request - FIXED: Convert numpy types"""
        # Convert state to serializable format
        serializable_state = convert_to_serializable(state)
        key = f"{json.dumps(serializable_state, sort_keys=True, cls=NumpyEncoder)}{context}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, state: Dict, context: str = "") -> Optional[Dict]:
        """Get cached response"""
        key = self._hash_request(state, context)
        
        if key in self.cache:
            entry = self.cache[key]
            # Check TTL
            if time.time() - entry['timestamp'] < entry['ttl']:
                self.hits += 1
                logger.debug(f"[Cache] Hit: {key[:8]}")
                return entry['response']
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, state: Dict, response: Dict, context: str = ""):
        """Cache response"""
        key = self._hash_request(state, context)
        
        # Check size limit
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest]
        
        self.cache[key] = {
            'response': response,
            'timestamp': time.time(),
            'ttl': self.ttl
        }
        logger.debug(f"[Cache] Set: {key[:8]}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0,
            'max_size': self.max_size
        }

# =========================
# Live LLM Client
# =========================
class LiveLLMClient:
    """Live LLM Client (OpenAI, Claude, Gemini, Groq)"""
    
    # Groq's currently available models (as of 2024)
    # Reference: https://console.groq.com/docs/models
    GROQ_MODELS = {
        "llama-3.1-70b-versatile": "Best quality, versatile",
        "llama-3.1-8b-instant": "Fastest, good quality",
        "llama-3.2-3b-preview": "Preview model",
        "mixtral-8x7b-32768": "Legacy (may be deprecated)",
        "gemma2-9b-it": "Google's Gemma 2"
    }
    
    def __init__(self, provider: str, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if not self.api_key:
            logger.warning(f"[LiveLLM] No API key for {provider}, using dummy")
            self.api_key = "dummy_key"
        
        self.client = None
        self.model = None
        
        try:
            if provider == "openai":
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.model = "gpt-4o-mini"
            elif provider == "claude":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.model = "claude-3-sonnet-20240229"
            elif provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            elif provider == "groq":
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                # FIXED: Use currently available Groq models
                # llama-3.1-70b-versatile - Best quality
                # llama-3.1-8b-instant - Fastest
                self.model = "llama-3.1-8b-instant"  # Fast and good quality
                # Alternative: "llama-3.1-70b-versatile" for better quality
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            logger.info(f"[LiveLLM] Initialized {provider} with {self.model}")
        except Exception as e:
            logger.warning(f"[LiveLLM] Failed to initialize {provider}: {e}")
            self.client = None
            self.model = None
    
    def get_response(self, request: LLMRequest) -> Dict:
        """Get response from live LLM"""
        if not self.client:
            raise Exception(f"LiveLLM client not initialized for {self.provider}")
        
        prompt = self._build_prompt(request.state, request.context)
        
        start_time = time.time()
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                result = response.choices[0].message.content
            
            elif self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system=self._system_prompt(),
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
            
            elif self.provider == "gemini":
                response = self.model.generate_content(
                    f"{self._system_prompt()}\n\n{prompt}",
                    generation_config={"temperature": request.temperature}
                )
                result = response.text
            
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                result = response.choices[0].message.content
            
            latency_ms = (time.time() - start_time) * 1000
            
            return self._parse_response(result, latency_ms)
            
        except Exception as e:
            logger.error(f"[LiveLLM] Error: {e}")
            raise
    
    def _system_prompt(self) -> str:
        return """You are a cybersecurity expert for spectrum defense.
        
        Given the current state (trust level, attack probability, mask status), 
        recommend the best defense action (0-5):
        - 0: All defenses ON (maximum protection)
        - 1: Turn OFF defense channel 1
        - 2: Turn OFF defense channel 2
        - 3: Turn OFF defense channel 3
        - 4: Turn OFF defense channel 4
        - 5: Turn OFF defense channel 5
        
        Rules:
        1. If trust < 0.25, ALWAYS recommend action 0
        2. If attack > 0.35, prefer defensive actions (0,1,2)
        3. If trust > 0.8 and attack < 0.2, you can optimize defenses
        
        Respond with ONLY the action number and a brief reason.
        """
    
    def _build_prompt(self, state: Dict, context: str = "") -> str:
        trust = state.get('trust', 0.5)
        attack = state.get('attack_prob', 0.1)
        mask = state.get('mask', [1,1,1,1,1])
        
        # Convert numpy types to Python types
        trust = float(trust) if isinstance(trust, (np.floating, float)) else trust
        attack = float(attack) if isinstance(attack, (np.floating, float)) else attack
        if isinstance(mask, np.ndarray):
            mask = mask.tolist()
        
        return f"""
        Current state:
        - Trust level: {trust:.2f}
        - Attack probability: {attack:.2f}
        - Defense mask: {mask}
        {context}
        
        Recommended action (0-5):
        """
    
    def _parse_response(self, response: str, latency_ms: float) -> Dict:
        """Parse LLM response"""
        import re
        
        # Extract action number
        numbers = re.findall(r'\d+', response)
        action = int(numbers[0]) if numbers else 0
        action = max(0, min(5, action))  # Clamp
        
        # Extract reason
        reason = response[:300]  # First 300 chars
        
        return {
            'action': action,
            'reason': reason,
            'model': self.model,
            'latency_ms': latency_ms,
            'raw_response': response
        }

# =========================
# Unified LLM Manager (Hybrid)
# =========================
class UnifiedLLM:
    """Unified LLM Manager with Hybrid Strategy"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Strategy
        strategy_str = self.config.get('strategy', 'hybrid')
        self.strategy = LLMStrategy(strategy_str) if isinstance(strategy_str, str) else strategy_str
        
        # Live LLM
        self.live_enabled = self.config.get('live_llm_enabled', False)
        self.live_provider = self.config.get('live_llm_provider', 'groq')
        self.live_model = self.config.get('live_llm_model', 'llama-3.1-8b-instant')
        self.live_client = None
        
        if self.live_enabled:
            try:
                self.live_client = LiveLLMClient(
                    provider=self.live_provider,
                    api_key=self.config.get('live_llm_api_key')
                )
                # Override model if specified in config
                if self.live_model and self.live_client:
                    self.live_client.model = self.live_model
                logger.info(f"[UnifiedLLM] Live LLM enabled with model: {self.live_model}")
            except Exception as e:
                logger.warning(f"[UnifiedLLM] Live LLM init failed: {e}")
                self.live_enabled = False
        
        # Local LLM (fallback)
        self.local_enabled = self.config.get('local_llm_enabled', True)
        
        # Cache
        self.cache_enabled = self.config.get('enable_cache', True)
        self.cache = LLMCache(
            max_size=self.config.get('cache_size', 1000),
            ttl=self.config.get('cache_ttl', 3600)
        ) if self.cache_enabled else None
        
        # Thresholds
        self.critical_trust = self.config.get('critical_trust_threshold', 0.25)
        self.critical_attack = self.config.get('critical_attack_threshold', 0.35)
        self.use_live_for_critical = self.config.get('use_live_for_critical', True)
        
        # Timeout
        self.timeout = self.config.get('llm_timeout', 5)
        self.max_retries = self.config.get('max_retries', 3)
        
        logger.info(f"[UnifiedLLM] Initialized with strategy: {self.strategy.value}")
    
    def get_recommendation(self, state: Dict, context: str = "") -> LLMResponse:
        """Get recommendation using hybrid strategy"""
        
        # Convert state to serializable format for caching
        serializable_state = convert_to_serializable(state)
        
        # 1. Check cache
        if self.cache_enabled and self.cache:
            cached = self.cache.get(serializable_state, context)
            if cached:
                return LLMResponse(
                    action=cached['action'],
                    reason=f"{cached['reason']} (cached)",
                    provider=LLMProvider(cached.get('provider', 'local')),
                    model=cached.get('model', 'cache'),
                    confidence=0.95,
                    cache_hit=True
                )
        
        # 2. Determine if we should use live LLM
        use_live = self._should_use_live(state)
        
        # 3. Try to get response
        if use_live and self.live_enabled and self.live_client:
            try:
                response = self._get_live_response(state, context)
                if response:
                    # Cache response
                    if self.cache_enabled and self.cache:
                        cache_data = {
                            'action': response.action,
                            'reason': response.reason,
                            'provider': response.provider.value,
                            'model': response.model
                        }
                        self.cache.set(serializable_state, cache_data, context)
                    return response
            except Exception as e:
                logger.warning(f"[UnifiedLLM] Live LLM failed: {e}")
        
        # 4. Fallback to local
        return self._get_local_response(state)
    
    def _should_use_live(self, state: Dict) -> bool:
        """Determine if live LLM should be used"""
        
        # Strategy check
        if self.strategy == LLMStrategy.LOCAL_ONLY:
            return False
        elif self.strategy == LLMStrategy.LIVE_ONLY:
            return self.live_enabled
        elif self.strategy == LLMStrategy.HYBRID_WITH_FALLBACK:
            return self.live_enabled
        elif self.strategy == LLMStrategy.HYBRID:
            if not self.use_live_for_critical:
                return False
            
            trust = state.get('trust', 0.5)
            attack = state.get('attack_prob', 0.1)
            
            # Convert to float if numpy type
            trust = float(trust) if isinstance(trust, (np.floating, float)) else trust
            attack = float(attack) if isinstance(attack, (np.floating, float)) else attack
            
            if trust < self.critical_trust:
                return True
            if attack > self.critical_attack:
                return True
            
            return False
        
        return self.live_enabled
    
    def _get_live_response(self, state: Dict, context: str) -> Optional[LLMResponse]:
        """Get response from live LLM with retries"""
        
        request = LLMRequest(
            state=state,
            context=context,
            temperature=0.3,
            max_tokens=150
        )
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.live_client.get_response(request)
                
                return LLMResponse(
                    action=response['action'],
                    reason=response['reason'],
                    provider=LLMProvider(self.live_provider),
                    model=response.get('model', self.live_client.model),
                    confidence=0.9,
                    latency_ms=response.get('latency_ms', 0)
                )
            except Exception as e:
                last_error = e
                logger.warning(f"[UnifiedLLM] Live attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Wait before retry
        
        if last_error:
            raise last_error
        
        return None
    
    def _get_local_response(self, state: Dict) -> LLMResponse:
        """Get response from local LLM (rule-based)"""
        
        trust = state.get('trust', 0.5)
        attack = state.get('attack_prob', 0.1)
        mask = state.get('mask', [1,1,1,1,1])
        
        # Convert numpy types
        trust = float(trust) if isinstance(trust, (np.floating, float)) else trust
        attack = float(attack) if isinstance(attack, (np.floating, float)) else attack
        if isinstance(mask, np.ndarray):
            mask = mask.tolist()
        
        available = [i for i, m in enumerate(mask) if m == 1]
        if not available:
            available = [0]
        
        # Local logic
        if trust < self.critical_trust:
            action = 0
            reason = "🚨 CRITICAL! Trust dangerously low (%.2f). Activating all defenses." % trust
        elif attack > self.critical_attack:
            defensive = [a for a in available if a in [0, 1, 2]]
            action = defensive[0] if defensive else 0
            reason = "⚠️ High attack probability (%.2f). Using defensive strategy." % attack
        else:
            action = available[0]
            reason = "✅ System safe. Optimizing defenses."
        
        return LLMResponse(
            action=action,
            reason=reason,
            provider=LLMProvider.LOCAL,
            model="rule-based",
            confidence=0.7,
            fallback_used=True
        )
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'strategy': self.strategy.value,
            'live_enabled': self.live_enabled,
            'live_provider': self.live_provider if self.live_enabled else 'disabled',
            'live_model': self.live_model if self.live_enabled else 'disabled',
            'local_enabled': self.local_enabled,
            'critical_trust': self.critical_trust,
            'critical_attack': self.critical_attack,
        }
        
        if self.cache_enabled and self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats