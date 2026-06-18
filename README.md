# FedGuard

.env
 .env
# =========================
# LLM API Keys
# =========================
#OPENAI_API_KEY=
#ANTHROPIC_API_KEY=sk-ant-...
#GEMINI_API_KEY=
# =========================
# Strategy Settings
# =========================
LLM_STRATEGY=hybrid
USE_LIVE_FOR_CRITICAL=true
CRITICAL_TRUST_THRESHOLD=0.25
CRITICAL_ATTACK_THRESHOLD=0.35

# =========================
# Cache Settings
# =========================
ENABLE_LLM_CACHE=true
CACHE_TTL=3600  # 1 hour
REDIS_URL=redis://localhost:6379

# =========================
# Fallback Settings
# =========================
LLM_TIMEOUT=5  # seconds
MAX_RETRIES=3
