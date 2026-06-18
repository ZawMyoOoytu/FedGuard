# llm/live_llm.py
import openai
import anthropic
import google.generativeai as genai
import os
from typing import Dict, Optional

class LiveLLMClient:
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        if provider == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model = "gpt-4-turbo-preview"  # or "gpt-3.5-turbo"
        elif provider == "claude":
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model = "claude-3-opus-20240229"  # or "claude-3-sonnet"
        elif provider == "gemini":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        elif provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.model = "mixtral-8x7b-32768"  # or "llama3-70b-8192"
        
        print(f"[LiveLLM] Initialized with {provider} using {self.model}")
    
    def get_recommendation(self, state: Dict) -> Dict:
        """Get action recommendation from live LLM"""
        
        # Build prompt
        prompt = self._build_prompt(state)
        
        # Call API
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            result = response.choices[0].message.content
        
        elif self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=0.3,
                system=self._system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text
        
        elif self.provider == "gemini":
            response = self.model.generate_content(
                f"{self._system_prompt()}\n\n{prompt}",
                generation_config={"temperature": 0.3}
            )
            result = response.text
        
        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            result = response.choices[0].message.content
        
        # Parse result
        return self._parse_response(result, state)
    
    def _system_prompt(self) -> str:
        return """You are a cybersecurity expert for spectrum defense. 
        Given the current state (trust level, attack probability, mask status), 
        recommend the best action (0-5):
        - 0: All defenses ON
        - 1-5: Turn OFF specific defense channel
        Always prioritize safety over performance.
        Respond with ONLY the action number and a brief reason."""
    
    def _build_prompt(self, state: Dict) -> str:
        trust = state.get('trust', 0.5)
        attack = state.get('attack_prob', 0.1)
        mask = state.get('mask', [1,1,1,1,1])
        
        return f"""
        Current state:
        - Trust level: {trust:.2f}
        - Attack probability: {attack:.2f}
        - Defense mask: {mask}
        
        Recommend action (0-5):
        """
    
    def _parse_response(self, response: str, state: Dict) -> Dict:
        """Parse LLM response"""
        try:
            # Extract action number from response
            import re
            numbers = re.findall(r'\d+', response)
            action = int(numbers[0]) if numbers else 0
            
            # Clamp action
            action = min(5, max(0, action))
            
            return {
                'action': action,
                'reason': response[:200],  # First 200 chars as reason
                'source': self.provider,
                'model': self.model
            }
        except:
            # Fallback
            return {
                'action': 0,
                'reason': "Could not parse LLM response, defaulting to all defenses ON",
                'source': self.provider,
                'model': self.model
            }