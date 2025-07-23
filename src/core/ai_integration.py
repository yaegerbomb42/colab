"""
AI Model Integration for Multi-Agent Development Environment
Supports multiple AI providers: OpenAI, Google Gemini, Anthropic Claude, Local models
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    """Supported AI providers"""
    MOCK = "mock"
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    LOCAL = "local"

class AIModelManager:
    """Manages different AI model integrations"""
    
    def __init__(self, provider: AIProvider = AIProvider.MOCK):
        self.provider = provider
        self.client = None
        self.model_name = None
        
        # Try to initialize the requested provider
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the AI provider"""
        try:
            if self.provider == AIProvider.GEMINI:
                self._init_gemini()
            elif self.provider == AIProvider.OPENAI:
                self._init_openai()
            elif self.provider == AIProvider.ANTHROPIC:
                self._init_anthropic()
            elif self.provider == AIProvider.LOCAL:
                self._init_local()
            else:
                self._init_mock()
        except Exception as e:
            logger.warning(f"Failed to initialize {self.provider.value} provider: {e}")
            logger.info("Falling back to mock responses")
            self.provider = AIProvider.MOCK
            self._init_mock()
    
    def _init_gemini(self):
        """Initialize Google Gemini"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel('gemini-pro')
            self.model_name = "gemini-pro"
            logger.info("Gemini AI initialized successfully")
            
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Gemini initialization failed: {e}")
    
    def _init_openai(self):
        """Initialize OpenAI"""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = "gpt-3.5-turbo"
            logger.info("OpenAI initialized successfully")
            
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI initialization failed: {e}")
    
    def _init_anthropic(self):
        """Initialize Anthropic Claude"""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = "claude-3-sonnet-20240229"
            logger.info("Anthropic Claude initialized successfully")
            
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise Exception(f"Anthropic initialization failed: {e}")
    
    def _init_local(self):
        """Initialize local model (placeholder for Llama, etc.)"""
        # This would integrate with local models like Llama
        self.model_name = "local-llama"
        logger.info("Local model support (placeholder)")
        raise NotImplementedError("Local model support not implemented yet")
    
    def _init_mock(self):
        """Initialize mock responses"""
        self.model_name = "mock"
        logger.info("Using mock AI responses for demonstration")
    
    async def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate AI response based on provider"""
        try:
            if self.provider == AIProvider.GEMINI:
                return await self._generate_gemini_response(prompt, context)
            elif self.provider == AIProvider.OPENAI:
                return await self._generate_openai_response(prompt, context)
            elif self.provider == AIProvider.ANTHROPIC:
                return await self._generate_anthropic_response(prompt, context)
            else:
                return self._generate_mock_response(prompt, context)
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return self._generate_mock_response(prompt, context)
    
    async def _generate_gemini_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Gemini"""
        try:
            # Build context-aware prompt
            full_prompt = self._build_context_prompt(prompt, context)
            
            # Generate response
            response = await asyncio.to_thread(self.client.generate_content, full_prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return self._generate_mock_response(prompt, context)
    
    async def _generate_openai_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using OpenAI"""
        try:
            # Build context-aware prompt
            full_prompt = self._build_context_prompt(prompt, context)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant working in a collaborative development environment."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return self._generate_mock_response(prompt, context)
    
    async def _generate_anthropic_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Anthropic Claude"""
        try:
            # Build context-aware prompt
            full_prompt = self._build_context_prompt(prompt, context)
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return self._generate_mock_response(prompt, context)
    
    def _build_context_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build a context-aware prompt"""
        if not context:
            return prompt
        
        context_parts = []
        
        if context.get('agent_role'):
            context_parts.append(f"Role: {context['agent_role']} agent")
        
        if context.get('current_files'):
            files_str = ", ".join(context['current_files'])
            context_parts.append(f"Current files: {files_str}")
        
        if context.get('recent_messages'):
            context_parts.append(f"Recent conversation context: {context['recent_messages']}")
        
        if context.get('file_content'):
            context_parts.append(f"Current file content: {context['file_content']}")
        
        if context_parts:
            context_str = "\\n".join(context_parts)
            return f"Context:\\n{context_str}\\n\\nUser request: {prompt}\\n\\nRespond concisely and helpfully:"
        
        return prompt
    
    def _generate_mock_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate mock response for testing"""
        prompt_lower = prompt.lower()
        
        if "create" in prompt_lower and "file" in prompt_lower:
            if "hello" in prompt_lower and "world" in prompt_lower:
                return "I'll create a Hello World program for you!"
            elif "python" in prompt_lower:
                return "I'll create a Python file with starter code!"
            else:
                return "I'll create that file for you!"
        
        elif "help" in prompt_lower:
            return "I'm here to help with coding tasks! Try asking me to create specific files or write code."
        
        elif any(word in prompt_lower for word in ["code", "write", "implement"]):
            return "I'd be happy to help with coding! What would you like me to create?"
        
        else:
            return "I understand! Let me help you with that coding task."

# Global instance
ai_manager = AIModelManager()

def set_ai_provider(provider: AIProvider):
    """Set the global AI provider"""
    global ai_manager
    ai_manager = AIModelManager(provider)

def get_ai_manager() -> AIModelManager:
    """Get the global AI manager"""
    return ai_manager
