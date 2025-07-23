"""
Enhanced LLM Agent with support for multiple AI providers
Supports Gemini, OpenAI, Anthropic, and local models
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)

class EnhancedLLMProvider:
    """Enhanced LLM provider with multiple API support"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Default to mock if no API keys
        self.provider = self._detect_provider()
        
    def _detect_provider(self) -> str:
        """Detect which AI provider to use based on available API keys"""
        if self.gemini_api_key:
            return "gemini"
        elif self.openai_api_key:
            return "openai"
        elif self.anthropic_api_key:
            return "anthropic"
        else:
            return "mock"
    
    async def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using the selected AI provider"""
        try:
            if self.provider == "gemini":
                return await self._gemini_request(prompt, context)
            elif self.provider == "openai":
                return await self._openai_request(prompt, context)
            elif self.provider == "anthropic":
                return await self._anthropic_request(prompt, context)
            else:
                return self._mock_response(prompt, context)
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return self._fallback_response(prompt)
    
    async def _gemini_request(self, prompt: str, context: str = "") -> str:
        """Make request to Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{context}\n\nUser Request: {prompt}\n\nRespond as a helpful coding assistant. Be concise and practical."
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['candidates'][0]['content']['parts'][0]['text']
                else:
                    logger.error(f"Gemini API error: {response.status}")
                    return self._fallback_response(prompt)
    
    async def _openai_request(self, prompt: str, context: str = "") -> str:
        """Make request to OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": f"You are a helpful coding assistant. {context}"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    logger.error(f"OpenAI API error: {response.status}")
                    return self._fallback_response(prompt)
    
    async def _anthropic_request(self, prompt: str, context: str = "") -> str:
        """Make request to Anthropic API"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 500,
            "messages": [
                {"role": "user", "content": f"{context}\n\n{prompt}"}
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['content'][0]['text']
                else:
                    logger.error(f"Anthropic API error: {response.status}")
                    return self._fallback_response(prompt)
    
    def _mock_response(self, prompt: str, context: str = "") -> str:
        """Generate intelligent mock responses for development"""
        prompt_lower = prompt.lower()
        
        # File creation requests
        if "create" in prompt_lower and ("file" in prompt_lower or "main.py" in prompt_lower):
            if "hello world" in prompt_lower or "hello" in prompt_lower:
                return "I'll create a main.py file with a hello world program!"
            elif "main.py" in prompt_lower:
                return "I'll create main.py for you!"
            else:
                return "I'll create that file for you!"
        
        # Help requests
        if "help" in prompt_lower:
            return "I'm here to help with coding tasks. What would you like me to create or work on?"
        
        # File editing
        if "edit" in prompt_lower:
            return "I'll edit that file for you!"
        
        # Default responses
        responses = [
            "I understand your request. Let me work on that!",
            "Got it! I'll help you with that task.",
            "I'll take care of that for you!",
            "Understood! Working on your request now."
        ]
        
        import random
        return random.choice(responses)
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when AI APIs fail"""
        prompt_lower = prompt.lower()
        
        if "create" in prompt_lower:
            return "I'll create that for you!"
        elif "help" in prompt_lower:
            return "I'm here to help with coding tasks!"
        else:
            return "I understand. Let me work on that!"

class SmartCodeGenerator:
    """Smart code generator for different file types"""
    
    @staticmethod
    def generate_hello_world(filename: str) -> str:
        """Generate hello world code based on file extension"""
        ext = filename.split('.')[-1].lower() if '.' in filename else 'py'
        
        if ext == 'py':
            return f"""#!/usr/bin/env python3
\"\"\"
{filename} - Hello World Program
\"\"\"

def main():
    print("Hello, World!")
    print("Welcome to collaborative coding!")

if __name__ == "__main__":
    main()
"""
        elif ext in ['js', 'javascript']:
            return f"""// {filename} - Hello World Program

function main() {{
    console.log("Hello, World!");
    console.log("Welcome to collaborative coding!");
}}

main();
"""
        elif ext in ['html', 'htm']:
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>Welcome to collaborative coding!</p>
</body>
</html>
"""
        elif ext == 'java':
            class_name = filename.replace('.java', '').title()
            return f"""public class {class_name} {{
    public static void main(String[] args) {{
        System.out.println("Hello, World!");
        System.out.println("Welcome to collaborative coding!");
    }}
}}
"""
        elif ext in ['c', 'cpp']:
            return f"""#include <stdio.h>

int main() {{
    printf("Hello, World!\\n");
    printf("Welcome to collaborative coding!\\n");
    return 0;
}}
"""
        else:
            return f"""// {filename} - Hello World Program

print("Hello, World!")
print("Welcome to collaborative coding!")
"""
    
    @staticmethod
    def generate_template(filename: str, template_type: str = "basic") -> str:
        """Generate code templates based on file type and template type"""
        ext = filename.split('.')[-1].lower() if '.' in filename else 'py'
        
        if ext == 'py':
            if template_type == "class":
                class_name = filename.replace('.py', '').title().replace('_', '')
                return f"""class {class_name}:
    \"\"\"
    {class_name} class
    \"\"\"
    
    def __init__(self):
        pass
    
    def example_method(self):
        return "Hello from {class_name}!"
"""
            elif template_type == "function":
                return f"""def main():
    \"\"\"
    Main function for {filename}
    \"\"\"
    print("Hello from {filename}!")

if __name__ == "__main__":
    main()
"""
            else:
                return SmartCodeGenerator.generate_hello_world(filename)
        
        return SmartCodeGenerator.generate_hello_world(filename)
