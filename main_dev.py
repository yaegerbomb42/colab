#!/usr/bin/env python3
"""
Development version of the Multi-Agent Live Development Environment
Simplified for testing and demonstration purposes
"""

import asyncio
import logging
import os
from pathlib import Path
from src.api.main import CollaborativeAPI
from src.core.ai_integration import set_ai_provider, AIProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_environment():
    """Load environment variables from .env file"""
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

def detect_ai_provider():
    """Detect and configure the best available AI provider"""
    if os.getenv('GEMINI_API_KEY'):
        try:
            import google.generativeai
            set_ai_provider(AIProvider.GEMINI)
            return "🧠 Google Gemini"
        except ImportError:
            print("⚠️  Gemini API key found but google-generativeai not installed")
            print("   Run: pip install google-generativeai")
    
    if os.getenv('OPENAI_API_KEY'):
        try:
            import openai
            set_ai_provider(AIProvider.OPENAI)
            return "🧠 OpenAI GPT"
        except ImportError:
            print("⚠️  OpenAI API key found but openai not installed")
            print("   Run: pip install openai")
    
    # Fallback to mock
    set_ai_provider(AIProvider.MOCK)
    return "🤖 Mock AI (demo mode)"

async def main():
    """Main entry point for development server"""
    print("🚀 Multi-Agent Live Development Environment (Dev Mode)")
    print("=" * 60)
    
    # Load environment variables
    load_environment()
    
    # Detect and configure AI provider
    ai_provider = detect_ai_provider()
    
    print(f"🔗 Starting development server...")
    print(f"📁 Workspace: ./workspace")
    print(f"� AI Provider: {ai_provider}")
    print(f"🤖 Smart agent responses enabled")
    print(f"💬 Chat system: Real-time collaboration")
    print(f"🔄 Token streaming: Live editing")
    
    if "Mock" in ai_provider:
        print("💡 For smarter agents, run 'python setup.py' to configure AI")
    
    print("=" * 60)
    
    # Create workspace if it doesn't exist
    os.makedirs("./workspace", exist_ok=True)
    
    # Initialize API
    api = CollaborativeAPI()
    
    # Start server
    await api.start_server(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down Multi-Agent Live Development Environment")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        logging.exception("Server startup error")
