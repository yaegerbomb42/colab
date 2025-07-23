#!/usr/bin/env python3
"""
Setup script for Multi-Agent Live Development Environment
Helps configure AI providers and environment variables
"""

import os
import sys
from pathlib import Path

def setup_gemini():
    """Setup Google Gemini API"""
    print("🔧 Setting up Google Gemini API")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Copy the API key")
    
    api_key = input("Enter your Gemini API key: ").strip()
    if api_key:
        # Create .env file
        env_file = Path(".env")
        
        # Read existing .env or create new
        env_content = ""
        if env_file.exists():
            env_content = env_file.read_text()
        
        # Update or add GEMINI_API_KEY
        lines = env_content.split('\n')
        updated = False
        
        for i, line in enumerate(lines):
            if line.startswith('GEMINI_API_KEY='):
                lines[i] = f'GEMINI_API_KEY={api_key}'
                updated = True
                break
        
        if not updated:
            lines.append(f'GEMINI_API_KEY={api_key}')
        
        # Write back to .env
        env_file.write_text('\n'.join(lines))
        
        print("✅ Gemini API key saved to .env file")
        print("💡 The environment will use Gemini for smarter agent responses!")
        return True
    else:
        print("❌ No API key provided")
        return False

def setup_openai():
    """Setup OpenAI API"""
    print("🔧 Setting up OpenAI API")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Create a new API key")
    print("3. Copy the API key")
    
    api_key = input("Enter your OpenAI API key: ").strip()
    if api_key:
        env_file = Path(".env")
        env_content = ""
        if env_file.exists():
            env_content = env_file.read_text()
        
        lines = env_content.split('\n')
        updated = False
        
        for i, line in enumerate(lines):
            if line.startswith('OPENAI_API_KEY='):
                lines[i] = f'OPENAI_API_KEY={api_key}'
                updated = True
                break
        
        if not updated:
            lines.append(f'OPENAI_API_KEY={api_key}')
        
        env_file.write_text('\n'.join(lines))
        print("✅ OpenAI API key saved to .env file")
        return True
    else:
        print("❌ No API key provided")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Check if we need to install AI packages
    choice = input("Which AI provider would you like to use? (gemini/openai/mock): ").lower().strip()
    
    if choice == "gemini":
        try:
            import google.generativeai
            print("✅ Google Generative AI already installed")
        except ImportError:
            print("📦 Installing Google Generative AI...")
            os.system("pip install google-generativeai")
    
    elif choice == "openai":
        try:
            import openai
            print("✅ OpenAI already installed")
        except ImportError:
            print("📦 Installing OpenAI...")
            os.system("pip install openai")
    
    return choice

def main():
    """Main setup function"""
    print("🚀 Multi-Agent Live Development Environment Setup")
    print("=" * 60)
    
    # Install dependencies
    provider = install_dependencies()
    
    # Setup API keys
    if provider == "gemini":
        setup_gemini()
    elif provider == "openai":
        setup_openai()
    else:
        print("💡 Using mock responses (no API key needed)")
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete!")
    print("Run 'python main_dev.py' to start the development server")
    print("🔗 Open http://localhost:8000 in your browser")
    print("🤖 Create agents and start collaborative coding!")

if __name__ == "__main__":
    main()
