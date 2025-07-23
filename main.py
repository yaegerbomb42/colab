#!/usr/bin/env python3
"""
Multi-Agent Live Development Environment
Enhanced version with better AI integration and stability
"""

import asyncio
import logging
import os
import sys
from src.api.main import CollaborativeAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the Multi-Agent Live Development Environment"""
    print("🚀 Multi-Agent Live Development Environment")
    print("=" * 60)
    print("🔗 Server starting at: http://localhost:8000")
    print("📁 Workspace: ./workspace")
    print("🤖 Agents: Smarter AI-powered responses")
    print("💬 Chat system: Real-time collaboration")
    print("🔄 Token streaming: Live editing")
    print("🧠 AI: Enhanced with local models and API support")
    print("=" * 60)
    
    # Create workspace directory if it doesn't exist
    workspace_path = "./workspace"
    os.makedirs(workspace_path, exist_ok=True)
    
    # Initialize the collaborative API
    try:
        api = CollaborativeAPI()
        
        # Start the server
        await api.start_server(host="0.0.0.0", port=8000)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down Multi-Agent Live Development Environment")
        print("Thanks for using the collaborative coding platform!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Unexpected error during startup")
