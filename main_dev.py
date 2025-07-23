#!/usr/bin/env python3
"""
Development version of the Multi-Agent Live Development Environment
Simplified for testing and demonstration purposes
"""

import asyncio
import logging
import os
from src.api.main import CollaborativeAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main entry point for development server"""
    print("🚀 Multi-Agent Live Development Environment (Dev Mode)")
    print("=" * 60)
    print("🔗 Starting development server...")
    print("📁 Workspace: ./workspace")
    print("🤖 Reduced agent activity for better testing")
    print("💬 Chat system: Real-time collaboration")
    print("🔄 Token streaming: Live editing")
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
