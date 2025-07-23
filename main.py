#!/usr/bin/env python3
"""
Multi-Agent Live Development Environment
Main entry point for starting the collaborative coding platform.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.main import CollaborativeAPI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main entry point"""
    try:
        # Create workspace directory if it doesn't exist
        workspace_path = Path("./workspace")
        workspace_path.mkdir(exist_ok=True)
        
        # Create static directory if it doesn't exist
        static_path = Path("./static")
        static_path.mkdir(exist_ok=True)
        
        # Initialize and start the API
        api = CollaborativeAPI()
        
        print("🚀 Multi-Agent Live Development Environment")
        print("=" * 50)
        print("🔗 Server starting at: http://localhost:8000")
        print("📁 Workspace: ./workspace")
        print("🤖 Agents will be created automatically")
        print("💬 Chat system: Real-time collaboration")
        print("🔄 Token streaming: Live editing")
        print("=" * 50)
        
        await api.start_server()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        logging.error(f"Error starting server: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
