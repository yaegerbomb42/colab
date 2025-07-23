#!/usr/bin/env python3
"""
Agent startup script for creating and managing multiple Llama 1B agents.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.streaming_engine import TokenStreamingEngine
from src.core.chat_system import CollaborativeChatSystem
from src.core.codebase_awareness import CodebaseAwarenessSystem
from src.agents.llama_agent import LlamaAgent
from src.core.models import AgentRole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentManager:
    """Manages multiple autonomous agents for collaborative development"""
    
    def __init__(self, workspace_path: str = "./workspace"):
        self.workspace_path = workspace_path
        self.streaming_engine = TokenStreamingEngine(workspace_path)
        self.chat_system = CollaborativeChatSystem()
        self.codebase_system = CodebaseAwarenessSystem(workspace_path)
        self.agents: List[LlamaAgent] = []
        self.agent_tasks: List[asyncio.Task] = []
    
    async def create_agent(self, name: str, role: AgentRole, model_name: str = "microsoft/DialoGPT-small") -> LlamaAgent:
        """Create and initialize a new agent"""
        agent_id = f"agent_{len(self.agents)}"
        
        agent = LlamaAgent(
            agent_id=agent_id,
            name=name,
            role=role,
            streaming_engine=self.streaming_engine,
            chat_system=self.chat_system,
            codebase_system=self.codebase_system,
            model_name=model_name
        )
        
        await agent.initialize()
        self.agents.append(agent)
        
        logger.info(f"Created agent: {name} ({role}) with ID: {agent_id}")
        return agent
    
    async def start_all_agents(self):
        """Start all created agents"""
        logger.info(f"Starting {len(self.agents)} agents...")
        
        for agent in self.agents:
            task = asyncio.create_task(agent.start())
            self.agent_tasks.append(task)
        
        logger.info("All agents started successfully")
    
    async def stop_all_agents(self):
        """Stop all agents"""
        logger.info("Stopping all agents...")
        
        for agent in self.agents:
            await agent.stop()
        
        for task in self.agent_tasks:
            task.cancel()
        
        await asyncio.gather(*self.agent_tasks, return_exceptions=True)
        logger.info("All agents stopped")
    
    async def run_collaboration_session(self, duration_minutes: int = 30):
        """Run a collaborative session for a specified duration"""
        logger.info(f"Starting {duration_minutes}-minute collaboration session")
        
        # Index the workspace first
        await self.codebase_system.index_workspace()
        
        # Give agents some initial tasks
        await self._assign_initial_tasks()
        
        # Run for specified duration
        try:
            await asyncio.sleep(duration_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
        
        logger.info("Collaboration session completed")
    
    async def _assign_initial_tasks(self):
        """Assign initial tasks to agents based on their roles"""
        for agent in self.agents:
            if agent.agent.role == AgentRole.GENERAL:
                agent.assign_task(
                    "Create a basic application structure",
                    ["main.py", "utils.py"]
                )
            elif agent.agent.role == AgentRole.UI:
                agent.assign_task(
                    "Design user interface components",
                    ["index.html", "styles.css", "app.js"]
                )
            elif agent.agent.role == AgentRole.SECURITY:
                agent.assign_task(
                    "Review code for security vulnerabilities",
                    []
                )
            elif agent.agent.role == AgentRole.LINTER:
                agent.assign_task(
                    "Maintain code quality and style",
                    []
                )
            elif agent.agent.role == AgentRole.TEST:
                agent.assign_task(
                    "Write comprehensive tests",
                    ["test_main.py", "test_utils.py"]
                )
    
    def get_session_summary(self) -> dict:
        """Get a summary of the current session"""
        total_edits = len(self.streaming_engine.edit_history)
        total_messages = len(self.chat_system.messages)
        
        agent_activity = {}
        for agent in self.agents:
            status = agent.get_status()
            agent_activity[agent.agent.name] = {
                "status": status["status"],
                "current_task": status["current_task"],
                "files_worked_on": len([edit for edit in self.streaming_engine.edit_history 
                                      if edit.agent_id == agent.agent.id])
            }
        
        return {
            "total_agents": len(self.agents),
            "total_edits": total_edits,
            "total_messages": total_messages,
            "agent_activity": agent_activity,
            "files_created": len(self.streaming_engine.file_states)
        }

async def create_demo_agents(manager: AgentManager, num_agents: int = 3):
    """Create a diverse set of demo agents"""
    agent_configs = [
        ("Alice", AgentRole.GENERAL, "Primary developer agent"),
        ("Bob", AgentRole.UI, "UI/UX specialist agent"),
        ("Carol", AgentRole.SECURITY, "Security-focused agent"),
        ("Dave", AgentRole.LINTER, "Code quality agent"),
        ("Eve", AgentRole.TEST, "Testing specialist agent")
    ]
    
    for i in range(min(num_agents, len(agent_configs))):
        name, role, description = agent_configs[i]
        agent = await manager.create_agent(name, role)
        logger.info(f"Created {description}: {name}")

async def run_demo_session(args):
    """Run a demonstration session with multiple agents"""
    print("🤖 Multi-Agent Live Development Environment")
    print("=" * 50)
    print(f"📊 Creating {args.num_agents} agents")
    print(f"⏱️  Session duration: {args.duration} minutes")
    print(f"🧠 Model: {args.model}")
    print(f"📁 Workspace: {args.workspace}")
    print("=" * 50)
    
    manager = AgentManager(args.workspace)
    
    try:
        # Create agents
        await create_demo_agents(manager, args.num_agents)
        
        # Start agents
        await manager.start_all_agents()
        
        # Let them chat and introduce themselves
        await manager.chat_system.send_message(
            agent_id="system",
            content="Welcome to the collaborative development session! Please introduce yourselves and discuss what you'd like to work on."
        )
        
        # Run collaboration session
        await manager.run_collaboration_session(args.duration)
        
    except KeyboardInterrupt:
        print("\n🛑 Session interrupted by user")
    finally:
        # Stop all agents
        await manager.stop_all_agents()
        
        # Print session summary
        summary = manager.get_session_summary()
        print("\n📊 Session Summary")
        print("=" * 30)
        print(f"👥 Agents: {summary['total_agents']}")
        print(f"✏️  Total edits: {summary['total_edits']}")
        print(f"💬 Messages: {summary['total_messages']}")
        print(f"📄 Files created: {summary['files_created']}")
        print("\n🤖 Agent Activity:")
        for name, activity in summary['agent_activity'].items():
            print(f"  {name}: {activity['files_worked_on']} edits, status: {activity['status']}")

def main():
    parser = argparse.ArgumentParser(description="Start multiple Llama 1B agents for collaborative development")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", 
                       help="Model to use for agents (default: microsoft/DialoGPT-small)")
    parser.add_argument("--num-agents", type=int, default=3,
                       help="Number of agents to create (default: 3)")
    parser.add_argument("--duration", type=int, default=10,
                       help="Session duration in minutes (default: 10)")
    parser.add_argument("--workspace", default="./workspace",
                       help="Workspace directory (default: ./workspace)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure workspace exists
    Path(args.workspace).mkdir(exist_ok=True)
    
    try:
        asyncio.run(run_demo_session(args))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
