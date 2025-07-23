"""
Advanced Multi-Agent Task Coordinator
Handles automatic agent creation, task distribution, and parallel execution
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.core.models import AgentRole, MessageType
from src.agents.fast_agent import FastAgent
from src.core.streaming_engine import TokenStreamingEngine
from src.core.chat_system import CollaborativeChatSystem
from src.core.codebase_awareness import CodebaseAwarenessSystem

logger = logging.getLogger(__name__)

@dataclass
class Task:
    """Represents a development task"""
    id: str
    description: str
    task_type: str  # 'create_file', 'edit_file', 'debug', 'research', etc.
    priority: int
    file_path: Optional[str] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    created_at: datetime = None
    estimated_duration: int = 60  # seconds
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []

class TaskCoordinator:
    """Coordinates tasks between multiple AI agents automatically"""
    
    def __init__(self, streaming_engine: TokenStreamingEngine,
                 chat_system: CollaborativeChatSystem,
                 codebase_system: CodebaseAwarenessSystem):
        
        self.streaming_engine = streaming_engine
        self.chat_system = chat_system
        self.codebase_system = codebase_system
        
        self.agents: Dict[str, FastAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.Queue()
        
        self.next_agent_id = 0
        self.next_task_id = 0
        
        # Auto-scaling parameters
        self.max_agents = 5
        self.min_agents = 1
        self.agent_creation_threshold = 3  # Create new agent if queue > 3 tasks
        
        # Task routing rules
        self.task_patterns = {
            'python': {
                'patterns': [r'\.py$', r'python', r'script'],
                'preferred_role': AgentRole.GENERAL,
                'priority': 1
            },
            'html': {
                'patterns': [r'\.html$', r'webpage', r'html', r'website'],
                'preferred_role': AgentRole.UI,
                'priority': 1
            },
            'css': {
                'patterns': [r'\.css$', r'style', r'css'],
                'preferred_role': AgentRole.UI,
                'priority': 2
            },
            'javascript': {
                'patterns': [r'\.js$', r'javascript', r'react', r'node'],
                'preferred_role': AgentRole.UI,
                'priority': 1
            },
            'debug': {
                'patterns': [r'fix', r'debug', r'error', r'bug'],
                'preferred_role': AgentRole.LINTER,
                'priority': 3
            },
            'test': {
                'patterns': [r'test', r'spec', r'unit'],
                'preferred_role': AgentRole.GENERAL,
                'priority': 2
            }
        }
    
    async def process_user_request(self, user_input: str) -> Dict[str, Any]:
        """Process a user request and create appropriate tasks"""
        
        # Parse the request into tasks
        tasks = await self._parse_request_to_tasks(user_input)
        
        if not tasks:
            return {
                "status": "error",
                "message": "Could not understand the request",
                "tasks_created": 0
            }
        
        # Add tasks to the system
        created_tasks = []
        for task in tasks:
            task_id = await self._add_task(task)
            created_tasks.append(task_id)
        
        # Ensure we have enough agents
        await self._auto_scale_agents()
        
        # Start task processing
        asyncio.create_task(self._process_task_queue())
        
        return {
            "status": "success",
            "message": f"Created {len(created_tasks)} tasks",
            "tasks_created": len(created_tasks),
            "task_ids": created_tasks,
            "estimated_completion": len(created_tasks) * 30  # seconds
        }
    
    async def _parse_request_to_tasks(self, user_input: str) -> List[Task]:
        """Parse user input into actionable tasks"""
        tasks = []
        user_lower = user_input.lower()
        
        # Pattern for file creation requests
        create_patterns = [
            (r'create\s+(\w+)\.py\s+(.+)', 'create_python_file'),
            (r'create\s+(.+?)\.html\s+(.+)', 'create_html_file'),
            (r'create\s+(.+?)\.js\s+(.+)', 'create_js_file'),
            (r'create\s+(.+?)\s+file', 'create_file'),
            (r'make\s+(.+?)\.py', 'create_python_file'),
            (r'build\s+(.+?)\s+(.+)', 'create_file'),
        ]
        
        # Check for file creation patterns
        for pattern, task_type in create_patterns:
            match = re.search(pattern, user_lower)
            if match:
                if task_type == 'create_python_file':
                    filename = f"{match.group(1)}.py"
                    description = match.group(2) if len(match.groups()) > 1 else "Python file"
                elif task_type == 'create_html_file':
                    filename = f"{match.group(1)}.html"
                    description = match.group(2) if len(match.groups()) > 1 else "HTML file"
                elif task_type == 'create_js_file':
                    filename = f"{match.group(1)}.js"
                    description = match.group(2) if len(match.groups()) > 1 else "JavaScript file"
                else:
                    filename = match.group(1)
                    description = user_input
                
                task = Task(
                    id=f"task_{self.next_task_id}",
                    description=f"Create {filename}: {description}",
                    task_type="create_file",
                    priority=1,
                    file_path=filename
                )
                tasks.append(task)
                self.next_task_id += 1
        
        # Check for complex multi-file requests
        complex_patterns = [
            (r'create\s+(.+?)\s+webpage\s+(.+)', 'create_webpage'),
            (r'build\s+(.+?)\s+app', 'create_app'),
            (r'make\s+(.+?)\s+project', 'create_project')
        ]
        
        for pattern, task_type in complex_patterns:
            match = re.search(pattern, user_lower)
            if match:
                if task_type == 'create_webpage':
                    theme = match.group(1)
                    details = match.group(2)
                    
                    # Create HTML task
                    html_task = Task(
                        id=f"task_{self.next_task_id}",
                        description=f"Create {theme} webpage HTML: {details}",
                        task_type="create_file",
                        priority=1,
                        file_path=f"{theme.replace(' ', '_')}.html"
                    )
                    tasks.append(html_task)
                    self.next_task_id += 1
                    
                    # Create CSS task
                    css_task = Task(
                        id=f"task_{self.next_task_id}",
                        description=f"Create {theme} webpage CSS: {details}",
                        task_type="create_file",
                        priority=2,
                        file_path=f"{theme.replace(' ', '_')}.css",
                        dependencies=[html_task.id]
                    )
                    tasks.append(css_task)
                    self.next_task_id += 1
        
        # If no specific patterns matched, create a general task
        if not tasks:
            task = Task(
                id=f"task_{self.next_task_id}",
                description=user_input,
                task_type="general",
                priority=1
            )
            tasks.append(task)
            self.next_task_id += 1
        
        return tasks
    
    async def _add_task(self, task: Task) -> str:
        """Add a task to the system"""
        self.tasks[task.id] = task
        await self.task_queue.put(task)
        
        # Notify via chat
        await self.chat_system.send_message(
            agent_id="system",
            content=f"📋 New task created: {task.description}",
            message_type=MessageType.TASK_UPDATE
        )
        
        return task.id
    
    async def _auto_scale_agents(self):
        """Automatically create agents based on workload"""
        active_agents = len([a for a in self.agents.values() if a.is_active])
        queue_size = self.task_queue.qsize()
        
        # Create new agent if needed
        if (queue_size > self.agent_creation_threshold and 
            active_agents < self.max_agents):
            
            await self._create_agent()
        
        # Ensure minimum agents
        if active_agents < self.min_agents:
            await self._create_agent()
    
    async def _create_agent(self) -> str:
        """Create a new AI agent"""
        agent_id = f"agent_{self.next_agent_id}"
        self.next_agent_id += 1
        
        # Determine role based on current tasks
        role = await self._determine_best_role()
        
        # Create agent
        agent = FastAgent(
            agent_id=agent_id,
            name=f"Agent {self.next_agent_id}",
            role=role,
            streaming_engine=self.streaming_engine,
            chat_system=self.chat_system,
            codebase_system=self.codebase_system
        )
        
        # Initialize and start
        await agent.initialize()
        await agent.start()
        
        self.agents[agent_id] = agent
        
        # Notify about new agent
        await self.chat_system.send_message(
            agent_id="system",
            content=f"🤖 New {role.value} agent created: {agent.agent.name}",
            message_type=MessageType.AGENT_JOIN
        )
        
        logger.info(f"Created new agent: {agent_id} with role {role.value}")
        return agent_id
    
    async def _determine_best_role(self) -> AgentRole:
        """Determine the best role for a new agent based on pending tasks"""
        role_needs = {
            AgentRole.GENERAL: 0,
            AgentRole.UI: 0,
            AgentRole.LINTER: 0
        }
        
        # Analyze pending tasks
        for task in self.tasks.values():
            if task.status == "pending":
                for task_type, config in self.task_patterns.items():
                    for pattern in config['patterns']:
                        if re.search(pattern, task.description.lower()):
                            role_needs[config['preferred_role']] += config['priority']
                            break
        
        # Return role with highest need
        return max(role_needs, key=role_needs.get) or AgentRole.GENERAL
    
    async def _process_task_queue(self):
        """Process the task queue by assigning tasks to agents"""
        while True:
            try:
                # Get next task
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Find best agent for this task
                agent = await self._find_best_agent(task)
                
                if agent:
                    # Assign task
                    task.assigned_agent = agent.agent.id
                    task.status = "assigned"
                    
                    # Send task to agent
                    await self._send_task_to_agent(agent, task)
                    
                else:
                    # No available agent, put back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
                await asyncio.sleep(1)
    
    async def _find_best_agent(self, task: Task) -> Optional[FastAgent]:
        """Find the best agent for a specific task"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.is_active and not agent.current_task
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on role match
        scored_agents = []
        for agent in available_agents:
            score = 1  # Base score
            
            # Role-based scoring
            for task_type, config in self.task_patterns.items():
                for pattern in config['patterns']:
                    if re.search(pattern, task.description.lower()):
                        if agent.agent.role == config['preferred_role']:
                            score += config['priority'] * 2
                        break
            
            scored_agents.append((score, agent))
        
        # Return highest scoring agent
        scored_agents.sort(key=lambda x: x[0], reverse=True)
        return scored_agents[0][1]
    
    async def _send_task_to_agent(self, agent: FastAgent, task: Task):
        """Send a task to a specific agent"""
        try:
            # Update task status
            task.status = "in_progress"
            agent.current_task = task
            
            # Create task data for agent
            task_data = {
                "type": "task_assignment",
                "content": task.description,
                "task_id": task.id,
                "file_path": task.file_path
            }
            
            # Send directly to agent's task handler
            await agent.handle_task(task_data)
            
            # Mark task as completed
            task.status = "completed"
            agent.current_task = None
            
            # Notify via chat
            await self.chat_system.send_message(
                agent_id="system",
                content=f"📋 Task completed by {agent.agent.name}: {task.description}",
                message_type=MessageType.TASK_UPDATE
            )
            
            logger.info(f"Completed task {task.id} with agent {agent.agent.id}")
            
        except Exception as e:
            logger.error(f"Error sending task to agent: {e}")
            task.status = "failed"
            agent.current_task = None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        active_agents = len([a for a in self.agents.values() if a.is_active])
        pending_tasks = len([t for t in self.tasks.values() if t.status == "pending"])
        in_progress_tasks = len([t for t in self.tasks.values() if t.status == "in_progress"])
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        
        return {
            "agents": {
                "total": len(self.agents),
                "active": active_agents,
                "agents": [
                    {
                        "id": agent.agent.id,
                        "name": agent.agent.name,
                        "role": agent.agent.role.value,
                        "status": "active" if agent.is_active else "idle",
                        "current_task": agent.current_task.description if agent.current_task else None
                    }
                    for agent in self.agents.values()
                ]
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": pending_tasks,
                "in_progress": in_progress_tasks,
                "completed": completed_tasks,
                "queue_size": self.task_queue.qsize()
            }
        }
    
    async def stop_all_agents(self):
        """Stop all agents"""
        for agent in self.agents.values():
            await agent.stop()
        
        logger.info("All agents stopped")

# Global coordinator instance
_coordinator = None

async def get_task_coordinator(streaming_engine: TokenStreamingEngine,
                             chat_system: CollaborativeChatSystem,
                             codebase_system: CodebaseAwarenessSystem) -> TaskCoordinator:
    """Get or create the global task coordinator"""
    global _coordinator
    
    if _coordinator is None:
        _coordinator = TaskCoordinator(streaming_engine, chat_system, codebase_system)
    
    return _coordinator
