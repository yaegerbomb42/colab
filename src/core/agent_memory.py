"""
Enhanced Agent Memory and Context System
Provides intelligent context management, tool access, and coordination capabilities
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    WORKING = "working"  # Current task context
    CONVERSATION = "conversation"  # Recent chat history
    CODEBASE = "codebase"  # Code understanding
    TOOL_USAGE = "tool_usage"  # Tool execution history
    COORDINATION = "coordination"  # Inter-agent coordination

@dataclass
class MemoryEntry:
    id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamp: datetime
    relevance_score: float = 1.0
    tags: List[str] = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class TaskContext:
    task_id: str
    description: str
    status: str
    assigned_files: List[str]
    dependencies: List[str]
    progress: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

@dataclass
class ToolCapability:
    name: str
    description: str
    parameters: Dict[str, Any]
    requires_file: bool = False
    async_execution: bool = False

class AgentMemorySystem:
    """Advanced memory system for agent context and coordination"""
    
    def __init__(self, agent_id: str, max_entries: int = 500):
        self.agent_id = agent_id
        self.max_entries = max_entries
        
        # Memory storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_index: Dict[MemoryType, List[str]] = {
            memory_type: [] for memory_type in MemoryType
        }
        
        # Current context
        self.current_task: Optional[TaskContext] = None
        self.active_files: Set[str] = set()
        self.recent_actions: List[Dict[str, Any]] = []
        
        # Tool capabilities
        self.available_tools: Dict[str, ToolCapability] = {}
        self._register_default_tools()
        
        # Coordination state
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
    def _register_default_tools(self):
        """Register default tool capabilities"""
        tools = [
            ToolCapability(
                name="create_file",
                description="Create a new file with specified content",
                parameters={"file_path": "str", "content": "str"},
                requires_file=False,
                async_execution=True
            ),
            ToolCapability(
                name="edit_file",
                description="Edit existing file content",
                parameters={"file_path": "str", "changes": "dict"},
                requires_file=True,
                async_execution=True
            ),
            ToolCapability(
                name="read_file",
                description="Read and analyze file content",
                parameters={"file_path": "str"},
                requires_file=True,
                async_execution=False
            ),
            ToolCapability(
                name="search_code",
                description="Search through codebase for patterns",
                parameters={"query": "str", "file_types": "list"},
                requires_file=False,
                async_execution=False
            ),
            ToolCapability(
                name="analyze_dependencies",
                description="Analyze file dependencies and imports",
                parameters={"file_path": "str"},
                requires_file=True,
                async_execution=False
            ),
            ToolCapability(
                name="collaborate",
                description="Coordinate with other agents",
                parameters={"message": "str", "target_agents": "list"},
                requires_file=False,
                async_execution=True
            )
        ]
        
        for tool in tools:
            self.available_tools[tool.name] = tool
    
    async def add_memory(self, memory_type: MemoryType, content: Dict[str, Any], 
                        tags: List[str] = None, expires_in_hours: Optional[int] = None) -> str:
        """Add a new memory entry"""
        memory_id = f"{memory_type.value}_{len(self.memories)}_{datetime.now().timestamp()}"
        
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.now() + timedelta(hours=expires_in_hours)
        
        memory = MemoryEntry(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            timestamp=datetime.now(),
            tags=tags or [],
            expires_at=expires_at
        )
        
        self.memories[memory_id] = memory
        self.memory_index[memory_type].append(memory_id)
        
        # Cleanup if needed
        await self._cleanup_old_memories()
        
        return memory_id
    
    async def get_memories(self, memory_type: Optional[MemoryType] = None, 
                          tags: List[str] = None, limit: int = 50) -> List[MemoryEntry]:
        """Retrieve memories with filtering"""
        memories = []
        
        for memory_id, memory in self.memories.items():
            # Check expiration
            if memory.expires_at and datetime.now() > memory.expires_at:
                continue
            
            # Type filter
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Tag filter
            if tags and not any(tag in memory.tags for tag in tags):
                continue
            
            memories.append(memory)
        
        # Sort by relevance and timestamp
        memories.sort(key=lambda m: (m.relevance_score, m.timestamp), reverse=True)
        
        return memories[:limit]
    
    async def update_task_context(self, task: TaskContext):
        """Update current task context"""
        self.current_task = task
        
        # Add to working memory
        await self.add_memory(
            MemoryType.WORKING,
            {
                "task_id": task.task_id,
                "description": task.description,
                "status": task.status,
                "files": task.assigned_files,
                "progress": task.progress
            },
            tags=["current_task", "active"],
            expires_in_hours=24
        )
    
    async def add_conversation_context(self, messages: List[Dict[str, Any]]):
        """Add conversation context to memory"""
        await self.add_memory(
            MemoryType.CONVERSATION,
            {
                "messages": messages,
                "participants": list(set(msg.get("agent_id", "unknown") for msg in messages)),
                "topics": self._extract_topics(messages)
            },
            tags=["recent_chat"],
            expires_in_hours=6
        )
    
    async def add_codebase_knowledge(self, file_path: str, analysis: Dict[str, Any]):
        """Add codebase analysis to memory"""
        await self.add_memory(
            MemoryType.CODEBASE,
            {
                "file_path": file_path,
                "analysis": analysis,
                "language": self._detect_language(file_path),
                "dependencies": analysis.get("dependencies", [])
            },
            tags=["code_analysis", self._detect_language(file_path)],
            expires_in_hours=48
        )
    
    async def record_tool_usage(self, tool_name: str, parameters: Dict[str, Any], 
                               result: Dict[str, Any], success: bool):
        """Record tool usage for learning"""
        await self.add_memory(
            MemoryType.TOOL_USAGE,
            {
                "tool": tool_name,
                "parameters": parameters,
                "result": result,
                "success": success,
                "execution_time": result.get("execution_time", 0)
            },
            tags=["tool_use", tool_name, "success" if success else "failure"],
            expires_in_hours=12
        )
    
    async def add_coordination_event(self, event_type: str, details: Dict[str, Any]):
        """Record coordination events with other agents"""
        await self.add_memory(
            MemoryType.COORDINATION,
            {
                "event_type": event_type,
                "details": details,
                "other_agents": details.get("agents", [])
            },
            tags=["coordination", event_type],
            expires_in_hours=6
        )
    
    async def get_relevant_context(self, query: str, context_type: Optional[MemoryType] = None) -> Dict[str, Any]:
        """Get relevant context for a query or task"""
        context = {
            "current_task": asdict(self.current_task) if self.current_task else None,
            "active_files": list(self.active_files),
            "recent_actions": self.recent_actions[-10:],  # Last 10 actions
            "relevant_memories": [],
            "available_tools": list(self.available_tools.keys()),
            "coordination_status": await self._get_coordination_status()
        }
        
        # Get relevant memories
        query_tags = self._extract_tags_from_query(query)
        memories = await self.get_memories(memory_type=context_type, tags=query_tags, limit=10)
        context["relevant_memories"] = [asdict(memory) for memory in memories]
        
        return context
    
    async def suggest_tools_for_task(self, task_description: str) -> List[str]:
        """Suggest appropriate tools for a task"""
        task_lower = task_description.lower()
        suggested_tools = []
        
        tool_keywords = {
            "create_file": ["create", "new", "write", "generate"],
            "edit_file": ["edit", "modify", "change", "update", "fix"],
            "read_file": ["read", "analyze", "check", "review"],
            "search_code": ["search", "find", "locate", "grep"],
            "analyze_dependencies": ["dependencies", "imports", "modules"],
            "collaborate": ["collaborate", "coordinate", "work with"]
        }
        
        for tool, keywords in tool_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                suggested_tools.append(tool)
        
        return suggested_tools
    
    async def can_execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """Check if tool can be executed with given parameters"""
        if tool_name not in self.available_tools:
            return False
        
        tool = self.available_tools[tool_name]
        
        # Check required parameters
        required_params = tool.parameters
        for param_name, param_type in required_params.items():
            if param_name not in parameters:
                return False
        
        # Check file requirements
        if tool.requires_file:
            file_path = parameters.get("file_path")
            if not file_path:
                return False
        
        return True
    
    def _extract_topics(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract topics from messages"""
        topics = set()
        
        for msg in messages:
            content = msg.get("content", "").lower()
            
            # Simple topic extraction
            if "create" in content:
                topics.add("file_creation")
            if "edit" in content or "modify" in content:
                topics.add("file_editing")
            if "debug" in content or "fix" in content:
                topics.add("debugging")
            if "html" in content or "webpage" in content:
                topics.add("web_development")
            if "python" in content or ".py" in content:
                topics.add("python_development")
        
        return list(topics)
    
    def _extract_tags_from_query(self, query: str) -> List[str]:
        """Extract relevant tags from a query"""
        query_lower = query.lower()
        tags = []
        
        # File type tags
        if "python" in query_lower or ".py" in query_lower:
            tags.append("python")
        if "html" in query_lower or ".html" in query_lower:
            tags.append("html")
        if "javascript" in query_lower or ".js" in query_lower:
            tags.append("javascript")
        
        # Action tags
        if "create" in query_lower:
            tags.append("creation")
        if "edit" in query_lower:
            tags.append("editing")
        if "debug" in query_lower:
            tags.append("debugging")
        
        return tags
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        extension = file_path.split('.')[-1].lower()
        
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'html': 'html',
            'css': 'css',
            'ts': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c'
        }
        
        return language_map.get(extension, 'text')
    
    async def _get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status with other agents"""
        recent_coordination = await self.get_memories(
            MemoryType.COORDINATION, 
            limit=5
        )
        
        return {
            "recent_events": len(recent_coordination),
            "known_agents": list(self.known_agents.keys()),
            "last_collaboration": recent_coordination[0].timestamp if recent_coordination else None
        }
    
    async def _cleanup_old_memories(self):
        """Clean up expired and excess memories"""
        now = datetime.now()
        expired_ids = []
        
        # Find expired memories
        for memory_id, memory in self.memories.items():
            if memory.expires_at and now > memory.expires_at:
                expired_ids.append(memory_id)
        
        # Remove expired memories
        for memory_id in expired_ids:
            memory = self.memories[memory_id]
            del self.memories[memory_id]
            if memory_id in self.memory_index[memory.memory_type]:
                self.memory_index[memory.memory_type].remove(memory_id)
        
        # Limit total memories
        if len(self.memories) > self.max_entries:
            # Remove oldest memories from least important types first
            type_priority = [
                MemoryType.TOOL_USAGE,
                MemoryType.CONVERSATION,
                MemoryType.COORDINATION,
                MemoryType.CODEBASE,
                MemoryType.WORKING
            ]
            
            for memory_type in type_priority:
                if len(self.memories) <= self.max_entries:
                    break
                
                type_memories = self.memory_index[memory_type]
                if type_memories:
                    # Remove oldest from this type
                    oldest_id = min(type_memories, key=lambda mid: self.memories[mid].timestamp)
                    del self.memories[oldest_id]
                    type_memories.remove(oldest_id)

    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of current memory state"""
        return {
            "total_memories": len(self.memories),
            "memories_by_type": {
                memory_type.value: len(self.memory_index[memory_type])
                for memory_type in MemoryType
            },
            "current_task": self.current_task.task_id if self.current_task else None,
            "active_files": list(self.active_files),
            "available_tools": len(self.available_tools),
            "known_agents": len(self.known_agents)
        }
