import asyncio
import json
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
import logging
from src.core.models import ChatMessage, Agent, MessageType
import re

logger = logging.getLogger(__name__)

class CollaborativeChatSystem:
    """
    Real-time chat system for agent collaboration with file referencing,
    conversation threading, and shared memory integration.
    """
    
    def __init__(self, max_history: int = 1000):
        self.messages: List[ChatMessage] = []
        self.agents: Dict[str, Agent] = {}
        self.message_listeners: List[Callable] = []
        self.max_history = max_history
        self.active_conversations: Dict[str, List[str]] = {}  # thread_id -> message_ids
        
    def add_message_listener(self, callback: Callable):
        """Add a callback to be notified of new messages"""
        self.message_listeners.append(callback)
    
    async def _notify_listeners(self, message: ChatMessage):
        """Notify all listeners of a new message"""
        for listener in self.message_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(message)
                else:
                    listener(message)
            except Exception as e:
                logger.error(f"Error in message listener: {e}")
    
    async def send_message(self, agent_id: str, content: str, 
                          message_type: MessageType = MessageType.CHAT,
                          file_reference: Optional[str] = None,
                          line_reference: Optional[int] = None,
                          parent_message_id: Optional[str] = None) -> ChatMessage:
        """Send a new chat message"""
        
        # Parse file references from content
        if not file_reference:
            file_ref, line_ref = self._extract_file_references(content)
            if file_ref:
                file_reference = file_ref
                line_reference = line_ref
        
        message = ChatMessage(
            agent_id=agent_id,
            message_type=message_type,
            content=content,
            file_reference=file_reference,
            line_reference=line_reference,
            parent_message_id=parent_message_id
        )
        
        # Add to history
        self.messages.append(message)
        
        # Trim history if needed
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        # Update agent activity
        if agent_id in self.agents:
            self.agents[agent_id].last_activity = datetime.now()
        
        # Notify listeners
        await self._notify_listeners(message)
        
        return message
    
    def _extract_file_references(self, content: str) -> tuple[Optional[str], Optional[int]]:
        """Extract file and line references from message content"""
        # Look for patterns like "file.py:123" or "src/file.js line 45"
        patterns = [
            r'([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+):(\d+)',  # file.py:123
            r'([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)\s+line\s+(\d+)',  # file.py line 123
            r'([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)\s+at\s+(\d+)',  # file.py at 123
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1), int(match.group(2))
        
        # Look for just file references
        file_pattern = r'([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)'
        match = re.search(file_pattern, content)
        if match:
            return match.group(1), None
        
        return None, None
    
    async def get_conversation_thread(self, message_id: str) -> List[ChatMessage]:
        """Get all messages in a conversation thread"""
        thread_messages = []
        
        # Find the root message
        root_message = None
        for msg in self.messages:
            if msg.id == message_id:
                root_message = msg
                break
        
        if not root_message:
            return []
        
        # Find the actual root (if this message is a reply)
        while root_message.parent_message_id:
            for msg in self.messages:
                if msg.id == root_message.parent_message_id:
                    root_message = msg
                    break
            else:
                break
        
        # Collect all messages in the thread
        def collect_replies(msg: ChatMessage):
            thread_messages.append(msg)
            for reply in self.messages:
                if reply.parent_message_id == msg.id:
                    collect_replies(reply)
        
        collect_replies(root_message)
        return sorted(thread_messages, key=lambda m: m.timestamp)
    
    async def search_messages(self, query: str, agent_id: Optional[str] = None,
                             file_path: Optional[str] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[ChatMessage]:
        """Search chat messages with various filters"""
        results = []
        
        for message in self.messages:
            # Text search
            if query.lower() not in message.content.lower():
                continue
            
            # Agent filter
            if agent_id and message.agent_id != agent_id:
                continue
            
            # File filter
            if file_path and message.file_reference != file_path:
                continue
            
            # Time range filter
            if start_time and message.timestamp < start_time:
                continue
            if end_time and message.timestamp > end_time:
                continue
            
            results.append(message)
        
        return sorted(results, key=lambda m: m.timestamp, reverse=True)
    
    async def get_recent_context(self, agent_id: str, minutes: int = 30) -> List[ChatMessage]:
        """Get recent conversation context for an agent"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_messages = [
            msg for msg in self.messages
            if msg.timestamp >= cutoff_time
        ]
        
        return sorted(recent_messages, key=lambda m: m.timestamp)
    
    async def get_file_discussions(self, file_path: str) -> List[ChatMessage]:
        """Get all messages that reference a specific file"""
        return [
            msg for msg in self.messages
            if msg.file_reference == file_path or file_path in msg.content
        ]
    
    async def create_task_from_discussion(self, message_ids: List[str], 
                                        agent_id: str) -> Optional[str]:
        """Create a task summary from a discussion thread"""
        messages = []
        for msg_id in message_ids:
            for msg in self.messages:
                if msg.id == msg_id:
                    messages.append(msg)
                    break
        
        if not messages:
            return None
        
        # Simple task extraction (in real implementation, could use LLM)
        task_content = "Task derived from discussion:\n"
        for msg in sorted(messages, key=lambda m: m.timestamp):
            task_content += f"- {msg.content[:100]}...\n"
        
        # Send as system message
        await self.send_message(
            agent_id="system",
            content=task_content,
            message_type=MessageType.TASK_UPDATE
        )
        
        return task_content
    
    async def mention_agent(self, mentioning_agent_id: str, 
                           mentioned_agent_id: str, 
                           content: str,
                           file_reference: Optional[str] = None,
                           line_reference: Optional[int] = None) -> ChatMessage:
        """Send a direct mention/notification to another agent"""
        mention_content = f"@{mentioned_agent_id}: {content}"
        
        return await self.send_message(
            agent_id=mentioning_agent_id,
            content=mention_content,
            message_type=MessageType.CHAT,
            file_reference=file_reference,
            line_reference=line_reference
        )
    
    async def suggest_code(self, agent_id: str, file_path: str, 
                          line_number: int, suggestion: str,
                          explanation: str = "") -> ChatMessage:
        """Send a code suggestion message"""
        content = f"Code suggestion for {file_path}:{line_number}\n"
        content += f"```\n{suggestion}\n```"
        if explanation:
            content += f"\n{explanation}"
        
        return await self.send_message(
            agent_id=agent_id,
            content=content,
            message_type=MessageType.CODE_SUGGESTION,
            file_reference=file_path,
            line_reference=line_number
        )
    
    def get_agent_participation(self, hours: int = 24) -> Dict[str, int]:
        """Get message count per agent in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        participation = {}
        
        for message in self.messages:
            if message.timestamp >= cutoff_time:
                agent_id = message.agent_id
                participation[agent_id] = participation.get(agent_id, 0) + 1
        
        return participation
    
    async def get_conversation_summary(self, hours: int = 1) -> str:
        """Get a summary of recent conversations"""
        recent_messages = await self.get_recent_context("", minutes=hours * 60)
        
        if not recent_messages:
            return "No recent conversations."
        
        # Group by agent and topic
        summary = f"Conversation summary (last {hours} hour(s)):\n"
        summary += f"Total messages: {len(recent_messages)}\n"
        
        # Agent participation
        agents = set(msg.agent_id for msg in recent_messages)
        summary += f"Active agents: {', '.join(agents)}\n"
        
        # File discussions
        files = set(msg.file_reference for msg in recent_messages if msg.file_reference)
        if files:
            summary += f"Files discussed: {', '.join(files)}\n"
        
        # Recent key messages
        key_messages = recent_messages[-5:]  # Last 5 messages
        summary += "\nRecent messages:\n"
        for msg in key_messages:
            summary += f"- {msg.agent_id}: {msg.content[:100]}...\n"
        
        return summary
