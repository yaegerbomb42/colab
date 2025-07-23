"""
Smart Agent with Enhanced AI capabilities
Much better at understanding user requests and generating appropriate code
"""

import asyncio
import json
import logging
import random
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.core.models import Agent, AgentRole, EditType, MessageType
from src.core.streaming_engine import TokenStreamingEngine
from src.core.chat_system import CollaborativeChatSystem
from src.core.codebase_awareness import CodebaseAwarenessSystem
from src.agents.enhanced_llm import EnhancedLLMProvider, SmartCodeGenerator

logger = logging.getLogger(__name__)

class SmartAgent:
    """
    Smart coding agent with enhanced AI capabilities
    Much better at understanding user requests and collaboration
    """
    
    def __init__(self, agent_id: str, name: str, role: AgentRole,
                 streaming_engine: TokenStreamingEngine,
                 chat_system: CollaborativeChatSystem,
                 codebase_system: CodebaseAwarenessSystem):
        
        self.agent = Agent(id=agent_id, name=name, role=role)
        self.streaming_engine = streaming_engine
        self.chat_system = chat_system
        self.codebase_system = codebase_system
        
        # Enhanced AI provider
        self.llm_provider = EnhancedLLMProvider()
        self.code_generator = SmartCodeGenerator()
        
        # Agent state
        self.is_active = False
        self.current_task = None
        self.edit_queue = asyncio.Queue()
        self.chat_queue = asyncio.Queue()
        
        # Behavioral parameters - much more conservative
        self.typing_speed = 0.02  # Faster typing
        self.autonomous_mode = False  # Never start autonomous
        self.last_action_time = None
        self.minimum_action_delay = 45  # 45 seconds between actions
        
        # Working memory
        self.conversation_context = []
        self.recent_user_requests = []
        
        # Setup callbacks
        streaming_engine.add_edit_listener(self._on_file_edit)
        chat_system.add_message_listener(self._on_chat_message)
        
        logger.info(f"Smart agent {name} initialized with {self.llm_provider.provider} AI provider")
    
    async def initialize(self):
        """Initialize the smart agent"""
        self.agent.status = "ready"
        logger.info(f"Smart agent {self.agent.name} ready with {self.llm_provider.provider} AI")
    
    async def start(self):
        """Start the agent's behavior loop"""
        if self.is_active:
            return
        
        self.is_active = True
        self.agent.status = "active"
        
        logger.info(f"Starting smart agent {self.agent.name}")
        
        # Start parallel tasks - fewer background tasks
        tasks = [
            asyncio.create_task(self._main_behavior_loop()),
            asyncio.create_task(self._process_chat_queue()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in smart agent {self.agent.name}: {e}")
        finally:
            self.is_active = False
            self.agent.status = "inactive"
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        await asyncio.sleep(0.1)
    
    async def _main_behavior_loop(self):
        """Main behavior loop - only responds to direct requests"""
        while self.is_active:
            try:
                await asyncio.sleep(2.0)  # Check every 2 seconds
                
                if not self.is_active:
                    break
                
                # Only check for user requests - no autonomous behavior
                action = await self._check_for_user_requests()
                
                if action:
                    await self._execute_action(action)
                
                # Update agent state
                self.agent.last_activity = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in main loop for {self.agent.name}: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_for_user_requests(self) -> Optional[Dict[str, Any]]:
        """Check for new user requests that need immediate attention"""
        
        # Prevent too frequent actions
        if self.last_action_time:
            time_since_last = datetime.now() - self.last_action_time
            if time_since_last.total_seconds() < self.minimum_action_delay:
                return None
        
        # Get recent messages
        recent_messages = await self.chat_system.get_recent_context("", minutes=2)
        
        # Look for user messages (not from agents or system)
        user_messages = [msg for msg in recent_messages 
                        if not msg.agent_id.startswith("agent_") and 
                        not msg.agent_id == "system"]
        
        if not user_messages:
            return None
        
        # Get the latest user message
        latest_msg = user_messages[-1]
        
        # Check if we've already seen this message
        if latest_msg.id in [req.get('message_id') for req in self.recent_user_requests]:
            return None
        
        # Check if another agent has already responded
        agent_responses = [msg for msg in recent_messages 
                          if msg.agent_id.startswith("agent_") and 
                          msg.timestamp > latest_msg.timestamp]
        
        # If other agents have responded, only respond if specifically mentioned
        if len(agent_responses) > 0:
            if not (f"@{self.agent.id}" in latest_msg.content or 
                   f"@{self.agent.name}" in latest_msg.content.lower()):
                return None
        
        # Analyze the request
        return await self._analyze_user_request(latest_msg)
    
    async def _analyze_user_request(self, message) -> Optional[Dict[str, Any]]:
        """Analyze user request and determine appropriate action"""
        content = message.content.lower()
        
        # Track this request
        self.recent_user_requests.append({
            'message_id': message.id,
            'content': content,
            'timestamp': message.timestamp
        })
        
        # Keep only recent requests
        if len(self.recent_user_requests) > 10:
            self.recent_user_requests = self.recent_user_requests[-10:]
        
        # Parse specific file creation requests
        if "create" in content:
            filename = self._extract_filename(message.content)
            
            if filename:
                # Check if this exact file creation was already requested recently
                recent_creates = [req for req in self.recent_user_requests 
                                if "create" in req['content'] and filename in req['content']]
                if len(recent_creates) > 1:  # Already handled
                    return None
                
                # Determine content type
                if "hello world" in content or "hello" in content:
                    file_content = self.code_generator.generate_hello_world(filename)
                else:
                    file_content = self.code_generator.generate_template(filename)
                
                return {
                    "type": "create_file",
                    "message": message,
                    "file_path": filename,
                    "content": file_content
                }
        
        # Handle help requests
        if "help" in content:
            return {
                "type": "provide_help",
                "message": message
            }
        
        # Handle edit requests
        if "edit" in content:
            filename = self._extract_filename(message.content)
            if filename:
                return {
                    "type": "edit_file",
                    "message": message,
                    "file_path": filename
                }
        
        # General assistance
        return {
            "type": "general_assistance",
            "message": message
        }
    
    def _extract_filename(self, text: str) -> Optional[str]:
        """Extract filename from user message"""
        # Look for common file patterns
        patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*\.py)',  # Python files
            r'([a-zA-Z_][a-zA-Z0-9_]*\.js)',  # JavaScript files
            r'([a-zA-Z_][a-zA-Z0-9_]*\.html)', # HTML files
            r'([a-zA-Z_][a-zA-Z0-9_]*\.css)',  # CSS files
            r'([a-zA-Z_][a-zA-Z0-9_]*\.java)', # Java files
            r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-z]+)', # General files
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Look for specific mentions
        words = text.split()
        for word in words:
            word = word.strip('.,!?')
            if '.' in word and not word.startswith('http'):
                return word
        
        # Default files based on request
        if 'main.py' in text:
            return 'main.py'
        elif 'python' in text.lower():
            return 'main.py'
        elif 'html' in text.lower():
            return 'index.html'
        elif 'javascript' in text.lower() or 'js' in text.lower():
            return 'script.js'
        
        return None
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute the determined action"""
        action_type = action["type"]
        
        try:
            if action_type == "create_file":
                await self._create_file(action)
                self.last_action_time = datetime.now()
                
            elif action_type == "edit_file":
                await self._edit_file(action)
                self.last_action_time = datetime.now()
                
            elif action_type == "provide_help":
                await self._provide_help(action)
                
            elif action_type == "general_assistance":
                await self._provide_general_assistance(action)
                
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
    
    async def _create_file(self, action: Dict[str, Any]):
        """Create a new file with smart content"""
        file_path = action["file_path"]
        content = action["content"]
        message = action["message"]
        
        # Generate AI response for the user
        context = f"I am a {self.agent.role} coding assistant. The user wants me to create {file_path}."
        ai_response = await self.llm_provider.generate_response(message.content, context)
        
        # Send response to user
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=ai_response,
            message_type=MessageType.CHAT,
            parent_message_id=message.id
        )
        
        # Stream the file creation
        await self._stream_file_creation(file_path, content)
    
    async def _edit_file(self, action: Dict[str, Any]):
        """Edit an existing file"""
        file_path = action["file_path"]
        message = action["message"]
        
        # Generate AI response
        context = f"I am a {self.agent.role} coding assistant. The user wants me to edit {file_path}."
        ai_response = await self.llm_provider.generate_response(message.content, context)
        
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=ai_response,
            message_type=MessageType.CHAT,
            parent_message_id=message.id
        )
    
    async def _provide_help(self, action: Dict[str, Any]):
        """Provide help to the user"""
        message = action["message"]
        
        context = f"""I am {self.agent.name}, a {self.agent.role} coding assistant. 
        I can help create files, edit code, and provide coding assistance.
        Available file types: Python (.py), JavaScript (.js), HTML (.html), CSS (.css), Java (.java)
        
        Examples:
        - "create main.py with hello world"
        - "create index.html"
        - "help me with Python"
        """
        
        ai_response = await self.llm_provider.generate_response(message.content, context)
        
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=ai_response,
            message_type=MessageType.CHAT,
            parent_message_id=message.id
        )
    
    async def _provide_general_assistance(self, action: Dict[str, Any]):
        """Provide general assistance"""
        message = action["message"]
        
        context = f"I am {self.agent.name}, a {self.agent.role} coding assistant ready to help with coding tasks."
        ai_response = await self.llm_provider.generate_response(message.content, context)
        
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=ai_response,
            message_type=MessageType.CHAT,
            parent_message_id=message.id
        )
    
    async def _stream_file_creation(self, file_path: str, content: str):
        """Stream file creation character by character"""
        self.agent.status = "editing"
        self.agent.current_file = file_path
        
        # Add as watcher
        await self.streaming_engine.add_watcher(self.agent.id, file_path)
        
        # Announce start
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=f"📝 Creating {file_path}...",
            message_type=MessageType.SYSTEM,
            file_reference=file_path
        )
        
        # Stream content character by character
        for i, char in enumerate(content):
            if not self.is_active:
                break
            
            try:
                await self.streaming_engine.stream_token(
                    agent_id=self.agent.id,
                    file_path=file_path,
                    position=i,
                    token=char,
                    edit_type=EditType.INSERT
                )
                
                # Typing simulation
                await asyncio.sleep(self.typing_speed * random.uniform(0.5, 1.5))
                
            except Exception as e:
                logger.error(f"Error streaming token: {e}")
                break
        
        self.agent.status = "idle"
        self.agent.current_file = None
        
        # Announce completion
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=f"✅ Finished creating {file_path}",
            message_type=MessageType.SYSTEM,
            file_reference=file_path
        )
    
    async def _on_file_edit(self, edit):
        """React to file edits from other agents"""
        if edit.agent_id == self.agent.id:
            return  # Ignore our own edits
        
        # Very rarely comment on others' work
        if random.random() < 0.02:  # Only 2% chance
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"👀 I see work being done on {edit.file_path}",
                message_type=MessageType.CHAT,
                file_reference=edit.file_path
            )
    
    async def _on_chat_message(self, message):
        """React to chat messages"""
        if message.agent_id == self.agent.id:
            return  # Ignore our own messages
        
        # Add to conversation context
        self.conversation_context.append(message)
        
        # Keep context limited
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        # Only respond if specifically mentioned
        if (f"@{self.agent.id}" in message.content or 
            f"@{self.agent.name}" in message.content.lower()):
            await self.chat_queue.put(message)
    
    async def _process_chat_queue(self):
        """Process chat messages directed at this agent"""
        while self.is_active:
            try:
                message = await asyncio.wait_for(self.chat_queue.get(), timeout=1.0)
                await self._handle_direct_mention(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing chat queue: {e}")
    
    async def _handle_direct_mention(self, message):
        """Handle when the agent is directly mentioned"""
        self.agent.status = "thinking"
        
        context = f"I am {self.agent.name}, a {self.agent.role} coding assistant. I was mentioned in a message."
        response = await self.llm_provider.generate_response(message.content, context)
        
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=response,
            message_type=MessageType.CHAT,
            parent_message_id=message.id
        )
        
        self.agent.status = "idle"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "id": self.agent.id,
            "name": self.agent.name,
            "role": self.agent.role,
            "status": self.agent.status,
            "current_file": self.agent.current_file,
            "is_active": self.is_active,
            "ai_provider": self.llm_provider.provider,
            "last_activity": self.agent.last_activity.isoformat() if self.agent.last_activity else None
        }
