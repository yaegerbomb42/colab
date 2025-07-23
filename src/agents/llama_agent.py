import asyncio
import json
import logging
import random
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

from src.core.models import Agent, AgentRole, EditType, MessageType
from src.core.streaming_engine import TokenStreamingEngine
from src.core.chat_system import CollaborativeChatSystem
from src.core.codebase_awareness import CodebaseAwarenessSystem
from src.core.ai_integration import get_ai_manager

logger = logging.getLogger(__name__)

class LlamaAgent:
    """
    Autonomous coding agent powered by Llama 1B model with real-time
    token streaming capabilities and collaborative awareness.
    """
    
    def __init__(self, agent_id: str, name: str, role: AgentRole,
                 streaming_engine: TokenStreamingEngine,
                 chat_system: CollaborativeChatSystem,
                 codebase_system: CodebaseAwarenessSystem,
                 model_name: str = "microsoft/DialoGPT-small"):  # Placeholder, will use actual Llama 1B
        
        self.agent = Agent(id=agent_id, name=name, role=role)
        self.streaming_engine = streaming_engine
        self.chat_system = chat_system
        self.codebase_system = codebase_system
        
        # Model setup
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Agent state
        self.is_active = False
        self.current_task = None
        self.thinking_mode = False
        self.edit_queue = asyncio.Queue()
        self.chat_queue = asyncio.Queue()
        
        # Behavioral parameters
        self.typing_speed = 0.03  # Even faster typing for better demo
        self.think_probability = 0.02  # Much less random thinking
        self.collaboration_frequency = 0.02  # Much less frequent auto-collaboration
        self.autonomous_mode = False  # Start in reactive mode
        self.idle_timeout = 600  # 10 minutes before becoming autonomous
        self.last_action_time = None  # Track last action to avoid spam
        
        # Working memory
        self.short_term_memory = []  # Recent context
        self.file_context = {}  # Files currently working on
        self.conversation_context = []  # Recent chat messages
        
        # Setup callbacks
        streaming_engine.add_edit_listener(self._on_file_edit)
        chat_system.add_message_listener(self._on_chat_message)
    
    async def initialize(self):
        """Initialize the agent's AI model"""
        logger.info(f"Initializing agent {self.agent.name} with model {self.model_name}")
        
        try:
            # In a real implementation, this would load Llama 1B
            # For now, using a smaller model for demonstration
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info(f"Agent {self.agent.name} initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize model for agent {self.agent.name}: {e}")
            logger.info(f"Agent {self.agent.name} will use mock responses for demonstration")
            # Fallback to mock responses for demo
            self.model = None
            self.tokenizer = None
    
    async def start(self):
        """Start the agent's autonomous behavior loop"""
        if self.is_active:
            return
        
        self.is_active = True
        self.agent.status = "active"
        
        logger.info(f"Starting agent {self.agent.name}")
        
        # Start parallel tasks
        tasks = [
            asyncio.create_task(self._main_behavior_loop()),
            asyncio.create_task(self._process_edit_queue()),
            asyncio.create_task(self._process_chat_queue()),
            asyncio.create_task(self._periodic_collaboration_check()),
            asyncio.create_task(self._autonomous_mode_timer())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in agent {self.agent.name}: {e}")
        finally:
            self.is_active = False
            self.agent.status = "inactive"
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        await asyncio.sleep(0.1)  # Allow cleanup
    
    async def _main_behavior_loop(self):
        """Main behavior loop for autonomous actions"""
        while self.is_active:
            try:
                await asyncio.sleep(1.0 + random.uniform(0, 2.0))  # Vary timing
                
                if not self.is_active:
                    break
                
                # Decide what to do based on current state and context
                action = await self._decide_next_action()
                
                if action:
                    await self._execute_action(action)
                
                # Update agent state
                self.agent.last_activity = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in main loop for {self.agent.name}: {e}")
                await asyncio.sleep(5.0)  # Wait before retry
    
    async def _decide_next_action(self) -> Optional[Dict[str, Any]]:
        """Decide what action to take next based on context and role"""
        
        # Prevent too frequent actions
        if self.last_action_time:
            time_since_last = datetime.now() - self.last_action_time
            if time_since_last.total_seconds() < 30:  # Wait at least 30 seconds between actions
                return None
        
        # Check if there are urgent messages or mentions
        recent_messages = await self.chat_system.get_recent_context(
            self.agent.id, minutes=5
        )
        
        mentions = [msg for msg in recent_messages 
                   if f"@{self.agent.id}" in msg.content or f"@{self.agent.name}" in msg.content]
        
        if mentions:
            return {
                "type": "respond_to_mention",
                "message": mentions[-1]
            }
        
        # Look for general requests in recent chat (exclude our own messages)
        user_messages = [msg for msg in recent_messages 
                        if not msg.agent_id.startswith("agent_") and 
                        not msg.agent_id == "system" and 
                        msg.agent_id != self.agent.id]
        
        if user_messages and not self.autonomous_mode:
            # Look for actionable requests that haven't been addressed
            latest_msg = user_messages[-1]
            
            # Check if another agent is already handling this request
            recent_agent_responses = [msg for msg in recent_messages 
                                    if msg.agent_id.startswith("agent_") and 
                                    msg.timestamp > latest_msg.timestamp]
            
            # If no agent has responded yet, and it's an actionable request
            if (len(recent_agent_responses) == 0 and 
                any(keyword in latest_msg.content.lower() for keyword in 
                    ["create", "write", "edit", "fix", "help", "work on", "build"])):
                return {
                    "type": "respond_to_request",
                    "message": latest_msg
                }
        
        # Check current task
        if self.current_task:
            return await self._continue_current_task()
        
        # Only do autonomous work if enabled and no recent user activity
        if self.autonomous_mode and len(user_messages) == 0:
            # Check if other agents are currently active to avoid conflicts
            active_agents = await self._count_active_agents()
            
            # Be more conservative about autonomous work if many agents are active
            if active_agents > 2 and random.random() < 0.7:
                return None
            
            # Look for new work based on role
            if self.agent.role == AgentRole.GENERAL:
                return await self._find_general_work()
            elif self.agent.role == AgentRole.UI:
                return await self._find_ui_work()
            elif self.agent.role == AgentRole.LINTER:
                return await self._find_linting_work()
        
        # Occasionally introduce self if new to conversation
        if (len(self.conversation_context) == 0 and 
            random.random() < 0.05 and  # Reduced probability
            len(recent_messages) > 0):
            return {"type": "introduce_self"}
        
        return None
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute a decided action"""
        action_type = action["type"]
        
        try:
            if action_type == "respond_to_mention":
                await self._respond_to_mention(action["message"])
            elif action_type == "respond_to_request":
                await self._respond_to_user_request(action["message"])
            elif action_type == "edit_file":
                await self._stream_file_edit(action)
                self.last_action_time = datetime.now()  # Track file edits
            elif action_type == "create_file":
                await self._create_new_file(action)
                self.last_action_time = datetime.now()  # Track file creation
            elif action_type == "start_conversation":
                await self._start_conversation(action.get("topic"))
            elif action_type == "introduce_self":
                await self._introduce_self()
            elif action_type == "explore_codebase":
                await self._explore_codebase()
            elif action_type == "continue_task":
                await self._continue_task_work(action)
            else:
                logger.warning(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
    
    async def _stream_file_edit(self, action: Dict[str, Any]):
        """Stream edit to a file token by token"""
        file_path = action["file_path"]
        content = action["content"]
        position = action.get("position", 0)
        edit_type = action.get("edit_type", EditType.INSERT)
        
        self.agent.status = "editing"
        self.agent.current_file = file_path
        
        # Add as watcher
        await self.streaming_engine.add_watcher(self.agent.id, file_path)
        
        # Announce what we're doing
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=f"Starting to edit {file_path}",
            message_type=MessageType.SYSTEM,
            file_reference=file_path
        )
        
        # Stream tokens one by one
        for i, char in enumerate(content):
            if not self.is_active:
                break
            
            try:
                await self.streaming_engine.stream_token(
                    agent_id=self.agent.id,
                    file_path=file_path,
                    position=position + i,
                    token=char,
                    edit_type=edit_type
                )
                
                # Simulate typing speed
                await asyncio.sleep(self.typing_speed * random.uniform(0.5, 1.5))
                
            except Exception as e:
                logger.error(f"Error streaming token: {e}")
                break
        
        self.agent.status = "idle"
        self.agent.current_file = None
        
        # Announce completion
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=f"Finished editing {file_path}",
            message_type=MessageType.SYSTEM,
            file_reference=file_path
        )
    
    async def _respond_to_mention(self, message):
        """Respond to being mentioned in chat"""
        self.agent.status = "thinking"
        
        # Analyze the message and generate response
        response = await self._generate_response_to_message(message)
        
        if response:
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=response,
                message_type=MessageType.CHAT,
                parent_message_id=message.id
            )
        
        self.agent.status = "idle"
    
    async def _respond_to_user_request(self, message):
        """Respond to a user request that's not a direct mention"""
        self.agent.status = "thinking"
        
        content = message.content.lower()
        response = None
        action = None
        
        # Check if other agents have already responded to this message
        recent_messages = await self.chat_system.get_recent_context("", minutes=1)
        recent_responses = [msg for msg in recent_messages 
                          if msg.timestamp > message.timestamp and 
                          msg.agent_id.startswith("agent_") and 
                          msg.agent_id != self.agent.id]
        
        # If other agents have already responded, don't duplicate
        if len(recent_responses) > 0:
            self.agent.status = "idle"
            return
        
        # Parse user intent more intelligently
        if "create" in content and ("python" in content or "file" in content):
            # Extract filename if mentioned, or create appropriate one
            words = message.content.split()
            filename = None
            
            # Look for specific filename
            for word in words:
                if word.endswith('.py') or word.endswith('.html') or word.endswith('.js') or word.endswith('.css'):
                    filename = word.strip(".,!?")
                    break
            
            # If no specific filename, create based on context
            if not filename:
                if "main" in content:
                    filename = "main.py"
                elif "hello" in content or "world" in content:
                    filename = "hello_world.py"
                elif self.agent.role == AgentRole.UI:
                    filename = "index.html"
                else:
                    filename = "new_file.py"
            
            # Check if the file actually exists
            existing_files = await self._get_existing_files()
            if filename in existing_files:
                response = f"I see {filename} already exists. Would you like me to edit it or create a different file?"
            else:
                # Create appropriate content based on request
                if "hello" in content and "world" in content:
                    file_content = self._get_hello_world_content(filename)
                    response = f"I'll create {filename} with a Hello World program!"
                else:
                    file_content = self._get_starter_content(filename)
                    response = f"I'll create {filename} for you!"
                
                action = {
                    "type": "create_file",
                    "file_path": filename,
                    "content": file_content
                }
        
        elif "hello" in content and "world" in content:
            # User wants hello world specifically
            filename = "hello_world.py" if "main.py" not in content else "main.py"
            response = f"I'll create {filename} with a Hello World program!"
            action = {
                "type": "create_file",
                "file_path": filename,
                "content": self._get_hello_world_content(filename)
            }
        
        elif "help" in content:
            response = f"I'm {self.agent.name}, a {self.agent.role} agent. I can create files, write code, and help with development tasks. Try asking me to 'create main.py with hello world' or 'create a Python file'."
        
        elif any(word in content for word in ["write", "code", "implement"]):
            response = f"I'd be happy to help with coding! Please specify what you'd like me to create. For example: 'create main.py with hello world' or 'create a Python calculator'."
        
        else:
            # More helpful generic response
            response = f"I can help you with coding tasks! Try asking me to:\n• 'create main.py with hello world'\n• 'create a Python file'\n• 'help me with [specific task]'"
        
        # Send response
        if response:
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=response,
                message_type=MessageType.CHAT,
                parent_message_id=message.id
            )
        
        # Execute action if needed
        if action:
            await asyncio.sleep(1)  # Brief pause before action
            await self._execute_action(action)
        
        self.agent.status = "idle"
    
    async def _introduce_self(self):
        """Introduce the agent to the conversation"""
        role_descriptions = {
            AgentRole.GENERAL: "general development tasks",
            AgentRole.UI: "user interface design and frontend development",
            AgentRole.SECURITY: "security analysis and vulnerability detection",
            AgentRole.LINTER: "code quality and style enforcement",
            AgentRole.TEST: "testing and quality assurance"
        }
        
        description = role_descriptions.get(self.agent.role, "various coding tasks")
        
        intro = f"Hi! I'm {self.agent.name}, and I specialize in {description}. I'm ready to help with collaborative coding. Just let me know what you'd like me to work on!"
        
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=intro,
            message_type=MessageType.CHAT
        )
    
    def _get_hello_world_content(self, filename: str) -> str:
        """Generate Hello World content based on file type"""
        ext = filename.split('.')[-1].lower() if '.' in filename else 'py'
        
        if ext == 'py':
            return f'# {filename}\n# Hello World program created by {self.agent.name}\n\ndef main():\n    """Simple Hello World function"""\n    print("Hello, World!")\n    print("Welcome to collaborative coding!")\n\nif __name__ == "__main__":\n    main()\n'
        elif ext in ['html', 'htm']:
            return f'<!DOCTYPE html>\n<html>\n<head>\n    <title>Hello World</title>\n    <meta charset="utf-8">\n</head>\n<body>\n    <h1>Hello, World!</h1>\n    <p>Created by {self.agent.name}</p>\n    <p>Welcome to collaborative coding!</p>\n</body>\n</html>\n'
        elif ext == 'js':
            return f'// {filename} - Hello World in JavaScript\n// Created by {self.agent.name}\n\nconsole.log("Hello, World!");\nconsole.log("Welcome to collaborative coding!");\n\n// Simple greeting function\nfunction greet(name = "World") {{\n    return `Hello, ${{name}}!`;\n}}\n\nconsole.log(greet());\n'
        else:
            return f'# {filename}\n# Hello World program\n# Created by {self.agent.name}\n\nprint("Hello, World!")\nprint("Welcome to collaborative coding!")\n'
    
    def _get_starter_content(self, filename: str) -> str:
        """Generate appropriate starter content based on file type and agent role"""
        ext = filename.split('.')[-1].lower() if '.' in filename else 'txt'
        
        if ext == 'py':
            if self.agent.role == AgentRole.TEST:
                return f"# Test file for {filename.replace('test_', '').replace('.py', '')}\nimport unittest\n\nclass Test{filename.replace('test_', '').replace('.py', '').title()}(unittest.TestCase):\n    def test_example(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n"
            else:
                return f"# {filename}\n# Created by {self.agent.name}\n\ndef main():\n    \"\"\"Main function\"\"\"\n    print('Hello from {filename}!')\n\nif __name__ == '__main__':\n    main()\n"
        
        elif ext in ['html', 'htm']:
            return f"<!DOCTYPE html>\n<html>\n<head>\n    <title>{filename}</title>\n    <meta charset='utf-8'>\n</head>\n<body>\n    <h1>Welcome</h1>\n    <p>Created by {self.agent.name}</p>\n</body>\n</html>\n"
        
        elif ext == 'css':
            return f"/* {filename} - Created by {self.agent.name} */\n\nbody {{\n    font-family: Arial, sans-serif;\n    margin: 0;\n    padding: 20px;\n}}\n\nh1 {{\n    color: #333;\n}}\n"
        
        elif ext == 'js':
            return f"// {filename} - Created by {self.agent.name}\n\n(function() {{\n    'use strict';\n    \n    console.log('Hello from {filename}!');\n    \n    // Your code here\n    \n}})();\n"
        
        elif ext == 'md':
            return f"# {filename.replace('.md', '').replace('_', ' ').title()}\n\nCreated by {self.agent.name}\n\n## Overview\n\nThis document describes...\n"
        
        else:
            return f"# {filename}\n# Created by {self.agent.name}\n\n"
    
    async def _generate_response_to_message(self, message) -> str:
        """Generate a contextual response to a chat message"""
        try:
            # Use AI integration for better responses
            ai_manager = get_ai_manager()
            
            context = {
                'agent_role': str(self.agent.role),
                'current_files': await self._get_existing_files(),
                'recent_messages': [msg.content for msg in self.conversation_context[-3:]]
            }
            
            prompt = f"A user mentioned me in a collaborative coding chat: '{message.content}'. How should I respond as a {self.agent.role} agent?"
            
            response = await ai_manager.generate_response(prompt, context)
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            # Fallback to rule-based responses
            return self._generate_fallback_response(message)
    
    def _generate_fallback_response(self, message) -> str:
        """Fallback rule-based response generation"""
        content = message.content.lower()
        
        if self.agent.role == AgentRole.SECURITY:
            if any(word in content for word in ["security", "vulnerability", "attack", "sql injection"]):
                return f"I'll review the security implications. Let me check for potential vulnerabilities."
            
        elif self.agent.role == AgentRole.UI:
            if any(word in content for word in ["ui", "interface", "component", "style", "css"]):
                return f"I can help with the UI implementation. What specific component needs work?"
        
        elif self.agent.role == AgentRole.LINTER:
            if any(word in content for word in ["error", "warning", "lint", "code quality"]):
                return f"I'll analyze the code quality and fix any linting issues."
        
        elif self.agent.role == AgentRole.TEST:
            if any(word in content for word in ["test", "testing", "bug", "error"]):
                return f"I'll write tests for this functionality. What should I focus on?"
        
        # General responses
        responses = [
            f"Thanks for mentioning me! I'll take a look at this.",
            f"Understood. Let me work on that.",
            f"I can help with that. Give me a moment.",
            f"Interesting point. Let me investigate."
        ]
        
        return random.choice(responses)
    
    async def _find_general_work(self) -> Optional[Dict[str, Any]]:
        """Find general development work to do"""
        # Look for files that need work
        project_overview = await self.codebase_system.get_project_overview()
        
        # Check if there are recent file creation activities to avoid duplicates
        recent_messages = await self.chat_system.get_recent_context("", minutes=5)  # Increased time window
        recent_file_creations = [msg for msg in recent_messages 
                               if "Starting to edit" in msg.content or "create" in msg.content.lower()]
        
        if len(recent_file_creations) > 0:
            # Other agents are creating files, don't duplicate
            return None
        
        # Only create files if workspace is truly empty
        if project_overview["total_files"] == 0:
            # Very low probability to avoid immediate creation
            if random.random() < 0.05:  # Only 5% chance
                return {
                    "type": "create_file", 
                    "file_path": "main.py",
                    "content": "# Main application file\n\ndef main():\n    print('Hello, collaborative coding!')\n\nif __name__ == '__main__':\n    main()\n"
                }
        
        # Disabled random functionality additions to prevent spam
        # if random.random() < 0.01:  # Reduced from 0.1 to 0.01
        #     return {
        #         "type": "edit_file",
        #         "file_path": "main.py",
        #         "content": "\n# Adding new functionality\n",
        #         "position": -1  # Append to end
        #     }
        
        return None
    
    async def _find_ui_work(self) -> Optional[Dict[str, Any]]:
        """Find UI-related work"""
        # Check if there are recent file creation activities to avoid duplicates
        recent_messages = await self.chat_system.get_recent_context("", minutes=2)
        recent_file_creations = [msg for msg in recent_messages 
                               if "Starting to edit" in msg.content or "create" in msg.content.lower()]
        
        if len(recent_file_creations) > 0:
            # Other agents are creating files, don't duplicate
            return None
        
        # Look for HTML/CSS/JS files or create them
        ui_files = []
        for file_path, file_info in self.codebase_system.file_index.items():
            if file_info.get('language') in ['html', 'css', 'javascript']:
                ui_files.append(file_path)
        
        if not ui_files:
            # Create a basic HTML file
            return {
                "type": "create_file",
                "file_path": "index.html",
                "content": "<!DOCTYPE html>\n<html>\n<head>\n    <title>Collaborative App</title>\n</head>\n<body>\n    <h1>Welcome to Collaborative Coding</h1>\n</body>\n</html>\n"
            }
        
        return None
    
    async def _find_security_work(self) -> Optional[Dict[str, Any]]:
        """Find security-related work"""
        # Scan for potential security issues
        # This is a simplified example
        return None
    
    async def _find_linting_work(self) -> Optional[Dict[str, Any]]:
        """Find linting/code quality work"""
        # Check for code quality issues
        return None
    
    async def _find_testing_work(self) -> Optional[Dict[str, Any]]:
        """Find testing work to do"""
        # Look for files that need tests
        return None
    
    async def _create_new_file(self, action: Dict[str, Any]):
        """Create a new file with content"""
        file_path = action["file_path"]
        content = action["content"]
        
        # Stream the file creation
        await self._stream_file_edit({
            "file_path": file_path,
            "content": content,
            "position": 0,
            "edit_type": EditType.INSERT
        })
    
    async def _start_conversation(self, topic: Optional[str] = None):
        """Start a conversation in chat"""
        if not topic:
            topics = [
                "What should we work on next?",
                "I'm looking at the current codebase structure.",
                "Anyone need help with their current task?",
                "Let's discuss the project architecture.",
                "I have some ideas for improvements."
            ]
            topic = random.choice(topics)
        
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=topic,
            message_type=MessageType.CHAT
        )
    
    async def _explore_codebase(self):
        """Explore and analyze the codebase"""
        self.agent.status = "reviewing"
        
        # Get project overview
        overview = await self.codebase_system.get_project_overview()
        
        # Comment on findings
        if overview["total_files"] > 0:
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"Project overview: {overview['total_files']} files, "
                       f"{overview['total_functions']} functions, "
                       f"{overview['total_classes']} classes. "
                       f"Main languages: {', '.join(overview['languages'].keys())}",
                message_type=MessageType.SYSTEM
            )
        
        self.agent.status = "idle"
    
    async def _continue_current_task(self) -> Optional[Dict[str, Any]]:
        """Continue working on current task"""
        # Placeholder for task continuation logic
        return None
    
    async def _continue_task_work(self, action: Dict[str, Any]):
        """Continue working on a specific task"""
        # Placeholder for specific task work
        pass
    
    async def _on_file_edit(self, edit):
        """React to file edits from other agents"""
        if edit.agent_id == self.agent.id:
            return  # Ignore our own edits
        
        # Add to our context
        self.short_term_memory.append({
            "type": "file_edit",
            "agent_id": edit.agent_id,
            "file_path": edit.file_path,
            "timestamp": edit.timestamp
        })
        
        # Keep memory limited
        if len(self.short_term_memory) > 50:
            self.short_term_memory = self.short_term_memory[-50:]
        
        # Maybe react to the edit
        if random.random() < 0.1 and edit.file_path in self.file_context:
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"I see changes being made to {edit.file_path}",
                message_type=MessageType.CHAT,
                file_reference=edit.file_path
            )
    
    async def _on_chat_message(self, message):
        """React to chat messages from other agents"""
        if message.agent_id == self.agent.id:
            return  # Ignore our own messages
        
        # Add to conversation context
        self.conversation_context.append(message)
        
        # Keep context limited
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]
        
        # Check if we should respond
        if (f"@{self.agent.id}" in message.content or 
            f"@{self.agent.name}" in message.content):
            await self.chat_queue.put(message)
    
    async def _count_active_agents(self) -> int:
        """Count how many agents are currently active"""
        recent_messages = await self.chat_system.get_recent_context("", minutes=5)
        active_agent_ids = set()
        
        for msg in recent_messages:
            if msg.agent_id.startswith("agent_"):
                active_agent_ids.add(msg.agent_id)
        
        return len(active_agent_ids)
    
    async def _get_existing_files(self) -> List[str]:
        """Get list of existing files in the workspace"""
        try:
            # Check actual filesystem
            import os
            workspace_path = "./workspace"
            if os.path.exists(workspace_path):
                return [f for f in os.listdir(workspace_path) 
                       if os.path.isfile(os.path.join(workspace_path, f))]
            return []
        except Exception as e:
            logger.error(f"Error checking existing files: {e}")
            # Fallback to codebase system
            return list(self.codebase_system.file_index.keys())
    
    async def _process_edit_queue(self):
        """Process queued edits"""
        while self.is_active:
            try:
                edit = await asyncio.wait_for(self.edit_queue.get(), timeout=1.0)
                await self._execute_action(edit)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing edit queue: {e}")
    
    async def _process_chat_queue(self):
        """Process queued chat responses"""
        while self.is_active:
            try:
                message = await asyncio.wait_for(self.chat_queue.get(), timeout=1.0)
                await self._respond_to_mention(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing chat queue: {e}")
    
    async def _periodic_collaboration_check(self):
        """Periodically check for collaboration opportunities"""
        while self.is_active:
            try:
                await asyncio.sleep(60.0 + random.uniform(0, 60.0))  # Every 60-120 seconds (reduced frequency)
                
                if random.random() < self.collaboration_frequency:
                    # Check recent activity and maybe contribute
                    recent_messages = await self.chat_system.get_recent_context("", minutes=15)
                    
                    # Only contribute if there's been some activity but not too much chatter
                    agent_messages = [msg for msg in recent_messages if msg.agent_id.startswith("agent_")]
                    user_messages = [msg for msg in recent_messages if not msg.agent_id.startswith("agent_") and msg.agent_id != "system"]
                    
                    if len(user_messages) > 0 and len(agent_messages) < 3 and random.random() < 0.2:
                        # Join ongoing conversation only if not too much agent chatter
                        await self.chat_system.send_message(
                            agent_id=self.agent.id,
                            content="I'm following the discussion. Let me know if I can help!",
                            message_type=MessageType.CHAT
                        )
                
            except Exception as e:
                logger.error(f"Error in collaboration check: {e}")
    
    async def _autonomous_mode_timer(self):
        """Enable autonomous mode after idle timeout"""
        while self.is_active:
            try:
                await asyncio.sleep(self.idle_timeout)
                
                # Check if there's been recent user activity
                recent_messages = await self.chat_system.get_recent_context("", minutes=10)
                user_messages = [msg for msg in recent_messages 
                               if not msg.agent_id.startswith("agent_") and not msg.agent_id == "system"]
                
                if len(user_messages) == 0 and not self.autonomous_mode:
                    logger.info(f"Agent {self.agent.name} entering autonomous mode after idle timeout")
                    self.autonomous_mode = True
                    
                    await self.chat_system.send_message(
                        agent_id=self.agent.id,
                        content="I haven't seen much activity lately. I'll work on some tasks autonomously. Feel free to give me directions anytime!",
                        message_type=MessageType.SYSTEM
                    )
                
            except Exception as e:
                logger.error(f"Error in autonomous mode timer: {e}")
    
    def assign_task(self, task_description: str, files: List[str] = None):
        """Assign a specific task to this agent"""
        self.current_task = {
            "description": task_description,
            "files": files or [],
            "assigned_at": datetime.now()
        }
        
        # Update agent goals
        self.agent.goals.append(task_description)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "id": self.agent.id,
            "name": self.agent.name,
            "role": self.agent.role,
            "status": self.agent.status,
            "current_file": self.agent.current_file,
            "current_task": self.current_task,
            "is_active": self.is_active,
            "last_activity": self.agent.last_activity.isoformat() if self.agent.last_activity else None,
            "goals": self.agent.goals
        }
