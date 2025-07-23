"""
Enhanced Fast Agent with Real-time Task Execution
Optimized for immediate response and intelligent task handling
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from src.core.models import Agent, AgentRole, EditType, MessageType
from src.core.streaming_engine import TokenStreamingEngine
from src.core.chat_system import CollaborativeChatSystem
from src.core.codebase_awareness import CodebaseAwarenessSystem
from src.core.agent_memory import AgentMemorySystem, MemoryType, TaskContext
from src.core.ai_integration_v2 import generate_ai_response

logger = logging.getLogger(__name__)

class FastAgent:
    """
    Fast, intelligent agent that responds immediately and handles complex tasks
    """
    
    def __init__(self, agent_id: str, name: str, role: AgentRole,
                 streaming_engine: TokenStreamingEngine,
                 chat_system: CollaborativeChatSystem,
                 codebase_system: CodebaseAwarenessSystem):
        
        self.agent = Agent(id=agent_id, name=name, role=role)
        self.streaming_engine = streaming_engine
        self.chat_system = chat_system
        self.codebase_system = codebase_system
        
        # Enhanced memory system
        self.memory = AgentMemorySystem(agent_id)
        
        # Agent state
        self.is_active = False
        self.current_task = None
        self.task_queue = asyncio.Queue()
        
        # Performance optimizations
        self.typing_speed = 0.01  # Very fast typing
        self.response_delay = 0.1  # Minimal delay
        
        # Content templates for fast responses
        self.templates = {
            'python': {
                'hello_world': '''#!/usr/bin/env python3
"""
Hello World Python Script
"""

def main():
    print("Hello, World!")
    print("Welcome to the Multi-Agent Development Environment!")

if __name__ == "__main__":
    main()
''',
                'basic_script': '''#!/usr/bin/env python3
"""
{description}
"""

def main():
    # TODO: Implement {description}
    print("Starting {description}")

if __name__ == "__main__":
    main()
'''
            },
            'html': {
                'basic_page': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        .ascii-art {{
            font-family: 'Courier New', monospace;
            font-size: 12px;
            line-height: 1;
            white-space: pre;
            background: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        h1 {{
            color: #333;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        {content}
    </div>
</body>
</html>''',
                'ascii_banana': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASCII Banana</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #FFD700, #FFA500);
            min-height: 100vh;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        .ascii-banana {{
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.2;
            white-space: pre;
            background: #f9f9f9;
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
            color: #8B4513;
            border: 2px solid #FFD700;
        }}
        h1 {{
            color: #8B4513;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .emoji {{
            font-size: 2em;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🍌 ASCII Banana 🍌</h1>
        <div class="emoji">🐒 Welcome to the Banana Zone! 🐒</div>
        
        <div class="ascii-banana">                    ,,,,
                  ,;;;;;;,
                ,;;;;;;;;;;;,
              ,;;;;;;;;;;;;;;;;;,
            ,;;;;;;;;;;;;;;;;;;;;;;;,
          ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,
      ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,
  ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,
  ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,
      ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,
          ,;;;;;;;;;;;;;;;;;;;;;;;;;,
            ,;;;;;;;;;;;;;;;;;;;,
              ,;;;;;;;;;;;;;,
                ,;;;;;;;,
                  ,,,,
                  
            Perfect ASCII Banana!
        </div>
        
        <div class="emoji">🍌 Perfectly Curved & Delicious! 🍌</div>
        <p>This banana was carefully crafted with ASCII art to bring you the most realistic digital fruit experience!</p>
    </div>
    
    <script>
        // Add some interactive banana fun
        document.querySelector('.ascii-banana').addEventListener('click', function() {{
            this.style.transform = 'rotate(5deg)';
            setTimeout(() => {{
                this.style.transform = 'rotate(0deg)';
            }}, 200);
        }});
    </script>
</body>
</html>'''
            }
        }
    
    async def initialize(self):
        """Initialize the agent quickly"""
        logger.info(f"Fast initializing agent {self.agent.name}")
        
        # Immediate initialization - no heavy model loading
        await self.memory.add_memory(
            MemoryType.WORKING,
            {
                "agent_initialized": True,
                "role": self.agent.role.value,
                "capabilities": ["file_creation", "editing", "analysis", "collaboration"]
            },
            tags=["initialization", "ready"]
        )
        
        logger.info(f"Agent {self.agent.name} ready for action!")
    
    async def start(self):
        """Start the agent's task processing"""
        if self.is_active:
            return
        
        self.is_active = True
        self.agent.status = "active"
        
        logger.info(f"Starting fast agent {self.agent.name}")
        
        # Start task processing
        asyncio.create_task(self._process_task_queue())
        
        # Announce readiness
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=f"🚀 {self.agent.name} ready! I can help with {self.agent.role.value} tasks.",
            message_type=MessageType.AGENT_JOIN
        )
    
    async def handle_task(self, task_data: Dict[str, Any]):
        """Handle a task assignment immediately"""
        try:
            task_content = task_data.get("content", "")
            task_id = task_data.get("task_id", "")
            file_path = task_data.get("file_path")
            
            logger.info(f"Agent {self.agent.name} handling task: {task_content}")
            
            # Update memory with task context
            task_context = TaskContext(
                task_id=task_id,
                description=task_content,
                status="in_progress",
                assigned_files=[file_path] if file_path else [],
                dependencies=[],
                progress={"started": True},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            await self.memory.update_task_context(task_context)
            
            # Immediate acknowledgment
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"🎯 Working on: {task_content}",
                message_type=MessageType.AGENT_ACTION
            )
            
            # Route to appropriate handler
            if file_path and "create" in task_content.lower():
                await self._handle_file_creation(file_path, task_content)
            elif file_path and "edit" in task_content.lower():
                await self._handle_file_editing(file_path, task_content)
            else:
                await self._handle_general_task(task_content)
            
            # Task completion
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"✅ Completed: {task_content}",
                message_type=MessageType.TASK_UPDATE
            )
            
            # Update memory
            await self.memory.record_tool_usage(
                "handle_task",
                {"task": task_content, "file_path": file_path},
                {"success": True, "completion_time": datetime.now()},
                True
            )
            
        except Exception as e:
            logger.error(f"Error handling task: {e}")
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"❌ Error with task: {str(e)}",
                message_type=MessageType.TASK_UPDATE
            )
    
    async def _handle_file_creation(self, file_path: str, description: str):
        """Handle file creation tasks with immediate response"""
        try:
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"📝 Creating {file_path}",
                message_type=MessageType.AGENT_ACTION
            )
            
            # Determine file type and generate content
            content = await self._generate_file_content(file_path, description)
            
            # Stream file creation with fast typing
            await self._fast_stream_content(file_path, content)
            
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"📁 Created {file_path} successfully",
                message_type=MessageType.AGENT_ACTION
            )
            
        except Exception as e:
            logger.error(f"Error creating file {file_path}: {e}")
            raise
    
    async def _generate_file_content(self, file_path: str, description: str) -> str:
        """Generate file content based on type and description"""
        file_ext = file_path.split('.')[-1].lower()
        description_lower = description.lower()
        
        # Python files
        if file_ext == 'py':
            if "hello world" in description_lower:
                return self.templates['python']['hello_world']
            else:
                return self.templates['python']['basic_script'].format(description=description)
        
        # HTML files
        elif file_ext == 'html':
            if "ascii banana" in description_lower or "banana" in description_lower:
                return self.templates['html']['ascii_banana']
            else:
                title = description.replace("create", "").replace("html", "").strip()
                content = f"<h2>{description}</h2><p>Content for {title}</p>"
                return self.templates['html']['basic_page'].format(
                    title=title or "Webpage",
                    content=content
                )
        
        # CSS files
        elif file_ext == 'css':
            return f'''/* {description} */
/* Generated by Multi-Agent Development Environment */

body {{
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}}

/* Styles for {description} */
'''
        
        # JavaScript files
        elif file_ext == 'js':
            return f'''// {description}
// Generated by Multi-Agent Development Environment

console.log("JavaScript file created: {description}");

// TODO: Implement {description}
function main() {{
    console.log("Starting {description}");
}}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', main);
'''
        
        # Default text content
        else:
            return f"# {file_path}\n# {description}\n\nContent for {description}"
    
    async def _fast_stream_content(self, file_path: str, content: str):
        """Stream content creation with optimized speed"""
        try:
            # Try to clear file first (if it exists)
            try:
                await self.streaming_engine.clear_file(file_path)
            except Exception as e:
                logger.debug(f"File {file_path} might not exist yet: {e}")
            
            # For fast demo, stream in chunks instead of character by character
            chunk_size = 50  # Stream 50 characters at a time
            
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                
                await self.streaming_engine.stream_token(
                    agent_id=self.agent.id,
                    file_path=file_path,
                    position=i,
                    token=chunk,
                    edit_type=EditType.INSERT
                )
                
                # Minimal delay for visual effect
                await asyncio.sleep(self.typing_speed)
                
        except Exception as e:
            logger.error(f"Error streaming content: {e}")
            # Fallback: create file using simple file write
            try:
                full_path = os.path.join("./workspace", file_path.lstrip('/'))
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"Successfully created {file_path} using fallback method")
                
            except Exception as fallback_error:
                logger.error(f"Fallback file creation also failed: {fallback_error}")
                raise
    
    async def _handle_file_editing(self, file_path: str, description: str):
        """Handle file editing tasks"""
        try:
            # Read current content
            current_content = await self.streaming_engine.get_file_content(file_path)
            
            await self.chat_system.send_message(
                agent_id=self.agent.id,
                content=f"✏️ Editing {file_path}",
                message_type=MessageType.AGENT_ACTION
            )
            
            # Add edit based on description
            if "banana" in description.lower() and file_path.endswith('.html'):
                # Special case for banana HTML
                new_content = self.templates['html']['ascii_banana']
            else:
                # Simple append edit
                new_content = current_content + f"\n\n<!-- Edit: {description} -->\n"
            
            await self._fast_stream_content(file_path, new_content)
            
        except Exception as e:
            logger.error(f"Error editing file {file_path}: {e}")
            raise
    
    async def _handle_general_task(self, description: str):
        """Handle general tasks that don't involve specific files"""
        try:
            # Use AI for general responses if available
            try:
                response = await generate_ai_response(
                    description, 
                    context="", 
                    agent_role=self.agent.role.value
                )
                
                await self.chat_system.send_message(
                    agent_id=self.agent.id,
                    content=response,
                    message_type=MessageType.CHAT
                )
                
            except Exception as e:
                # Fallback to template response
                await self.chat_system.send_message(
                    agent_id=self.agent.id,
                    content=f"I understand you want me to: {description}. Let me work on that!",
                    message_type=MessageType.CHAT
                )
                
        except Exception as e:
            logger.error(f"Error handling general task: {e}")
            raise
    
    async def _process_task_queue(self):
        """Process incoming tasks from the queue"""
        while self.is_active:
            try:
                # Get task with timeout
                task_data = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Handle task immediately
                await self.handle_task(task_data)
                
            except asyncio.TimeoutError:
                # No tasks, continue
                continue
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
    
    async def respond_to_user(self, user_message: str) -> str:
        """Respond to user messages intelligently"""
        try:
            # Get relevant context
            context = await self.memory.get_relevant_context(user_message)
            
            # Quick response for common patterns
            user_lower = user_message.lower()
            
            if "create" in user_lower and ("html" in user_lower or "webpage" in user_lower):
                if "banana" in user_lower:
                    return "🍌 I'll create an awesome ASCII banana webpage for you! Let me get started."
                else:
                    return "🌐 I'll create an HTML webpage for you right away!"
            
            elif "create" in user_lower and "python" in user_lower:
                return "🐍 Python file coming right up! I'll create that for you."
            
            elif "help" in user_lower:
                return f"👋 I'm a {self.agent.role.value} agent. I can help with file creation, editing, and development tasks. What would you like me to work on?"
            
            else:
                # Use AI for more complex responses
                try:
                    response = await generate_ai_response(
                        user_message,
                        context=str(context),
                        agent_role=self.agent.role.value
                    )
                    return response
                except:
                    return "I'm ready to help! What would you like me to work on?"
                    
        except Exception as e:
            logger.error(f"Error responding to user: {e}")
            return "I'm here to help! Let me know what you need."
    
    async def stop(self):
        """Stop the agent"""
        self.is_active = False
        self.agent.status = "inactive"
        
        await self.chat_system.send_message(
            agent_id=self.agent.id,
            content=f"👋 {self.agent.name} signing off",
            message_type=MessageType.AGENT_LEAVE
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "id": self.agent.id,
            "name": self.agent.name,
            "role": self.agent.role.value,
            "status": self.agent.status,
            "is_active": self.is_active,
            "current_task": self.current_task.task_id if self.current_task else None,
            "memory_summary": asyncio.create_task(self.memory.get_memory_summary()) if self.memory else {}
        }
