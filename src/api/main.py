import asyncio
import json
import logging
import os
from typing import Dict, List, Set, Optional
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from src.core.models import Agent, AgentRole, EditType, MessageType, FileEdit, ChatMessage
from src.core.streaming_engine import TokenStreamingEngine
from src.core.chat_system import CollaborativeChatSystem
from src.core.codebase_awareness import CodebaseAwarenessSystem
from src.core.task_coordinator import get_task_coordinator
from src.agents.llama_agent import LlamaAgent

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time collaboration"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_connections: Dict[str, str] = {}  # agent_id -> connection_id
        self.connection_agents: Dict[str, str] = {}  # connection_id -> agent_id
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Connect a new WebSocket"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info(f"WebSocket connected: {connection_id}")
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.connection_agents:
            agent_id = self.connection_agents[connection_id]
            del self.connection_agents[connection_id]
            if agent_id in self.agent_connections:
                del self.agent_connections[agent_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    def associate_agent(self, connection_id: str, agent_id: str):
        """Associate a connection with an agent"""
        self.agent_connections[agent_id] = connection_id
        self.connection_agents[connection_id] = agent_id
    
    async def send_to_connection(self, connection_id: str, message: dict):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending to connection {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def send_to_agent(self, agent_id: str, message: dict):
        """Send message to specific agent"""
        if agent_id in self.agent_connections:
            connection_id = self.agent_connections[agent_id]
            await self.send_to_connection(connection_id, message)
    
    async def broadcast(self, message: dict, exclude_connection: Optional[str] = None):
        """Broadcast message to all connections"""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            if connection_id == exclude_connection:
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)

class CollaborativeAPI:
    """Main API class for the collaborative development environment"""
    
    def __init__(self):
        self.app = FastAPI(title="Multi-Agent Live Development Environment")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Core systems
        self.streaming_engine = TokenStreamingEngine()
        self.chat_system = CollaborativeChatSystem()
        self.codebase_system = CodebaseAwarenessSystem()
        
        # Task coordination system
        self.task_coordinator = None
        
        # WebSocket management
        self.ws_manager = WebSocketManager()
        
        # Agent management (now handled by task coordinator)
        self.agents: Dict[str, LlamaAgent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        
        # Setup event listeners
        self.streaming_engine.add_edit_listener(self._on_file_edit)
        self.chat_system.add_message_listener(self._on_chat_message)
        
        # Setup routes
        self._setup_routes()
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/favicon.ico")
        async def get_favicon():
            """Serve favicon to avoid 404 errors"""
            return HTMLResponse("", status_code=204)
        
        @self.app.get("/")
        async def get_main():
            """Serve the main interface"""
            return HTMLResponse(self._get_main_html())
        
        @self.app.websocket("/ws/{connection_id}")
        async def websocket_endpoint(websocket: WebSocket, connection_id: str):
            await self.ws_manager.connect(websocket, connection_id)
            
            try:
                # Send initial connection confirmation
                await self.ws_manager.send_to_connection(connection_id, {
                    "type": "connected",
                    "connection_id": connection_id
                })
                
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self._handle_websocket_message(connection_id, message)
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                self.ws_manager.disconnect(connection_id)
            except Exception as e:
                logger.error(f"WebSocket error for {connection_id}: {e}")
                self.ws_manager.disconnect(connection_id)
        
        @self.app.post("/api/agents/create")
        async def create_agent(agent_data: dict):
            """Create a new agent"""
            try:
                agent_id = agent_data.get("id", f"agent_{len(self.agents)}")
                name = agent_data.get("name", f"Agent {len(self.agents) + 1}")
                role = AgentRole(agent_data.get("role", "general"))
                
                agent = LlamaAgent(
                    agent_id=agent_id,
                    name=name,
                    role=role,
                    streaming_engine=self.streaming_engine,
                    chat_system=self.chat_system,
                    codebase_system=self.codebase_system
                )
                
                await agent.initialize()
                self.agents[agent_id] = agent
                
                # Start the agent
                self.agent_tasks[agent_id] = asyncio.create_task(agent.start())
                
                return {"status": "success", "agent_id": agent_id}
                
            except Exception as e:
                logger.error(f"Error creating agent: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/agents/{agent_id}")
        async def stop_agent(agent_id: str):
            """Stop an agent"""
            if agent_id in self.agents:
                await self.agents[agent_id].stop()
                if agent_id in self.agent_tasks:
                    self.agent_tasks[agent_id].cancel()
                    del self.agent_tasks[agent_id]
                del self.agents[agent_id]
                return {"status": "success"}
            else:
                raise HTTPException(status_code=404, detail="Agent not found")
        
        @self.app.post("/api/process-request")
        async def process_user_request(request_data: dict):
            """Process a user request using the task coordinator"""
            try:
                if not self.task_coordinator:
                    raise HTTPException(status_code=500, detail="Task coordinator not initialized")
                
                user_input = request_data.get("message", "")
                if not user_input:
                    raise HTTPException(status_code=400, detail="Message is required")
                
                # Process the request
                result = await self.task_coordinator.process_user_request(user_input)
                
                # Get system status
                status = await self.task_coordinator.get_system_status()
                
                return {
                    "status": "success",
                    "result": result,
                    "system_status": status
                }
                
            except Exception as e:
                logger.error(f"Error processing user request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/system-status")
        async def get_system_status():
            """Get the current system status"""
            try:
                if not self.task_coordinator:
                    return {"status": "Task coordinator not initialized"}
                
                return await self.task_coordinator.get_system_status()
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/agents")
        async def list_agents():
            """List all agents"""
            return {
                agent_id: agent.get_status()
                for agent_id, agent in self.agents.items()
            }
        
        @self.app.post("/api/files/{file_path:path}/edit")
        async def edit_file(file_path: str, edit_data: dict):
            """Manually edit a file"""
            try:
                edit = await self.streaming_engine.stream_token(
                    agent_id=edit_data.get("agent_id", "manual"),
                    file_path=file_path,
                    position=edit_data["position"],
                    token=edit_data["content"],
                    edit_type=EditType(edit_data.get("edit_type", "insert"))
                )
                return {"status": "success", "edit_id": edit.id}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/files/{file_path:path}")
        async def get_file(file_path: str):
            """Get file content"""
            try:
                content = await self.streaming_engine.get_file_content(file_path)
                return {"content": content}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/chat/send")
        async def send_chat_message(message_data: dict):
            """Send a chat message"""
            try:
                message = await self.chat_system.send_message(
                    agent_id=message_data["agent_id"],
                    content=message_data["content"],
                    message_type=MessageType(message_data.get("message_type", "chat")),
                    file_reference=message_data.get("file_reference"),
                    line_reference=message_data.get("line_reference")
                )
                return {"status": "success", "message_id": message.id}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/chat/history")
        async def get_chat_history(limit: int = 50):
            """Get chat history"""
            messages = self.chat_system.messages[-limit:]
            return {
                "messages": [
                    {
                        "id": msg.id,
                        "agent_id": msg.agent_id,
                        "content": msg.content,
                        "message_type": msg.message_type,
                        "timestamp": msg.timestamp.isoformat(),
                        "file_reference": msg.file_reference,
                        "line_reference": msg.line_reference
                    }
                    for msg in messages
                ]
            }
        
        @self.app.post("/api/search")
        async def search_codebase(search_data: dict):
            """Search the codebase"""
            try:
                results = await self.codebase_system.search_code(
                    query=search_data["query"],
                    file_type=search_data.get("file_type"),
                    max_results=search_data.get("max_results", 10)
                )
                
                return {
                    "results": [
                        {
                            "file_path": result.file_path,
                            "line_number": result.line_number,
                            "content": result.content,
                            "score": result.score,
                            "context_before": result.context_before,
                            "context_after": result.context_after
                        }
                        for result in results
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/index")
        async def reindex_workspace():
            """Reindex the workspace"""
            try:
                await self.codebase_system.index_workspace()
                return {"status": "success"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_websocket_message(self, connection_id: str, message: dict):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        try:
            if message_type == "register_agent":
                agent_id = message["agent_id"]
                self.ws_manager.associate_agent(connection_id, agent_id)
                
                await self.ws_manager.send_to_connection(connection_id, {
                    "type": "registration_success",
                    "agent_id": agent_id
                })
            
            elif message_type == "register_user":
                # User connection registration
                await self.ws_manager.send_to_connection(connection_id, {
                    "type": "user_registered",
                    "connection_id": connection_id
                })
            
            elif message_type == "send_message":
                # User sending a chat message via WebSocket
                chat_message = await self.chat_system.send_message(
                    agent_id=message.get("agent_id", connection_id),
                    content=message["content"],
                    message_type=MessageType(message.get("message_type", "chat")),
                    file_reference=message.get("file_reference"),
                    line_reference=message.get("line_reference")
                )
                
                # Process through task coordinator if available
                if self.task_coordinator and message.get("content"):
                    try:
                        result = await self.task_coordinator.process_user_request(message["content"])
                        await self.ws_manager.send_to_connection(connection_id, {
                            "type": "task_processed",
                            "result": result,
                            "original_message_id": chat_message.id
                        })
                    except Exception as e:
                        logger.error(f"Error processing task: {e}")
                
                await self.ws_manager.send_to_connection(connection_id, {
                    "type": "message_sent",
                    "message_id": chat_message.id
                })
            
            elif message_type == "file_watch":
                # Add connection as watcher for file
                file_path = message["file_path"]
                agent_id = self.ws_manager.connection_agents.get(connection_id, "manual")
                await self.streaming_engine.add_watcher(agent_id, file_path)
                
            elif message_type == "file_unwatch":
                # Remove connection as watcher for file
                file_path = message["file_path"]
                agent_id = self.ws_manager.connection_agents.get(connection_id, "manual")
                await self.streaming_engine.remove_watcher(agent_id, file_path)
            
            elif message_type == "ping":
                await self.ws_manager.send_to_connection(connection_id, {"type": "pong"})
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.ws_manager.send_to_connection(connection_id, {
                "type": "error",
                "message": str(e)
            })
    
    async def _on_file_edit(self, edit: FileEdit):
        """Broadcast file edits to all connected clients"""
        message = {
            "type": "file_edit",
            "edit": {
                "id": edit.id,
                "agent_id": edit.agent_id,
                "file_path": edit.file_path,
                "edit_type": edit.edit_type,
                "position": edit.position,
                "content": edit.content,
                "timestamp": edit.timestamp.isoformat(),
                "line_number": edit.line_number,
                "column_number": edit.column_number
            }
        }
        
        await self.ws_manager.broadcast(message)
    
    async def _on_chat_message(self, message: ChatMessage):
        """Broadcast chat messages to all connected clients"""
        ws_message = {
            "type": "chat_message",
            "message": {
                "id": message.id,
                "agent_id": message.agent_id,
                "content": message.content,
                "message_type": message.message_type,
                "timestamp": message.timestamp.isoformat(),
                "file_reference": message.file_reference,
                "line_reference": message.line_reference
            }
        }
        
        await self.ws_manager.broadcast(ws_message)
    
    def _get_main_html(self) -> str:
        """Return the main HTML interface"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Live Development Environment</title>
    <style>
        body {
            font-family: 'Monaco', 'Consolas', monospace;
            margin: 0;
            padding: 0;
            background: #1e1e1e;
            color: #d4d4d4;
            display: flex;
            height: 100vh;
        }
        
        .container {
            display: flex;
            width: 100%;
            height: 100%;
        }
        
        .editor-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #333;
        }
        
        .chat-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #252526;
        }
        
        .file-tabs {
            background: #2d2d30;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        
        .tab {
            display: inline-block;
            padding: 8px 16px;
            background: #3c3c3c;
            margin-right: 5px;
            cursor: pointer;
            border-radius: 3px 3px 0 0;
        }
        
        .tab.active {
            background: #1e1e1e;
        }
        
        .editor {
            flex: 1;
            padding: 20px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 14px;
            background: #1e1e1e;
            color: #d4d4d4;
            border: none;
            outline: none;
            resize: none;
            white-space: pre;
            overflow: auto;
        }
        
        .chat-header {
            background: #2d2d30;
            padding: 10px;
            border-bottom: 1px solid #333;
            font-weight: bold;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }
        
        .message {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 5px;
            background: #3c3c3c;
        }
        
        .message.system {
            background: #2d4a2d;
        }
        
        .message-header {
            font-size: 12px;
            color: #888;
            margin-bottom: 4px;
        }
        
        .chat-input {
            display: flex;
            padding: 10px;
            border-top: 1px solid #333;
        }
        
        .chat-input input {
            flex: 1;
            padding: 8px;
            background: #3c3c3c;
            border: 1px solid #555;
            color: #d4d4d4;
            border-radius: 3px;
        }
        
        .chat-input button {
            padding: 8px 16px;
            margin-left: 5px;
            background: #0e639c;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
        }
        
        .status-bar {
            background: #007acc;
            color: white;
            padding: 5px 10px;
            font-size: 12px;
        }
        
        .agents-list {
            background: #2d2d30;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        
        .agent {
            padding: 5px;
            margin: 2px 0;
            background: #3c3c3c;
            border-radius: 3px;
            font-size: 12px;
        }
        
        .agent.active {
            background: #2d4a2d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="editor-panel">
            <div class="file-tabs">
                <div class="tab active" id="main-tab">
                    <span id="current-file">main.py</span>
                </div>
                <div class="tab" id="new-tab">+ New File</div>
            </div>
            <textarea class="editor" id="editor" placeholder="// Live collaborative editing..."></textarea>
            <div class="status-bar">
                <span id="status">Connected • Waiting for agents...</span>
            </div>
        </div>
        
        <div class="chat-panel">
            <div class="chat-header">🤖 Multi-Agent Development Environment</div>
            
            <div class="agents-list" id="agents-list">
                <div class="agent">Starting agents...</div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message system">
                    <div class="message-header">🚀 System</div>
                    <div>Welcome to the Multi-Agent Live Development Environment!</div>
                </div>
                <div class="message system">
                    <div class="message-header">💡 How to use</div>
                    <div>Just type what you want! Examples:</div>
                    <div>• "create a Python hello world file"</div>
                    <div>• "create a banana webpage for monkeys"</div>
                    <div>• "help me build a calculator app"</div>
                    <div>• "debug my code"</div>
                </div>
                <div class="message system">
                    <div class="message-header">🤖 Intelligent Agents</div>
                    <div>Agents will automatically join based on your needs!</div>
                </div>
            </div>
            
            <div class="chat-input">
                <input type="text" id="chat-input" placeholder="Type your request: 'create a Python file', 'build a webpage', etc." style="flex: 1;">
                <button onclick="sendMessage()" title="Send request to agents">� Send</button>
            </div>
        </div>
    </div>

    <script>
        let ws;
        let currentFile = 'main.py';
        let connectionId = 'user_' + Math.random().toString(36).substr(2, 9);
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.host;
            ws = new WebSocket(`${protocol}//${host}/ws/${connectionId}`);
            
            ws.onopen = function(event) {
                console.log('Connected to WebSocket');
                updateStatus('Connected • Ready');
                loadFile(currentFile);
                
                // Register as a user connection
                ws.send(JSON.stringify({
                    type: 'register_user',
                    connection_id: connectionId
                }));
            };
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket closed');
                updateStatus('Disconnected');
                setTimeout(connect, 3000); // Reconnect after 3 seconds
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateStatus('Connection Error');
            };
        }
        
        function handleMessage(message) {
            console.log('Received message:', message);
            switch(message.type) {
                case 'connected':
                    console.log('Connection confirmed:', message.connection_id);
                    break;
                case 'user_registered':
                    console.log('User registered:', message.connection_id);
                    break;
                case 'message_sent':
                    console.log('Message sent successfully:', message.message_id);
                    break;
                case 'file_edit':
                    handleFileEdit(message.edit);
                    break;
                case 'chat_message':
                    addChatMessage(message.message);
                    break;
                case 'pong':
                    console.log('Pong received');
                    break;
                case 'error':
                    console.error('Server error:', message.message);
                    updateStatus('Error: ' + message.message);
                    break;
                default:
                    console.warn('Unknown message type:', message.type);
            }
        }
        
        function handleFileEdit(edit) {
            // Apply the edit to the editor if it's the current file
            if (edit.file_path === currentFile) {
                const editor = document.getElementById('editor');
                let currentContent = editor.value;
                
                if (edit.edit_type === 'insert') {
                    const pos = Math.min(edit.position, currentContent.length);
                    const newContent = currentContent.slice(0, pos) + 
                                      edit.content + 
                                      currentContent.slice(pos);
                    editor.value = newContent;
                } else if (edit.edit_type === 'delete') {
                    const pos = Math.min(edit.position, currentContent.length);
                    const deleteLength = Math.min(edit.content.length, currentContent.length - pos);
                    const newContent = currentContent.slice(0, pos) + 
                                      currentContent.slice(pos + deleteLength);
                    editor.value = newContent;
                } else if (edit.edit_type === 'replace') {
                    const pos = Math.min(edit.position, currentContent.length);
                    const deleteLength = Math.min(edit.content.length, currentContent.length - pos);
                    const newContent = currentContent.slice(0, pos) + 
                                      edit.content + 
                                      currentContent.slice(pos + deleteLength);
                    editor.value = newContent;
                }
                
                updateStatus(`Live edit by ${edit.agent_id} in ${edit.file_path}`);
            } else {
                // If editing a different file, offer to switch
                if (edit.edit_type === 'insert' && edit.position === 0) {
                    // This might be a new file being created
                    console.log(`File ${edit.file_path} is being created by ${edit.agent_id}`);
                    if (confirm(`${edit.agent_id} is creating ${edit.file_path}. Switch to view it?`)) {
                        currentFile = edit.file_path;
                        loadFile(currentFile);
                    }
                }
            }
        }
        
        function addChatMessage(message) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${message.message_type}`;
            
            messageDiv.innerHTML = `
                <div class="message-header">${message.agent_id} • ${new Date(message.timestamp).toLocaleTimeString()}</div>
                <div>${message.content}</div>
            `;
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const content = input.value.trim();
            
            if (content && ws && ws.readyState === WebSocket.OPEN) {
                // Send via WebSocket for real-time communication
                ws.send(JSON.stringify({
                    type: 'send_message',
                    agent_id: connectionId,
                    content: content,
                    message_type: 'chat'
                }));
                
                input.value = '';
            } else {
                console.warn('WebSocket not connected, cannot send message');
            }
        }
        
        function loadFile(filename) {
            fetch(`/api/files/${filename}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    document.getElementById('editor').value = data.content || '';
                    document.getElementById('current-file').textContent = filename;
                    currentFile = filename;
                    
                    // Watch this file for changes
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'file_watch',
                            file_path: filename
                        }));
                    }
                })
                .catch(error => {
                    console.error('Error loading file:', error);
                    // File might not exist yet, create empty editor
                    document.getElementById('editor').value = '';
                    document.getElementById('current-file').textContent = filename + ' (new)';
                    currentFile = filename;
                });
        }
        
        function updateStatus(status) {
            document.getElementById('status').textContent = status;
        }
        
        function loadSystemStatus() {
            fetch('/api/system-status')
                .then(response => response.json())
                .then(status => {
                    const agentsList = document.getElementById('agents-list');
                    agentsList.innerHTML = '';
                    
                    if (status.agents && status.agents.agents) {
                        const agents = status.agents.agents;
                        if (agents.length === 0) {
                            agentsList.innerHTML = '<div class="agent">🤖 Agents will join as needed</div>';
                        } else {
                            agents.forEach(agent => {
                                const agentDiv = document.createElement('div');
                                agentDiv.className = `agent ${agent.status === 'active' ? 'active' : ''}`;
                                agentDiv.innerHTML = `
                                    🤖 ${agent.name} (${agent.role})
                                    ${agent.current_task ? ` • Working: ${agent.current_task}` : ' • Ready'}
                                `;
                                agentsList.appendChild(agentDiv);
                            });
                        }
                        
                        // Show task status
                        if (status.tasks) {
                            const taskDiv = document.createElement('div');
                            taskDiv.className = 'agent';
                            taskDiv.innerHTML = `📋 Tasks: ${status.tasks.in_progress} active, ${status.tasks.pending} pending`;
                            agentsList.appendChild(taskDiv);
                        }
                    } else {
                        agentsList.innerHTML = '<div class="agent">🚀 System starting...</div>';
                    }
                })
                .catch(error => {
                    console.error('Error loading system status:', error);
                    const agentsList = document.getElementById('agents-list');
                    agentsList.innerHTML = '<div class="agent">⚠️ System status unavailable</div>';
                });
        }
        
        // Event listeners
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Initialize
        connect();
        
        // Load system status periodically
        setInterval(loadSystemStatus, 3000);
        
        // Initial status load
        setTimeout(loadSystemStatus, 1000);
    </script>
</body>
</html>
        '''
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the collaborative development server"""
        logger.info(f"Starting Multi-Agent Live Development Environment on {host}:{port}")
        
        # Create workspace directory if it doesn't exist
        workspace_path = "./workspace"
        os.makedirs(workspace_path, exist_ok=True)
        
        # Initialize task coordinator
        self.task_coordinator = await get_task_coordinator(
            self.streaming_engine,
            self.chat_system,
            self.codebase_system
        )
        
        # Index the workspace
        await self.codebase_system.index_workspace()
        
        # Start the server
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

# Main entry point
async def main():
    api = CollaborativeAPI()
    await api.start_server()

if __name__ == "__main__":
    asyncio.run(main())
