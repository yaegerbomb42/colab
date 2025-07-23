from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid

class AgentRole(str, Enum):
    GENERAL = "general"
    UI = "ui"
    SECURITY = "security"
    LINTER = "linter"
    TEST = "test"
    HELPER = "helper"

class MessageType(str, Enum):
    CHAT = "chat"
    SYSTEM = "system"
    FILE_REFERENCE = "file_reference"
    TASK_UPDATE = "task_update"
    CODE_SUGGESTION = "code_suggestion"

class EditType(str, Enum):
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"

class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: AgentRole
    status: str = "idle"  # idle, thinking, editing, reviewing
    current_file: Optional[str] = None
    current_line: Optional[int] = None
    goals: List[str] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    
class FileEdit(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    file_path: str
    edit_type: EditType
    position: int  # Character position in file
    content: str  # The token/character being added/removed
    timestamp: datetime = Field(default_factory=datetime.now)
    line_number: Optional[int] = None
    column_number: Optional[int] = None

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    message_type: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    file_reference: Optional[str] = None
    line_reference: Optional[int] = None
    parent_message_id: Optional[str] = None

class FileState(BaseModel):
    path: str
    content: str
    last_modified: datetime = Field(default_factory=datetime.now)
    last_modified_by: str
    edit_count: int = 0
    watchers: List[str] = Field(default_factory=list)  # Agent IDs watching this file

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    assigned_agent: Optional[str] = None
    status: str = "open"  # open, in_progress, completed, blocked
    priority: int = 1  # 1-5, 5 being highest
    files_involved: List[str] = Field(default_factory=list)
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class CodeSearchResult(BaseModel):
    file_path: str
    line_number: int
    content: str
    context_before: List[str] = Field(default_factory=list)
    context_after: List[str] = Field(default_factory=list)
    score: float = 0.0

class ProjectContext(BaseModel):
    files: Dict[str, FileState] = Field(default_factory=dict)
    agents: Dict[str, Agent] = Field(default_factory=dict)
    chat_history: List[ChatMessage] = Field(default_factory=list)
    edit_history: List[FileEdit] = Field(default_factory=list)
    tasks: Dict[str, Task] = Field(default_factory=dict)
    project_summary: str = ""
    architecture_notes: List[str] = Field(default_factory=list)
