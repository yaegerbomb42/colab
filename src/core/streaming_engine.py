import asyncio
import json
from typing import Dict, List, Set, Optional, Callable
from datetime import datetime
import logging
from src.core.models import FileEdit, Agent, FileState, EditType
import aiofiles
import os

logger = logging.getLogger(__name__)

class TokenStreamingEngine:
    """
    Core engine for real-time token-based file editing with live synchronization.
    Handles character-by-character streaming edits across multiple agents.
    """
    
    def __init__(self, workspace_path: str = "./workspace"):
        self.workspace_path = workspace_path
        self.file_states: Dict[str, FileState] = {}
        self.active_editors: Dict[str, Set[str]] = {}  # file_path -> set of agent_ids
        self.edit_listeners: List[Callable] = []
        self.edit_history: List[FileEdit] = []
        self._locks: Dict[str, asyncio.Lock] = {}
        
        # Ensure workspace directory exists
        os.makedirs(workspace_path, exist_ok=True)
    
    def add_edit_listener(self, callback: Callable):
        """Add a callback to be notified of all edits"""
        self.edit_listeners.append(callback)
    
    async def _notify_listeners(self, edit: FileEdit):
        """Notify all listeners of a new edit"""
        for listener in self.edit_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(edit)
                else:
                    listener(edit)
            except Exception as e:
                logger.error(f"Error in edit listener: {e}")
    
    async def get_file_lock(self, file_path: str) -> asyncio.Lock:
        """Get or create a lock for a specific file"""
        if file_path not in self._locks:
            self._locks[file_path] = asyncio.Lock()
        return self._locks[file_path]
    
    async def stream_token(self, agent_id: str, file_path: str, position: int, 
                          token: str, edit_type: EditType = EditType.INSERT) -> FileEdit:
        """
        Stream a single token/character edit to a file.
        This is the core method for real-time editing.
        """
        async with await self.get_file_lock(file_path):
            # Load current file state
            if file_path not in self.file_states:
                await self._load_file_state(file_path)
            
            file_state = self.file_states[file_path]
            
            # Calculate line and column numbers
            line_num, col_num = self._get_line_col_from_position(file_state.content, position)
            
            # Create edit record
            edit = FileEdit(
                agent_id=agent_id,
                file_path=file_path,
                edit_type=edit_type,
                position=position,
                content=token,
                line_number=line_num,
                column_number=col_num
            )
            
            # Apply the edit
            new_content = await self._apply_edit(file_state, edit)
            
            # Update file state
            file_state.content = new_content
            file_state.last_modified = datetime.now()
            file_state.last_modified_by = agent_id
            file_state.edit_count += 1
            
            # Save to disk
            await self._save_file_to_disk(file_path, new_content)
            
            # Record edit in history
            self.edit_history.append(edit)
            
            # Notify all listeners (for real-time sync)
            await self._notify_listeners(edit)
            
            return edit
    
    async def _apply_edit(self, file_state: FileState, edit: FileEdit) -> str:
        """Apply a single edit to the file content"""
        content = file_state.content
        position = edit.position
        
        if edit.edit_type == EditType.INSERT:
            # Insert token at position
            new_content = content[:position] + edit.content + content[position:]
        elif edit.edit_type == EditType.DELETE:
            # Delete character(s) at position
            delete_length = len(edit.content) if edit.content else 1
            new_content = content[:position] + content[position + delete_length:]
        elif edit.edit_type == EditType.REPLACE:
            # Replace character(s) at position
            delete_length = len(edit.content) if edit.content else 1
            new_content = content[:position] + edit.content + content[position + delete_length:]
        else:
            raise ValueError(f"Unknown edit type: {edit.edit_type}")
        
        return new_content
    
    def _get_line_col_from_position(self, content: str, position: int) -> tuple[int, int]:
        """Convert character position to line/column numbers"""
        if position > len(content):
            position = len(content)
        
        lines = content[:position].split('\n')
        line_number = len(lines)
        column_number = len(lines[-1]) + 1 if lines else 1
        
        return line_number, column_number
    
    async def _load_file_state(self, file_path: str):
        """Load file state from disk or create new file"""
        full_path = os.path.join(self.workspace_path, file_path.lstrip('/'))
        
        try:
            if os.path.exists(full_path):
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                content = ""
                # Create parent directories
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            content = ""
        
        self.file_states[file_path] = FileState(
            path=file_path,
            content=content,
            last_modified_by="system"
        )
    
    async def _save_file_to_disk(self, file_path: str, content: str):
        """Save file content to disk"""
        full_path = os.path.join(self.workspace_path, file_path.lstrip('/'))
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
                await f.write(content)
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
    
    async def get_file_content(self, file_path: str) -> str:
        """Get current content of a file"""
        if file_path not in self.file_states:
            await self._load_file_state(file_path)
        return self.file_states[file_path].content
    
    async def clear_file(self, file_path: str):
        """Clear a file's content"""
        if file_path not in self.file_states:
            await self._load_file_state(file_path)
        
        # Clear the file content
        self.file_states[file_path].content = ""
        
        # Save empty file to disk
        await self._save_file_to_disk(file_path, "")
    
    async def add_watcher(self, agent_id: str, file_path: str):
        """Add an agent as a watcher of a file"""
        if file_path not in self.file_states:
            await self._load_file_state(file_path)
        
        if agent_id not in self.file_states[file_path].watchers:
            self.file_states[file_path].watchers.append(agent_id)
        
        if file_path not in self.active_editors:
            self.active_editors[file_path] = set()
        self.active_editors[file_path].add(agent_id)
    
    async def remove_watcher(self, agent_id: str, file_path: str):
        """Remove an agent as a watcher of a file"""
        if file_path in self.file_states:
            if agent_id in self.file_states[file_path].watchers:
                self.file_states[file_path].watchers.remove(agent_id)
        
        if file_path in self.active_editors:
            self.active_editors[file_path].discard(agent_id)
    
    async def get_live_diff(self, file_path: str, from_edit_id: Optional[str] = None) -> List[FileEdit]:
        """Get live diff of edits since a specific edit ID"""
        if from_edit_id is None:
            return [edit for edit in self.edit_history if edit.file_path == file_path]
        
        # Find the starting point
        start_index = 0
        for i, edit in enumerate(self.edit_history):
            if edit.id == from_edit_id:
                start_index = i + 1
                break
        
        return [edit for edit in self.edit_history[start_index:] if edit.file_path == file_path]
    
    async def undo_edit(self, edit_id: str, agent_id: str) -> bool:
        """Undo a specific edit (with rate limiting)"""
        # Find the edit
        edit_to_undo = None
        for edit in reversed(self.edit_history):
            if edit.id == edit_id and edit.agent_id == agent_id:
                edit_to_undo = edit
                break
        
        if not edit_to_undo:
            return False
        
        # Create reverse edit
        if edit_to_undo.edit_type == EditType.INSERT:
            reverse_edit = FileEdit(
                agent_id=agent_id,
                file_path=edit_to_undo.file_path,
                edit_type=EditType.DELETE,
                position=edit_to_undo.position,
                content=edit_to_undo.content
            )
        elif edit_to_undo.edit_type == EditType.DELETE:
            reverse_edit = FileEdit(
                agent_id=agent_id,
                file_path=edit_to_undo.file_path,
                edit_type=EditType.INSERT,
                position=edit_to_undo.position,
                content=edit_to_undo.content
            )
        else:
            return False  # Can't easily undo replace operations
        
        # Apply the reverse edit
        await self.stream_token(
            agent_id=reverse_edit.agent_id,
            file_path=reverse_edit.file_path,
            position=reverse_edit.position,
            token=reverse_edit.content,
            edit_type=reverse_edit.edit_type
        )
        
        return True
