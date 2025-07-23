# Multi-Agent Live Development Environment

A collaborative, real-time development platform where multiple autonomous agents work together on a shared codebase using token-streaming editing and contextual chat.

## Features

🔁 **Live Token-Based Editing**
- Agents edit files by streaming tokens one at a time
- Real-time synchronization across all agents
- Character-by-character delete capabilities
- No commits, merges, or file locks

🧱 **Shared File Context**
- Codebase-aware context system
- File summaries and embeddings
- Indexed code search
- Live diffs and change logs
- Agent attribution tracking

💬 **Agent Side Chat**
- Real-time collaboration chat
- File/line reference linking
- Shared conversation history
- Implementation discussion space

🧠 **Agent Memory & Coordination**
- Working memory for goals and tasks
- Project architecture awareness
- Conversation history tracking
- Task assignment and coordination

🔍 **Codebase Search & Awareness**
- File/function/keyword search
- AST-level code analysis
- Documentation and type signatures
- Usage examples

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python main.py
```

3. Open multiple browser tabs to http://localhost:8000 to simulate multiple agents

4. For Llama 1B integration:
```bash
python scripts/start_agents.py --model llama1b --num-agents 3
```

## Architecture

- **Backend**: FastAPI with WebSocket support
- **Real-time**: WebSocket-based token streaming
- **Database**: SQLite with vector embeddings
- **Search**: ChromaDB for semantic code search
- **AI Models**: Transformers with Llama 1B support
- **Frontend**: HTML/JS real-time editor

## Agent Types

- **General Agent**: Full-stack development
- **UI Agent**: Frontend specialization
- **Security Agent**: Security analysis
- **Linter Agent**: Code quality
- **Test Agent**: Testing and validation