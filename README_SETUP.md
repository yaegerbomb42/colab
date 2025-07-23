# Multi-Agent Live Development Environment Setup

## Quick Start

1. **Start the server:**
   ```bash
   python main.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8000` (or use the GitHub Codespaces URL)

3. **Use the interface:**
   - **Create files:** Type "create a python file" or "create main.py and hello world it"
   - **Add agents:** Click the emoji buttons (👨‍💻 General, 🎨 UI, 🔍 Linter)
   - **Chat with agents:** Type your requests in the chat box

## Enhanced AI Integration

The system now supports multiple AI providers for smarter agent responses:

### Option 1: Google Gemini API (Recommended)
1. Get a free API key from [Google AI Studio](https://aistudio.google.com/)
2. Set environment variable:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
3. Restart the server

### Option 2: OpenAI API
1. Get an API key from [OpenAI](https://platform.openai.com/)
2. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
3. Install the package:
   ```bash
   pip install openai
   ```

### Option 3: Local Models
The system automatically falls back to intelligent responses without any setup required.

## Features Fixed

✅ **No more "Adding new functionality" spam** - Agents now work only when requested
✅ **Smart file detection** - Agents check if files actually exist before claiming they do
✅ **Better AI responses** - Context-aware responses based on agent roles
✅ **Improved collaboration** - Agents coordinate better and avoid duplicating work
✅ **Live file preview** - See code being generated in real-time
✅ **WebSocket stability** - Better connection handling for real-time features

## Testing the System

Try these example requests:
- "create main.py and hello world it"
- "create a python file for a calculator"
- "help me with UI design"
- "create an HTML page"

The agents will now provide much more intelligent and helpful responses!

## Troubleshooting

- **502 errors:** Server might be starting up, wait a moment and refresh
- **WebSocket issues:** Check browser console, might need to refresh the page
- **No agent responses:** Make sure the server is running and check the terminal output

## System Architecture

- **Frontend:** Real-time WebSocket interface with live code streaming
- **Backend:** FastAPI with WebSocket support
- **AI:** Multiple provider support (Gemini, OpenAI, Local models)
- **Agents:** Role-based AI agents with collaborative behavior
- **Storage:** File-based workspace with vector search integration
