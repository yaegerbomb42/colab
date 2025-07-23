#!/bin/bash
# Setup script for Multi-Agent Live Development Environment

echo "🚀 Setting up Multi-Agent Live Development Environment..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p workspace
mkdir -p static
mkdir -p logs
mkdir -p chroma_db

# Make scripts executable
chmod +x scripts/start_agents.py
chmod +x main.py

echo "✅ Setup complete!"
echo ""
echo "🏃 Quick start:"
echo "  1. source venv/bin/activate"
echo "  2. python main.py"
echo "  3. Open http://localhost:8000 in your browser"
echo ""
echo "🤖 To start agents:"
echo "  python scripts/start_agents.py --num-agents 3 --duration 5"
