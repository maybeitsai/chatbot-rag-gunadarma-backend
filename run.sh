#!/bin/bash

# Gunadarma RAG Pipeline Runner Script

echo "=== Gunadarma RAG Pipeline ==="
echo "Starting setup and deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with required environment variables."
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

echo "✅ Data files found"

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | less

# Install dependencies with uv
echo "📦 Installing dependencies..."
uv sync

# Run setup
echo "🔧 Setting up database and vector store..."
uv run setup.py

if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully!"
    
    # Start the API server
    echo "🚀 Starting API server..."
    uv run main.py
else
    echo "❌ Setup failed!"
    exit 1
fi