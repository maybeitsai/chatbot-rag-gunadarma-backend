# 🤖 Chatbot RAG Gunadarma Backend

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.12+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](docker-compose.yml)

A high-performance Retrieval-Augmented Generation (RAG) backend API specifically designed for Gunadarma University information system. Features semantic caching, hybrid search, and async processing for fast and accurate responses.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Usage](#-api-usage)
- [Development](#-development)
- [Deployment](#-deployment)
- [Contributing](#-contributing)

## ✨ Features

- **🔍 Hybrid Search**: Combines semantic and keyword search for better accuracy
- **⚡ Semantic Caching**: Automatic caching for similar queries using embedding similarity
- **📊 Vector Database**: PostgreSQL with vector extensions for efficient similarity search
- **🌐 FastAPI Backend**: High-performance REST API with auto-generated documentation
- **💬 WebSocket Support**: Real-time chat functionality
- **🕷️ Smart Crawler**: Intelligent web crawling with content extraction
- **🐳 Docker Ready**: Complete containerization for easy deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 15+
- Docker & Docker Compose (recommended)

### 1. Clone & Setup

```bash
git clone https://github.com/maybeitsai/chatbot-rag-gunadarma-backend.git
cd chatbot-rag-gunadarma-backend

# Using UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

uv sync
```

### 2. Start Services

```bash
# Start database services
docker-compose up -d postgres redis

# Run the application
uv run python main.py
```

### 3. Test the API

```bash
curl http://localhost:8000/api/v1/health
```

Access API documentation at: http://localhost:8000/docs

## 🛠️ Installation

### Option 1: Using UV (Recommended)

UV is a fast Python package manager. This project is configured to work with UV.

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# or for Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install project dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Option 2: Using Standard Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies from pyproject.toml
pip install -e .
```

### Database Setup

```bash
# Using Docker (recommended)
docker-compose up -d postgres redis

# Manual PostgreSQL setup (if not using Docker)
createdb chatbot_rag
psql -d chatbot_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
# Server
HOST=0.0.0.0
PORT=8000
RELOAD=true

# Database
DATABASE_URL=postgresql://chatbot_user:chatbot_password@localhost:5432/chatbot_rag

# Optional Redis Cache
REDIS_URL=redis://localhost:6379

# API Keys (add your actual keys)
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key

# RAG Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
ENABLE_CACHE=true

# Logging
LOG_LEVEL=INFO
```

## � API Usage

### Start the Server

```bash
# Development server
uv run python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Core Endpoints

#### Ask Questions

```bash
curl -X POST "http://localhost:8000/api/v1/question" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is Gunadarma University?"}'
```

#### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

#### WebSocket Chat

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/chat");
ws.send(
  JSON.stringify({
    question: "Tell me about computer science program",
    session_id: "user-session-123",
  })
);
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Development

### Project Structure

```
app/
├── api/          # FastAPI application & routes
├── rag/          # RAG pipeline & processing
└── crawl/        # Web crawling modules

scripts/          # Utility scripts
tests/            # Test suite
data/             # Crawled content & datasets
cache/            # Semantic cache storage
logs/             # Application logs
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking (if using mypy)
mypy app/
```

## 🐳 Deployment

### Docker Deployment

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale the application
docker-compose up -d --scale app=3
```

### Production Environment

```bash
# Set production variables
export RELOAD=false
export LOG_LEVEL=WARNING

# Run with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables for Production

```env
RELOAD=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://user:pass@prod-db:5432/chatbot_rag
REDIS_URL=redis://prod-redis:6379
CORS_ORIGINS=["https://your-frontend-domain.com"]
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Install dependencies: `uv sync --dev`
4. Make your changes and add tests
5. Run tests: `pytest`
6. Format code: `black . && ruff check .`
7. Commit: `git commit -m 'Add new feature'`
8. Push: `git push origin feature/new-feature`
9. Create a Pull Request

### Commit Convention

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring

## 🔧 Troubleshooting

### Common Issues

**Database Connection Error**

```bash
# Check if PostgreSQL is running
docker-compose ps postgres
docker-compose logs postgres
```

**Import Errors**

```bash
# Reinstall dependencies
uv sync --reinstall
```

**Slow Performance**

- Enable Redis caching: `ENABLE_CACHE=true`
- Reduce chunk size: `CHUNK_SIZE=300`
- Check memory usage: `docker stats`

## 📊 Performance

- **Response Time**: < 2 seconds for cached queries
- **Throughput**: > 100 requests/second
- **Cache Hit Rate**: > 70% for common queries

## � License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**Made with ❤️ for Gunadarma University**

[⬆ Back to top](#-chatbot-rag-gunadarma-backend)

</div>
