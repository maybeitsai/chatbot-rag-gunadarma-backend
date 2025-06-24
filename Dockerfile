# STAGE 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies including Playwright requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Create virtual environment
RUN uv venv --python 3.12

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN . .venv/bin/activate && uv sync --no-cache

# Install Playwright browsers in the virtual environment
RUN . .venv/bin/activate && playwright install chromium
RUN . .venv/bin/activate && playwright install-deps

# STAGE 2: Final Image
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies including Playwright system dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    wget \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and application code
COPY --from=builder /app/.venv .venv
COPY . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PLAYWRIGHT_BROWSERS_PATH="/app/.venv/lib/python3.12/site-packages/playwright/driver"

# Create necessary directories
RUN mkdir -p /app/cache /app/data /app/logs

# Set permissions for cache and data directories
RUN chmod 755 /app/cache /app/data /app/logs

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]