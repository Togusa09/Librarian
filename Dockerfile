FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal build deps for some wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY librarian_mcp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY librarian_mcp /app/librarian_mcp
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Defaults that can be overridden at runtime
ENV DOCUMENT_ARCHIVE_PATH=/docs
ENV DATA_PATH=/data
ENV BIND_HOST=0.0.0.0
ENV PORT=8000
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3
ENV OLLAMA_KEEP_ALIVE=30m
ENV OLLAMA_PREWARM=1
ENV OLLAMA_PREWARM_TIMEOUT_SEC=12

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-u", "librarian_mcp/server.py"]
