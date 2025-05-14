# 1. Base image
FROM python:3.11-slim

# 2. Set working dir
WORKDIR /app

# 3. Copy only requirements first (for build caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the code
COPY src/ ./src/
COPY config/ ./config/

# 6. Expose any ports (if you have a web UI; likely none here)
# EXPOSE 8000

# 7. Default entrypoint & CLI
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["-c", "config/config.yaml"]
