FROM python:3.11-slim

WORKDIR /app

# 1) Install OpenMP runtime and any build essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgomp1 \
      build-essential \
      gcc \
      g++ && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy source & config
COPY src/ ./src/
COPY config/ ./config/

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["-c", "config/public/config.yaml", "-s", "config/private/config.example.yaml"]