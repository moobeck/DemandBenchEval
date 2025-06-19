FROM python:3.11-slim

WORKDIR /app

# Accept GitHub token as build argument
ARG GITHUB_TOKEN

# Install OpenMP runtime, build essentials, Git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgomp1 \
      build-essential \
      gcc \
      g++ \
      git && \
    rm -rf /var/lib/apt/lists/*

# Configure Git to use GitHub token for authentication
RUN if [ -n "$GITHUB_TOKEN" ]; then \
      git config --global url."https://${GITHUB_TOKEN}:@github.com/".insteadOf "https://github.com/"; \
    fi

# Clone the repository
RUN git clone https://github.com/DataDog/toto.git /app/toto

# Install dependencies of the cloned repository
RUN pip install --no-cache-dir -r /app/toto/requirements.txt

# Install Python dependencies from our app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source & config
COPY src/ ./src/
COPY config/ ./config/

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["-c", "config/public/config.yaml", "-s", "config/private/config.example.yaml"]
