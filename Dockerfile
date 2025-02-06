FROM python:3.12.8-slim-bookworm

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    openssh-client \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Setup SSH for private repo
RUN mkdir -p -m 0700 /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Create workspace
WORKDIR /workspace

# Clone private repo
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh git clone git@github.com:RSE-UZH/pyasp.git && \
    cd pyasp && \
    python3 -m pip install --upgrade pip && \
    pip3 install setuptools && \
    pip3 install -e . && \  
    pip3 install pytest pytest-mock

# Run tests
# RUN cd pyasp && pytest

# Set entrypoint
ENTRYPOINT ["/bin/bash"]