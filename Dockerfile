# ==========================================
# Build Stage
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip build setuptools wheel

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Build wheel
RUN python -m build --wheel --no-isolation

# ==========================================
# Runtime Stage
# ==========================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Create non-root user
RUN groupadd -r agentsnr && useradd -r -g agentsnr -d /app -s /sbin/nologin agentsnr && \
    chown -R agentsnr:agentsnr /app

# Copy wheel from builder and install
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Switch to non-root user
USER agentsnr

# OCI labels for GitHub Packages repository linking
LABEL org.opencontainers.image.source="https://github.com/Sandeep-Hipparagi/agent-snr" \
      org.opencontainers.image.description="L4/L5 Signal Firewall for AI Agents: context compression, loop entropy circuit breaking, attention budgeting, and skip-level routing" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="Agent SNR" \
      org.opencontainers.image.version="0.1.1"

ENTRYPOINT ["agent-snr"]
CMD ["--help"]