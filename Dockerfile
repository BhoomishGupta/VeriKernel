# ==============================================================================
# VeriKernel: Physics-Aware AI Code Optimization Referee
# Production Docker Container Specification
# ==============================================================================

FROM gcc:13

# Install python3, venv, curl, ca-certificates, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    make \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set container working directory
WORKDIR /app

# Create dedicated python virtual environment
COPY requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

# Copy full application codebase into container
COPY . .

# Compile computational harnesses and verify reference ground truth
RUN make all && \
    (test -f reference_cavity_129.bin || make ref)

# Create dedicated unprivileged user for defense-in-depth container sandboxing
RUN useradd -m -u 1000 verikernel && \
    mkdir -p /app/build /app/cache /tmp/verikernel_run && \
    chown -R verikernel:verikernel /app /opt/venv /tmp/verikernel_run

USER verikernel

# Multi-threaded OpenMP Pinning & Performance Configuration
ENV OMP_PROC_BIND=close
ENV OMP_PLACES=cores
ENV OMP_NUM_THREADS=8
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Strict API Guardrail Defaults (configurable via container env)
ENV IP_HOURLY_LIMIT=5
ENV DAILY_RUNS_LIMIT=30
ENV DAILY_SPEND_LIMIT_USD=5.00
ENV JUDGE_SECRET_TOKEN=verikernel-judge-2026

# Container Healthcheck for Availability Protection
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
