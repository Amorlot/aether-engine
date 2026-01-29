# ==========================================
# STAGE 1: Builder
# Compiles the Rust binary using the official Debian-based image
# ==========================================
FROM rust:bookworm AS builder

# Install system dependencies required for OpenSSL and pkg-config
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ------------------------------------------
# Dependency Caching Layer
# ------------------------------------------
# Copy manifest files only to cache dependencies
COPY Cargo.toml Cargo.lock* ./

# Create a dummy main.rs to force Cargo to build dependencies only
# This layer is cached unless Cargo.toml changes
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

# ------------------------------------------
# Application Build Layer
# ------------------------------------------
# Remove dummy source and copy actual application code
RUN rm -rf src
COPY src ./src

# Update timestamp to force a rebuild of the application logic
# and compile the release binary
RUN touch src/main.rs && cargo build --release

# ==========================================
# STAGE 2: Runtime
# Uses NVIDIA CUDA base image to ensure compatibility with Host GPU Drivers
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

WORKDIR /app

# ------------------------------------------
# Runtime Dependencies
# ------------------------------------------
# - ca-certificates: Essential for HTTPS requests (Reqwest)
# - libssl3: Required by the compiled Rust binary
# - procps: Provides 'top' command for CPU monitoring
# NOTE: 'nvidia-smi' is injected by the NVIDIA Container Toolkit at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------
# Artifacts & Assets
# ------------------------------------------
# Retrieve the compiled binary from the builder stage
COPY --from=builder /app/target/release/aether-api .

# Retrieve static assets (HTML/JS/CSS) for the frontend
COPY static /app/static

# ------------------------------------------
# Configuration & Startup
# ------------------------------------------
# Configure Rocket to listen on all interfaces
ENV ROCKET_ADDRESS=0.0.0.0
ENV ROCKET_PORT=8000

# Expose the port (informative, Docker Compose handles mapping)
EXPOSE 8000

# Launch the Aether Engine API
CMD ["./aether-api"]