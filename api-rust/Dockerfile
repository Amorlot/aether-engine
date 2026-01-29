# STAGE 1: Builder
# Using the full Bookworm image to ensure all build tools are available
FROM rust:bookworm AS builder

# Install system dependencies required for compiling Rust crates (like openssl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Cache Dependencies
# Copy only the configuration files first to cache the heavy download layer
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

# 2. Real Build
# Remove the skeleton and copy the actual source code
RUN rm -rf src
COPY src ./src
# Update the timestamp of main.rs to force a fresh build of the local code
RUN touch src/main.rs && cargo build --release

# ---

# STAGE 2: Runtime
# Using a slim image to keep the final size small (~100MB)
FROM debian:bookworm-slim
WORKDIR /app

# Install runtime dependencies:
# - ca-certificates: Required for HTTPS requests to Ollama/Qdrant
# - libssl3: Required for the compiled binary to run
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/aether-api .

# Copy the static assets for the Web UI
COPY static /app/static

# Rocket configuration for Docker environment
ENV ROCKET_ADDRESS=0.0.0.0
ENV ROCKET_PORT=8000

# Run the backend
CMD ["./aether-api"]