# STAGE 1: Build using the latest stable Rust
FROM rust:bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy manifest files
COPY Cargo.toml Cargo.lock ./

# Step 10: This trick caches your dependencies
# If it fails here, it's usually a version mismatch in Cargo.toml
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

# Copy real source code
COPY src ./src

# Final build
RUN touch src/main.rs && cargo build --release

# STAGE 2: Tiny runtime image
FROM debian:bookworm-slim

WORKDIR /app

# Required for HTTPS requests (reqwest)
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/aether-api .

ENV ROCKET_ADDRESS=0.0.0.0
ENV ROCKET_PORT=8000

CMD ["./aether-api"]