# ==========================================
# STAGE 1: Builder
# ==========================================
FROM rust:bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency Caching
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release

# Application Build
RUN rm -rf src
COPY src ./src
# Copiamo anche static qui nel builder se necessario per macro di inclusione, 
# ma la copia principale serve nel runtime stage
RUN touch src/main.rs && cargo build --release

# ==========================================
# STAGE 2: Runtime
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

WORKDIR /app

# Runtime Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Artifacts & Assets
# Copia il binario compilato
COPY --from=builder /app/target/release/aether-api .

# Copia la cartella static (Assicurati che la cartella 'static' sia nella stessa dir del Dockerfile)
COPY static ./static

# Configurazione Runtime
ENV ROCKET_ADDRESS=0.0.0.0
ENV ROCKET_PORT=8000
# Abilita log dettagliati per errori Rust
ENV RUST_BACKTRACE=1

EXPOSE 8000

CMD ["./aether-api"]