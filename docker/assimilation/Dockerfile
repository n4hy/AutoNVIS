FROM ubuntu:22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++-11 \
    libeigen3-dev \
    libhdf5-dev \
    libboost-all-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY src/assimilation/ ./src/assimilation/

# Build C++ assimilation core
RUN cd src/assimilation && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j$(nproc)

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD test -f /app/src/assimilation/build/assimilation_service || exit 1

# Run assimilation service
CMD ["./src/assimilation/build/assimilation_service"]
