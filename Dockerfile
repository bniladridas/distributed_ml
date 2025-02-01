FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    wget \
    build-essential \
    libssl-dev \
    libz-dev \
    pkg-config \
    libeigen3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgrpc++-dev \
    protobuf-compiler-grpc \
    libgtest-dev \
    libomp-dev \
    libpthread-stubs0-dev

# Install spdlog
RUN git clone https://github.com/gabime/spdlog.git /tmp/spdlog \
    && mkdir /tmp/spdlog/build \
    && cd /tmp/spdlog/build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/spdlog

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Build the project
RUN mkdir build && cd build \
    && cmake -DCMAKE_PREFIX_PATH=/usr/local .. \
    && make \
    && make install

# Expose gRPC port
EXPOSE 50051

# Set the default command
CMD ["distributed_ml"]
