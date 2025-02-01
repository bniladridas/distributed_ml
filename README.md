# Distributed Machine Learning Application

## Overview
This is a C++ distributed machine learning application designed to run on Kubernetes with automated scaling and monitoring.

## Prerequisites
- CMake 3.15+
- C++17 Compiler
- gRPC
- Protobuf
- OpenMP
- Eigen3
- Kubernetes Cluster
- Prometheus

## Build Instructions
```bash
mkdir build && cd build
cmake ..
make
```

## Kubernetes Deployment
```bash
# Build Docker image
docker build -t distributed-ml:latest .

# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml
```

## Monitoring
Prometheus configuration is provided in `monitoring/prometheus.yaml`. 

## Architecture
- Core ML Model: Extensible base for different machine learning algorithms
- Distributed Trainer: Handles parameter synchronization across workers
- Kubernetes Deployment: Automated scaling and high availability
- Prometheus Monitoring: Real-time performance tracking

## Performance Metrics
- Horizontal Pod Autoscaler configured to scale based on CPU utilization
- Minimum 3 replicas
- Maximum 10 replicas
- Scale-up threshold: 70% CPU usage

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
