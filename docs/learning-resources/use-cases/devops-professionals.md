---
description: "Infrastructure automation and CI/CD pipeline use cases for DevOps Professionals focused on infrastructure-as-code and automation workflows."
categories: ["infrastructure-operations"]
tags: ["docker", "kubernetes", "monitoring", "launcher", "distributed-training", "enterprise"]
personas: ["devops-focused", "admin-focused"]
difficulty: "advanced"
content_type: "example"
modality: "universal"
---

# DevOps Professionals Use Cases

Infrastructure automation and CI/CD pipeline use cases for DevOps Professionals focused on infrastructure-as-code, automation workflows, and observability with NeMo AutoModel.

:::{note}
**Target Audience**: DevOps Professionals  
**Focus**: Infrastructure-as-code, automation workflows, CI/CD pipelines, observability, containerization, high availability
:::

## Overview

As a DevOps Professional, you need automated infrastructure provisioning, robust CI/CD pipelines, and comprehensive observability for ML training and deployment. NeMo AutoModel provides infrastructure-as-code templates, containerization strategies, and monitoring integrations for production ML operations.

---

## Use Case 1: Infrastructure-as-Code for ML Training Environments

**Context**: Automated provisioning and management of scalable ML training infrastructure using Terraform and Kubernetes.

### NeMo AutoModel Solution

**Terraform Infrastructure Configuration**
```hcl
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

# GPU-enabled EKS cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "nemo-automodel-cluster"
  cluster_version = "1.28"
  
  node_groups = {
    gpu_nodes = {
      desired_capacity = 4
      max_capacity     = 16
      min_capacity     = 2
      
      instance_types = ["p3.8xlarge", "p4d.24xlarge"]
      
      k8s_labels = {
        "node-type" = "gpu"
        "nvidia.com/gpu" = "true"
      }
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "nemo-automodel"
  }
}

# Persistent storage for checkpoints
resource "aws_efs_file_system" "checkpoint_storage" {
  creation_token = "nemo-checkpoints"
  
  tags = {
    Name = "nemo-checkpoint-storage"
  }
}
```

**Kubernetes Deployment Configuration**
```yaml
# k8s/nemo-training-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-automodel-training
  namespace: ml-training
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nemo-automodel
  template:
    metadata:
      labels:
        app: nemo-automodel
    spec:
      nodeSelector:
        node-type: gpu
      
      containers:
      - name: nemo-training
        image: nemo-automodel:latest
        resources:
          requests:
            nvidia.com/gpu: 4
            memory: "64Gi"
            cpu: "16"
          limits:
            nvidia.com/gpu: 4
            memory: "128Gi"
            cpu: "32"
        
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
        
        volumeMounts:
        - name: checkpoint-storage
          mountPath: /checkpoints
        - name: data-storage
          mountPath: /data
          
        command: ["automodel", "finetune", "llm"]
        args: ["-c", "/config/training.yaml"]
        
      volumes:
      - name: checkpoint-storage
        persistentVolumeClaim:
          claimName: checkpoint-pvc
      - name: data-storage
        persistentVolumeClaim:
          claimName: data-pvc
```

---

## Use Case 2: CI/CD Pipeline for ML Model Training and Deployment

**Context**: Automated CI/CD pipeline for model training, validation, and deployment with quality gates.

### NeMo AutoModel Solution

**GitHub Actions Workflow**
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Training and Deployment Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install black flake8 pytest
        pip install -r requirements.txt
    
    - name: Code formatting check
      run: black --check .
    
    - name: Linting
      run: flake8 .
    
    - name: Unit tests
      run: pytest tests/unit/

  model-training:
    needs: code-quality
    runs-on: [self-hosted, gpu]
    steps:
    - uses: actions/checkout@v4
    
    - name: Run training
      run: |
        docker run --gpus all \
          -v ${{ github.workspace }}/data:/data \
          -v ${{ github.workspace }}/checkpoints:/checkpoints \
          nemo-automodel:latest \
          automodel finetune llm -c /config/training.yaml
    
    - name: Validate model
      run: |
        python scripts/validate_model.py \
          --checkpoint /checkpoints/best_model.pt \
          --threshold 0.85

  deploy-staging:
    needs: [model-training]
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to staging
      run: |
        kubectl apply -f k8s/staging/ --namespace=staging

  deploy-production:
    needs: [model-training]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        kubectl apply -f k8s/production/ --namespace=production
```

---

## Use Case 3: Observability and Monitoring Stack

**Context**: Comprehensive observability for ML training with metrics, logging, and alerting.

### NeMo AutoModel Solution

**Prometheus Configuration**
```yaml
# monitoring/prometheus.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'nemo-automodel'
    static_configs:
      - targets: ['nemo-training-service:8000']
    
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['gpu-exporter:9445']

alerting:
  alertmanagers:
  - static_configs:
    - targets: ["alertmanager:9093"]
```

**Alerting Rules**
```yaml
# monitoring/alerts.yml
groups:
- name: ml_training_alerts
  rules:
  - alert: HighGPUTemperature
    expr: nvidia_gpu_temperature_celsius > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "GPU temperature is high"
      
  - alert: TrainingJobFailed
    expr: increase(training_job_failures_total[5m]) > 0
    labels:
      severity: critical
    annotations:
      summary: "Training job failed"
```

---

## Get Started for DevOps Professionals

### Prerequisites
- Kubernetes cluster management experience
- Infrastructure-as-code tools (Terraform, Helm)
- CI/CD pipeline development (GitHub Actions, ArgoCD)
- Monitoring and observability tools (Prometheus, Grafana)

### Development Path
1. **Infrastructure Automation**: Deploy scalable ML infrastructure with Terraform
2. **CI/CD Implementation**: Build automated training and deployment pipelines
3. **Observability Setup**: Implement comprehensive monitoring and alerting
4. **High Availability**: Configure fault tolerance and disaster recovery

### Quick Start
```bash
# Deploy infrastructure
terraform init && terraform apply

# Install monitoring stack
helm install prometheus prometheus-community/kube-prometheus-stack

# Deploy NeMo AutoModel
kubectl apply -f k8s/
```

### Resources
- {doc}`../../guides/deployment/kubernetes` - Kubernetes deployment guide
- {doc}`../../examples/infrastructure` - Infrastructure-as-code examples
- Terraform AWS EKS module documentation

---

**Success Metrics for DevOps Professionals:**
- **Infrastructure Reliability**: 99.9%+ uptime with automated recovery
- **Deployment Automation**: 100% automated deployments with quality gates
- **Observability Coverage**: Complete metrics, logging, and alerting coverage
- **Pipeline Efficiency**: <30 minute end-to-end CI/CD pipeline execution
