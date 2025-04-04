#!/bin/bash

set -e

# Define the namespace
NAMESPACE="weaviate"


echo "====================================="
echo "Step 1: Create namespace if it doesn't exist"
echo "====================================="
if ! minikube kubectl -- get namespace "$NAMESPACE" >/dev/null 2>&1; then
  echo "Creating namespace: $NAMESPACE"
  minikube kubectl -- create namespace "$NAMESPACE"
else
  echo "Namespace $NAMESPACE already exists."
fi

echo "====================================="
echo "Step 2: Apply Persistent Volume (PV) and Persistent Volume Claim (PVC)"
echo "====================================="
minikube kubectl -- apply -f weaviate-pv.yaml
minikube kubectl -- apply -f weaviate-pvc.yaml

echo "====================================="
echo "Step 3: Deploy Weaviate clusters and expose them via Services (REST and gRPC)"
echo "====================================="
for i in {1..5}; do
  echo "Deploying Weaviate Cluster $i"
  minikube kubectl -- apply -f "./deployment/weaviate-cluster-${i}.yaml" -n "$NAMESPACE"
  minikube kubectl -- apply -f "./service/weaviate-cluster-${i}-service.yaml" -n "$NAMESPACE"
  minikube kubectl -- apply -f "./grpc_service/weaviate-cluster-${i}-grpc-service.yaml" -n "$NAMESPACE"
done

echo "====================================="
echo "Step 4: Waiting for pods to become ready"
echo "====================================="
minikube kubectl -- wait --for=condition=ready pod --all -n "$NAMESPACE" --timeout=120s

echo "====================================="
echo "Step 5: Displaying pod status and Minikube IP"
echo "====================================="
minikube kubectl -- get pods -n "$NAMESPACE"
echo "Minikube IP: $(minikube ip)"

echo "Note: To assign external IPs to LoadBalancer services (gRPC), run 'minikube tunnel' in a separate terminal."
echo "Deployment complete!"

