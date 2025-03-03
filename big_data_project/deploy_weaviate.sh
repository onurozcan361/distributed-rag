#!/bin/bash
set -e

# Define the namespace
NAMESPACE="weaviate"

echo "====================================="
echo "Step 1: Create namespace if it doesn't exist"
echo "====================================="
if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
  echo "Creating namespace: $NAMESPACE"
  kubectl create namespace "$NAMESPACE"
else
  echo "Namespace $NAMESPACE already exists."
fi

echo "====================================="
echo "Step 2: Apply Persistent Volume (PV) and Persistent Volume Claim (PVC)"
echo "====================================="
kubectl apply -f weaviate-pv.yaml
kubectl apply -f weaviate-pvc.yaml

echo "====================================="
echo "Step 3: Deploy Weaviate clusters and expose them via Services (REST and gRPC)"
echo "====================================="
for i in {1..5}; do
  echo "Deploying Weaviate Cluster $i"
  kubectl apply -f "./deployment/weaviate-cluster-${i}.yaml" -n "$NAMESPACE"
  kubectl apply -f "./service/weaviate-cluster-${i}-service.yaml" -n "$NAMESPACE"
  kubectl apply -f "./grpc_service/weaviate-cluster-${i}-grpc-service.yaml" -n "$NAMESPACE"
done

echo "====================================="
echo "Step 4: Waiting for pods to become ready"
echo "====================================="
kubectl wait --for=condition=ready pod --all -n "$NAMESPACE" --timeout=120s

echo "====================================="
echo "Step 5: Displaying pod status and Minikube IP"
echo "====================================="
kubectl get pods -n "$NAMESPACE"
echo "Minikube IP: $(minikube ip)"

echo "Note: To assign external IPs to LoadBalancer services (gRPC), run 'minikube tunnel' in a separate terminal."
echo "Deployment complete!"

