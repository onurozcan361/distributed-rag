import subprocess
import weaviate
import json

def get_external_ips(namespace="weaviate"):
    # Run the kubectl command to get services in JSON format
    cmd = ["kubectl", "get", "svc", "-n", namespace, "-o", "json"]
    output = subprocess.check_output(cmd)
    svc_data = json.loads(output)
    
    external_ips = {}
    for svc in svc_data.get("items", []):
        svc_name = svc["metadata"]["name"]
        # Check if the loadBalancer field has an ingress entry
        lb = svc.get("status", {}).get("loadBalancer", {})
        if "ingress" in lb:
            ip = lb["ingress"][0].get("ip", "N/A")
        else:
            ip = "No External IP"
        external_ips[svc_name] = ip
    return external_ips


def get_minikube_ip():
    """
    Retrieves the Minikube IP address using the 'minikube ip' command.
    """
    try:
        result = subprocess.run(["minikube", "ip"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Error fetching Minikube IP:", e)
        return None

def check_cluster(http_host, http_port, grpc_host):
    """
    Creates a Weaviate client for the cluster at the given IP and port,
    then checks if the cluster is ready.
    """

    try:
        client = weaviate.connect_to_custom(
            http_host=http_host,   # The external IP of the weaviate service
            http_port=http_port,                         # The default REST port is 8080
            http_secure=False,                    # Whether to use https (secure) for the HTTP API connection
            grpc_host=grpc_host,       # The external IP of the weaviate-grpc service
            grpc_port=50051,                      # The default gRPC port is 50051
            grpc_secure=False                     # Set to True if the gRPC connection is secure
            )
        # Using the is_ready() method to check if the cluster is ready
        if client.is_ready():
            print(f"Cluster at host {http_host} is ready.")
        else:
            print(f"Cluster at host {http_host} is not ready.")
    except Exception as e:
        print(f"Error connecting to cluster at host {http_host}: {e}")
        
    client.close()

def main():
    minikube_ip = get_minikube_ip()
    if not minikube_ip:
        print("Unable to retrieve Minikube IP. Exiting.")
        return

    print("Minikube IP:", minikube_ip)

    # Define the NodePorts for the 5 clusters
    ips = get_external_ips()
    grpc_host = ips.pop("weaviate-grpc")
    for name, ip in ips.items():
        check_cluster(http_host=ip, http_port=8080, grpc_host=grpc_host)

if __name__ == "__main__":
    main()

