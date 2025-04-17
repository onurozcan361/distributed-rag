import subprocess
import weaviate
import json

def get_external_ips(namespace="weaviate"):
    # Run the kubectl command to get services in JSON format
    cmd = ["minikube", "kubectl", "--", "get", "svc", "-n", namespace, "-o", "json"]
    output = subprocess.check_output(cmd)
    svc_info = {}
    svc_data = json.loads(output)
    for svc in svc_data.get("items", []):
        svc_spec = svc.get('spec')
        ip = svc_spec.get('clusterIP', 'N/A')
        port, protocol = svc_spec['ports'][0]['port'], svc_spec['ports'][0]['protocol']

        if 'name' in svc_spec['ports'][0]:
            name = svc_spec['ports'][0]['name']
        else:
            name = svc_spec.get('selector', {}).get('app', 'unknown')

        svc_info[name] = {
            'ip': ip,
            'port': port,
            'protocol': protocol,
            'name' : name
        }


    return svc_info



def close_all_clients():
    services = get_external_ips()
    grpc_host = services["grpc"]["ip"]

    for service in services.values():
        if service['name'] != "grpc":
            try:
                client = weaviate.connect_to_custom(
                    http_host=service['ip'],
                    http_port=8080,
                    http_secure=False,
                    grpc_host=grpc_host,
                    grpc_port=50051,
                    grpc_secure=False
                )
                client.close()
            except Exception as e:
                print(f"Error closing client for {service['ip']}: {e}")

def host_init(session_state, delete=False):
    services = get_external_ips()

    session_state.grpc_host = services["grpc"]["ip"]
    session_state.client_ips = []

    for service in services.values():
                if service['name'] != "grpc":
                    session_state.client_ips.append(service['ip'])
                    print(f"Service IP: {service['ip']}")
                
                if delete:
                    try:
                        client = weaviate.connect_to_custom(
                            http_host=service['ip'],
                            http_port=8080,
                            http_secure=False,
                            grpc_host=session_state.grpc_host,
                            grpc_port=50051,
                            grpc_secure=False
                            )
                        

                        for i in range(5):
                            client.collections.delete(f"dist_data_{i}")
                        client.close()
                    except Exception as e:
                        print(f"err -> {service['ip']}: {e}") 


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

def check_cluster(http_host, http_port, grpc_host, grpc_port):
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
    

def main():
    minikube_ip = get_minikube_ip()
    if not minikube_ip:
        print("Unable to retrieve Minikube IP. Exiting.")
        return

    print("Minikube IP:", minikube_ip)

    # Define the NodePorts for the 5 clusters
    ips = get_external_ips()
    grpc_host = ips.pop("grpc")
    for name, svc in ips.items():
        check_cluster(http_host=svc['ip'], http_port=svc['port'], grpc_host=grpc_host['ip'], grpc_port=grpc_host['port'])

if __name__ == "__main__":
    main()

