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

def host_init(session_state, delete=True):
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

