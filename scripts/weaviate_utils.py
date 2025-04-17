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

