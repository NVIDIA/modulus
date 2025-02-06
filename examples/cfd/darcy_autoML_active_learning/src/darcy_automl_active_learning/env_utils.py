import os
import socket

def is_docker_env_via_dockerenv():
    return os.path.exists('/.dockerenv')

def is_docker_env_via_cgroup():
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            for line in f:
                if 'docker' in line or 'kubepods' in line:
                    return True
    except Exception:
        pass
    return False

def is_docker_env_via_env_vars():
    docker_env_vars = ['DOCKER', 'CONTAINER', 'KUBERNETES_SERVICE_HOST']
    for var in docker_env_vars:
        if os.getenv(var) is not None:
            return True
    return False

def is_docker_env_via_hostname():
    hostname = socket.gethostname()
    return '.' in hostname  # Example heuristic

def is_running_in_docker():
    return (
        is_docker_env_via_dockerenv() or
        is_docker_env_via_cgroup() or
        is_docker_env_via_env_vars() or
        is_docker_env_via_hostname()
    )

if __name__ == "__main__":
    if is_running_in_docker():
        print("Running inside a Docker container.")
    else:
        print("Not running inside a Docker container.")
