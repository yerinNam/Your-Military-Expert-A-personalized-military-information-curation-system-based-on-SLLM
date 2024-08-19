import subprocess

def check_gpu_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

if __name__ == "__main__":
    check_gpu_usage()
