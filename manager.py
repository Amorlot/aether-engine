import subprocess
import sys
import os

class AetherManager:
    def __init__(self):
        self.container_name = "api-rust"

    def run_cmd(self, cmd):
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing: {cmd}\n{e}")

    def clean(self):
        print("Cleaning locks and fastembed cache...")
        self.run_cmd("find . -name '*.lock' -delete")
        self.run_cmd("rm -rf api-rust/.fastembed_cache")

    def start(self):
        print("Starting services...")
        self.run_cmd("docker compose up -d")

    def stop(self):
        print("Stopping services...")
        self.run_cmd("docker compose down")

    def rebuild(self):
        self.stop()
        self.clean()
        print("Rebuilding API without cache...")
        self.run_cmd("docker compose build --no-cache api-rust")
        self.start()

    def logs(self):
        print(f"Streaming logs for {self.container_name}...")
        self.run_cmd(f"docker compose logs -f {self.container_name}")

    def status(self):
        self.run_cmd("docker compose ps")

if __name__ == "__main__":
    manager = AetherManager()
    
    commands = {
        "start": manager.start,
        "stop": manager.stop,
        "rebuild": manager.rebuild,
        "logs": manager.logs,
        "clean": manager.clean,
        "status": manager.status
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python manager.py [{'|'.join(commands.keys())}]")
    else:
        commands[sys.argv[1]]()